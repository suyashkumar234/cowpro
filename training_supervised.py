"""
Supervised Training Script for CoWPro Model - FIXED VERSION
Modified from the original self-supervised training to use real ground truth masks
"""
import os
import shutil
import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import torch.backends.cudnn as cudnn
import numpy as np

from models.grid_proto_fewshot import FewShotSeg
from dataloaders.SupervisedDataset import SupervisedDataset
from dataloaders.dataset_utils import DATASET_INFO
import dataloaders.augutils as myaug

from util.utils import set_seed, compose_wt_simple, get_tversky_loss
from util.metric import Metric

# FIXED: Import from config_supervised instead of config_ssl_upload
from config_ssl_upload import ex
import tqdm
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# config pre-trained model caching path
os.environ['TORCH_HOME'] = "./pretrained_model"

@ex.automain
def main(_run, _config, _log):
    if _run.observers:
        os.makedirs(f'{_run.observers[0].dir}/snapshots', exist_ok=True)
        os.makedirs(f'{_run.observers[0].dir}/trainsnaps', exist_ok=True)
        for source_file, _ in _run.experiment_info['sources']:
            os.makedirs(os.path.dirname(f'{_run.observers[0].dir}/source/{source_file}'),
                        exist_ok=True)
            _run.observers[0].save_file(source_file, f'source/{source_file}')
        shutil.rmtree(f'{_run.observers[0].basedir}/_sources')

    set_seed(_config['seed'])
    cudnn.enabled = True
    cudnn.benchmark = True
    torch.cuda.set_device(device=_config['gpu_id'])
    torch.set_num_threads(1)

    _log.info('###### Create model ######')
    model = FewShotSeg(pretrained_path=None, cfg=_config['model'])
    model = model.cuda()
    model.train()

    _log.info('###### Load data ######')
    data_name = _config['dataset']
    
    # Map dataset names to base names
    if data_name.endswith('_Supervised'):
        baseset_name = data_name.replace('_Supervised', '')
    else:
        baseset_name = data_name
        
    if baseset_name == 'SABS':
        base_dataset_name = 'SABS'
    elif baseset_name == 'CHAOST2':
        base_dataset_name = 'CHAOST2'
    else:
        raise ValueError(f'Dataset: {data_name} not found')

    # Transforms for data augmentation
    tr_transforms = myaug.augs[_config['which_aug']]
    
    test_labels = DATASET_INFO[base_dataset_name]['LABEL_GROUP']['pa_all'] - \
                  DATASET_INFO[base_dataset_name]['LABEL_GROUP'][_config["label_sets"]]
    
    _log.info(f'###### Labels excluded in training : {[lb for lb in _config["exclude_cls_list"]]} ######')
    _log.info(f'###### Unseen labels evaluated in testing: {[lb for lb in test_labels]} ######')

    # Create supervised dataset
    tr_parent = SupervisedDataset(
        which_dataset=base_dataset_name,
        base_dir=_config['path'][base_dataset_name]['data_dir'],
        idx_split=_config['eval_fold'],
        mode='train',
        transform_param_limits=tr_transforms,
        scan_per_load=_config['scan_per_load'],
        nsup=_config['task']['n_shots'],
        exclude_list=_config["exclude_cls_list"],
        fix_length=_config.get("max_iters_per_load"),
        min_slice_distance=_config.get('min_slice_distance', 4),
        max_distance_ratio=_config.get('max_distance_ratio', 1/6)
    )

    # DataLoader
    trainloader = DataLoader(
        tr_parent,
        batch_size=_config['batch_size'],
        shuffle=True,
        num_workers=_config['num_workers'],
        pin_memory=True,
        drop_last=True
    )

    _log.info('###### Set optimizer ######')
    if _config['optim_type'] == 'sgd':
        print("Using SGD optimizer")
        optimizer = torch.optim.SGD(model.parameters(), **_config['optim'])
    elif _config['optim_type'] == 'adam':
        print("Using ADAM optimizer")
        optimizer = torch.optim.Adam(model.parameters(), **_config['optim'])
    else:
        raise NotImplementedError

    scheduler = MultiStepLR(optimizer, milestones=_config['lr_milestones'], gamma=_config['lr_step_gamma'])

    my_weight = compose_wt_simple(_config["use_wce"], base_dataset_name)
    criterion = nn.CrossEntropyLoss(ignore_index=_config['ignore_label'], weight=my_weight)

    i_iter = 0
    n_sub_epoches = _config['n_steps'] // _config['max_iters_per_load']
    
    log_loss = {'loss': 0, 'align_loss': 0}

    _log.info('###### Training ######')
    stime = time.time()
    
    for sub_epoch in range(n_sub_epoches):
        _log.info(f'###### This is epoch {sub_epoch} of {n_sub_epoches} epoches ######')
        
        for batch_idx, sample_batched in enumerate(trainloader):
            i_iter += 1
            
            # Prepare input data
            support_images = [[shot.float().cuda() for shot in way]
                              for way in sample_batched['support_images']]
            support_fg_mask = [[shot[f'fg_mask'].float().cuda() for shot in way]
                               for way in sample_batched['support_mask']]
            support_bg_mask = [[shot[f'bg_mask'].float().cuda() for shot in way]
                               for way in sample_batched['support_mask']]

            query_images = [query_image.float().cuda()
                            for query_image in sample_batched['query_images']]
            query_labels = torch.cat(
                [query_label.long().cuda() for query_label in sample_batched['query_labels']], dim=0)

            optimizer.zero_grad()
            
            # Forward pass
            query_pred, align_loss, debug_vis, assign_mats = model(
                support_images,
                support_fg_mask, 
                support_bg_mask, 
                query_images, 
                isval=False, 
                val_wsize=None
            )

            # Compute losses
            query_loss = criterion(query_pred, query_labels)
            
            # Add Tversky loss for better handling of class imbalance
            tversky_loss = get_tversky_loss(
                query_pred.argmax(dim=1, keepdim=True), 
                query_labels[None, ...], 
                0.3, 0.7, 1.0
            )
            
            loss = query_loss + tversky_loss + align_loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Log losses
            query_loss_val = query_loss.detach().data.cpu().numpy()
            align_loss_val = align_loss.detach().data.cpu().numpy() if align_loss != 0 else 0
            tversky_loss_val = tversky_loss.detach().data.cpu().numpy()

            _run.log_scalar('loss', query_loss_val)
            _run.log_scalar('align_loss', align_loss_val)
            _run.log_scalar('tversky_loss', tversky_loss_val)
            
            log_loss['loss'] += query_loss_val
            log_loss['align_loss'] += align_loss_val

            # Print loss and save snapshots
            if (i_iter + 1) % _config['print_interval'] == 0:
                nt = time.time()
                
                loss_avg = log_loss['loss'] / _config['print_interval']
                align_loss_avg = log_loss['align_loss'] / _config['print_interval']
                
                log_loss['loss'] = 0
                log_loss['align_loss'] = 0

                # Create visualization
                try:
                    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
                    
                    # Support image and mask
                    support_img = support_images[0][0][0].cpu().numpy().transpose((1, 2, 0))
                    support_img = (support_img - support_img.min()) / (support_img.max() - support_img.min() + 1e-6)
                    axes[0, 0].imshow(support_img)
                    axes[0, 0].set_title('Support Image')
                    axes[0, 0].axis('off')
                    
                    support_mask = support_fg_mask[0][0][0].cpu().numpy()
                    axes[0, 1].imshow(support_mask, cmap='gray')
                    axes[0, 1].set_title('Support Mask')
                    axes[0, 1].axis('off')
                    
                    # Query image and prediction
                    query_img = query_images[0][0].cpu().numpy().transpose((1, 2, 0))
                    query_img = (query_img - query_img.min()) / (query_img.max() - query_img.min() + 1e-6)
                    axes[1, 0].imshow(query_img)
                    axes[1, 0].set_title('Query Image')
                    axes[1, 0].axis('off')
                    
                    # Overlay prediction and ground truth
                    query_pred_mask = query_pred.argmax(dim=1).float().cpu().numpy()[0]
                    query_gt_mask = query_labels[0].cpu().numpy()
                    
                    # Create overlay
                    overlay = np.zeros((*query_pred_mask.shape, 3))
                    overlay[query_gt_mask == 1] = [0, 1, 0]  # Green for GT
                    overlay[query_pred_mask == 1] = [1, 0, 0]  # Red for prediction
                    overlay[(query_gt_mask == 1) & (query_pred_mask == 1)] = [1, 1, 0]  # Yellow for overlap
                    
                    axes[1, 1].imshow(overlay)
                    axes[1, 1].set_title('Prediction (Red) vs GT (Green)')
                    axes[1, 1].axis('off')
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(f'{_run.observers[0].dir}/trainsnaps', f'{i_iter + 1}.png'), 
                              bbox_inches='tight', dpi=150)
                    plt.close(fig)
                except Exception as e:
                    print(f"Visualization error: {e}")

                print(f'step {i_iter+1}: loss: {loss_avg:.4f}, align_loss: {align_loss_avg:.4f}, '
                      f'tversky_loss: {tversky_loss_val:.4f}, time: {(nt-stime)/60:.2f} mins')
                      
                # Log additional metrics
                print(f'Support scan: {sample_batched["support_scan_id"][0]}, '
                      f'slice: {sample_batched["support_z_id"][0].item()}')
                print(f'Query scan: {sample_batched["query_scan_id"][0]}, '
                      f'slice: {sample_batched["query_z_id"][0].item()}')
                print(f'Target class: {sample_batched["target_class"][0].item()}')
                print(f'Distance: {abs(sample_batched["support_z_id"][0].item() - sample_batched["query_z_id"][0].item())} slices')

            if (i_iter + 1) % _config['save_snapshot_every'] == 0:
                _log.info('###### Taking snapshot ######')
                torch.save({
                    'model': model.state_dict(),
                    'opt': optimizer.state_dict(),
                    'sch': scheduler.state_dict(),
                    'iter': i_iter + 1
                }, os.path.join(f'{_run.observers[0].dir}/snapshots', f'{i_iter + 1}.pth'))

            if (i_iter - 2) > _config['n_steps']:
                return 1

    return 1
