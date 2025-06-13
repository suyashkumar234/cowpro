"""
Training the model with UNet backbone
Extended from original implementation of PANet by Wang et al.
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

from models.grid_proto_fewshot_unet2 import FewShotSeg
from dataloaders.dev_customized_med import med_fewshot
from dataloaders.GenericSuperDatasetv2 import SuperpixelDataset
from dataloaders.dataset_utils import DATASET_INFO
import dataloaders.augutils as myaug

from util.utils import set_seed, t2n, to01, compose_wt_simple, get_tversky_loss
from util.metric import Metric

from config_ssl_upload_unet import ex
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

    _log.info('###### Create UNet-based model ######')

    # Create model with UNet backbone
    model = FewShotSeg(pretrained_path=None, cfg=_config['model'])
    
    model = model.cuda()
    model.train()

    _log.info('###### Load data ######')
    ### Training set
    data_name = _config['dataset']
    if data_name == 'SABS_Superpix':
        baseset_name = 'SABS'
    elif data_name == 'C0_Superpix':
        raise NotImplementedError
        baseset_name = 'C0'
    elif data_name == 'CHAOST2_Superpix':
        baseset_name = 'CHAOST2'
    else:
        raise ValueError(f'Dataset: {data_name} not found')

    ###================== Transforms for data augmentation =============================== ###
    tr_transforms = myaug.transform_with_label({'aug': myaug.augs[_config['which_aug']]})
    ### ================================================================================== ###
    assert _config['scan_per_load'] < 0 # by default we load the entire dataset directly

    test_labels = DATASET_INFO[baseset_name]['LABEL_GROUP']['pa_all'] - DATASET_INFO[baseset_name]['LABEL_GROUP'][_config["label_sets"]]
    _log.info(f'###### Labels excluded in training : {[lb for lb in _config["exclude_cls_list"]]} ######')
    _log.info(f'###### Unseen labels evaluated in testing: {[lb for lb in test_labels]} ######')

    tr_parent = SuperpixelDataset( # base dataset
        which_dataset = baseset_name,
        base_dir=_config['path'][data_name]['data_dir'],
        idx_split = _config['eval_fold'],
        mode='train',
        min_fg=str(_config["min_fg_data"]), # dummy entry for superpixel dataset
        transforms =tr_transforms, #
        transform_param_limits = myaug.augs[_config['which_aug']],#tr_transforms, #
        nsup = _config['task']['n_shots'],
        scan_per_load = _config['scan_per_load'],
        exclude_list = _config["exclude_cls_list"],
        superpix_scale = _config["superpix_scale"],
        fix_length = _config["max_iters_per_load"] if (data_name == 'C0_Superpix') or (data_name == 'CHAOST2_Superpix') else None,
        dataset_config = _config['DATASET_CONFIG']
    )

    ### dataloaders
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
        print("Using SGD optimizer for UNet")
        optimizer = torch.optim.SGD(model.parameters(), **_config['optim'])
    elif _config['optim_type'] == 'adam':
        print("Using ADAM optimizer for UNet")
        optimizer = torch.optim.Adam(model.parameters(), **_config['optim'])
    else:
        raise NotImplementedError

    scheduler = MultiStepLR(optimizer, milestones=_config['lr_milestones'],  gamma = _config['lr_step_gamma'])

    my_weight = compose_wt_simple(_config["use_wce"], data_name)
    criterion = nn.CrossEntropyLoss(ignore_index=_config['ignore_label'], weight = my_weight)

    i_iter = 0 # total number of iteration
    n_sub_epoches = _config['n_steps'] // _config['max_iters_per_load'] # number of times for reloading

    log_loss = {'loss': 0, 'align_loss': 0}

    _log.info('###### Training UNet-based Few-Shot Segmentation ######')
    stime = time.time()
    for sub_epoch in range(n_sub_epoches):
        _log.info(f'###### This is epoch {sub_epoch} of {n_sub_epoches} epoches ######')
        for _, sample_batched in enumerate(trainloader):
            # Prepare input
            i_iter += 1
            # add writers
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
            
            mean = [m for m in sample_batched["mean"]]
            std = [s for s in sample_batched["std"]]
            
            ########################################################################
            # Forward pass with UNet encoder-decoder
            query_pred, align_loss, debug_vis, assign_mats = model(support_images,
                                                                    support_fg_mask, 
                                                                    support_bg_mask, 
                                                                    query_images, 
                                                                    isval = False, val_wsize = None)
            ########################################################################

            # Compute loss with Tversky loss for better handling of class imbalance
            query_loss = criterion(query_pred, query_labels) + get_tversky_loss(
                query_pred.argmax(dim = 1, keepdim = True), 
                query_labels[None, ...], 
                _config['tversky_params']['tversky_alpha'], 
                _config['tversky_params']['tversky_beta'],
                _config['tversky_params']['tversky_gamma']
            )
            
            loss = query_loss + align_loss
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()

            # Log loss
            query_loss = query_loss.detach().data.cpu().numpy()
            align_loss = align_loss.detach().data.cpu().numpy() if align_loss != 0 else 0

            _run.log_scalar('loss', query_loss)
            _run.log_scalar('align_loss', align_loss)
            _run.log_scalar('learning_rate', scheduler.get_last_lr()[0])
            
            log_loss['loss'] += query_loss
            log_loss['align_loss'] += align_loss

            # print loss and take snapshots
            if (i_iter + 1) % _config['print_interval'] == 0:

                nt = time.time()

                loss = log_loss['loss'] / _config['print_interval']
                align_loss = log_loss['align_loss'] / _config['print_interval']

                log_loss['loss'] = 0
                log_loss['align_loss'] = 0

                # Enhanced visualization
                fig, ax = plt.subplots(2, 3, figsize=(15, 10))
                
                # Support image and mask
                si = (support_images[0][0][0].cpu()*std[0]+mean[0]).numpy().transpose((1,2,0))
                si = (si - si.min())/(si.max() - si.min() + 1e-6)
                ax[0,0].imshow(si)
                ax[0,0].set_title('Support Image')
                
                sm = support_fg_mask[0][0][0].cpu().numpy()
                ax[0,0].imshow(np.stack([np.zeros(sm.shape),sm,np.zeros(sm.shape)], axis = 2), alpha = 0.3)

                # Query image, ground truth, and prediction
                qi = (query_images[0][0].cpu()*std[0]+mean[0]).numpy().transpose((1,2,0))
                qi = (qi - qi.min())/(qi.max() - qi.min() + 1e-6)
                ax[0,1].imshow(qi)
                ax[0,1].set_title('Query Image')
                
                qm = query_labels[0].cpu().numpy()
                ax[0,1].imshow(np.stack([np.zeros(qm.shape),qm,np.zeros(qm.shape)], axis = 2), alpha = 0.3)
                
                qp = query_pred.argmax(dim = 1).float().cpu().numpy()[0]
                ax[0,2].imshow(qi)
                ax[0,2].set_title('Query Prediction')
                ax[0,2].imshow(np.stack([qp,np.zeros(qp.shape),np.zeros(qp.shape)], axis = 2), alpha = 0.4)
                
                # Probability maps
                prob_fg = torch.softmax(query_pred, dim=1)[0,1].cpu().numpy()
                prob_bg = torch.softmax(query_pred, dim=1)[0,0].cpu().numpy()
                
                ax[1,0].imshow(prob_bg, cmap='Blues')
                ax[1,0].set_title('Background Probability')
                ax[1,1].imshow(prob_fg, cmap='Reds') 
                ax[1,1].set_title('Foreground Probability')
                
                # Assignment visualization if available
                if len(assign_mats) > 0 and assign_mats[0] is not None:
                    assign_vis = assign_mats[0][0].cpu().numpy()
                    ax[1,2].imshow(assign_vis, cmap='viridis')
                    ax[1,2].set_title('Prototype Assignment')
                else:
                    ax[1,2].axis('off')

                plt.tight_layout()
                plt.savefig(os.path.join(f'{_run.observers[0].dir}/trainsnaps', f'{i_iter + 1}.png'), 
                           bbox_inches='tight', dpi=100)
                plt.close(fig)

                print(f'step {i_iter+1}: loss: {loss:.4f}, align_loss: {align_loss:.4f}, '
                      f'lr: {scheduler.get_last_lr()[0]:.6f}, time: {(nt-stime)/60:.2f} mins')

            if (i_iter + 1) % _config['save_snapshot_every'] == 0:
                _log.info('###### Taking snapshot ######')
                torch.save({'model':model.state_dict(),'opt':optimizer.state_dict(),'sch':scheduler.state_dict()},
                           os.path.join(f'{_run.observers[0].dir}/snapshots', f'{i_iter + 1}.pth'))

            if data_name == 'C0_Superpix' or data_name == 'CHAOST2_Superpix':
                if (i_iter + 1) % _config['max_iters_per_load'] == 0:
                    _log.info('###### Reloading dataset ######')
                    trainloader.dataset.reload_buffer()
                    print(f'###### New dataset with {len(trainloader.dataset)} slices has been loaded ######')

            if (i_iter - 2) > _config['n_steps']:
                return 1 # finish up