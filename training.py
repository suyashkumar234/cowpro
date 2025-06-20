"""
Training the model
Extended from original implementation of PANet by Wang et al.
"""
"""
Training the model with Wandb integration and periodic evaluation
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
import wandb

from models.grid_proto_fewshot import FewShotSeg
from dataloaders.dev_customized_med import med_fewshot, update_loader_dset, med_fewshot_val 
from dataloaders.GenericSuperDatasetv2 import SuperpixelDataset
from dataloaders.dataset_utils import DATASET_INFO, get_normalize_op
import dataloaders.augutils as myaug

from util.utils import set_seed, t2n, to01, compose_wt_simple, get_tversky_loss
from util.metric import Metric

from config_ssl_upload import ex
import tqdm
import json, copy, ast
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# config pre-trained model caching path
os.environ['TORCH_HOME'] = "./pretrained_model"

# Maximum labels for different datasets
max_labels = {
    'CHAOST2': 4,
    'SABS': 13,
    'C0': 3
}

def evaluate_model(model, max_label, client_data, te_dataset, te_parent, test_labels, _config, run=None, round=0, client=True, save_client_preds=False):
    """
    Evaluation function for the model
    """
    model.eval()
    
    # Setup metric tracking
    mar_val_metric_nodes = {}
    _scan_ids = te_parent.scan_ids
    
    with torch.no_grad():
        save_pred_buffer_ = {} # indexed by class
        
        for sup_idx in range(len(_scan_ids)):
            print(f'###### Set validation nodes for support id {_scan_ids[sup_idx]} ######')
            mar_val_metric_nodes[_scan_ids[sup_idx]] = Metric(max_label=max_label, n_scans=len(te_dataset.dataset.pid_curr_load) - _config['task']['n_shots'])
            mar_val_metric_nodes[_scan_ids[sup_idx]].reset()

            save_pred_buffer = {}

            for curr_lb in test_labels:
                te_dataset.set_curr_cls(curr_lb)
                support_batched = te_parent.get_support(curr_class=curr_lb, 
                                                        class_idx=[curr_lb], 
                                                        scan_idx=[sup_idx], 
                                                        npart=_config['task']['npart'])

                support_images = [[shot.cuda() for shot in way]
                                    for way in support_batched['support_images']]
                
                suffix = 'mask'
                support_fg_mask = [[shot[f'fg_{suffix}'].float().cuda() for shot in way]
                                    for way in support_batched['support_mask']]
                support_bg_mask = [[shot[f'bg_{suffix}'].float().cuda() for shot in way]
                                    for way in support_batched['support_mask']]

                curr_scan_count = -1
                _lb_buffer = {}

                # Create test dataloader
                testloader = DataLoader(
                    te_dataset,
                    batch_size=1,
                    shuffle=False,
                    num_workers=0,
                    pin_memory=False,
                    drop_last=False
                )

                for sample_batched in testloader:
                    _scan_id = sample_batched["scan_id"][0]
                    if _scan_id in te_parent.potential_support_sid:
                        continue
                        
                    if sample_batched["is_start"]:
                        ii = 0
                        curr_scan_count += 1
                        _scan_id = sample_batched["scan_id"][0]
                        outsize = te_dataset.dataset.info_by_scan[_scan_id]["array_size"]
                        outsize = (256, 256, outsize[0])
                        _pred = np.zeros(outsize, dtype=np.float32)
                        _pred.fill(np.nan)
                        _labels = np.zeros(outsize, dtype=np.float32)

                    q_part = sample_batched["part_assign"]
                    query_images = [sample_batched['image'].cuda()]
                    query_labels = torch.cat([sample_batched['label'].cuda()], dim=0)

                    sup_img_part = [[shot_tensor.unsqueeze(0) for shot_tensor in support_images[0][q_part]]]
                    sup_fgm_part = [[shot_tensor.unsqueeze(0) for shot_tensor in support_fg_mask[0][q_part]]]
                    sup_bgm_part = [[shot_tensor.unsqueeze(0) for shot_tensor in support_bg_mask[0][q_part]]]

                    query_pred, _, _, _ = model(sup_img_part, sup_fgm_part, sup_bgm_part, query_images, 
                                              isval=True, val_wsize=_config["val_wsize"])

                    query_pred = query_pred.argmax(dim=1).float().cpu().numpy()
                    _pred[..., ii] = query_pred[0].copy()
                    _labels[..., ii] = query_labels[0].detach().cpu().clone().numpy()

                    if (sample_batched["z_id"] - sample_batched["z_max"] <= _config['z_margin']) and \
                       (sample_batched["z_id"] - sample_batched["z_min"] >= -1 * _config['z_margin']):
                        mar_val_metric_nodes[_scan_ids[sup_idx]].record(_pred[..., ii], _labels[..., ii], 
                                                                       labels=[curr_lb], n_scan=curr_scan_count)

                    ii += 1
                    
                    if sample_batched["is_end"]:
                        _lb_buffer[_scan_id] = [_pred.transpose(2,0,1), _labels.transpose(2,0,1)]

                save_pred_buffer[str(curr_lb)] = _lb_buffer
            save_pred_buffer_[_scan_ids[sup_idx]] = save_pred_buffer

    # Compute metrics
    all_dice_scores = []
    all_prec_scores = []
    all_rec_scores = []
    
    for i in range(len(_scan_ids)):
        print(f"========== Metrics for Support Scan {_scan_ids[i]} ============")
        m_classDice, _, m_meanDice, _, m_rawDice = mar_val_metric_nodes[_scan_ids[i]].get_mDice(
            labels=sorted(test_labels), n_scan=None, give_raw=True)
        
        m_classPrec, _, m_meanPrec, _, m_classRec, _, m_meanRec, _, m_rawPrec, m_rawRec = \
            mar_val_metric_nodes[_scan_ids[i]].get_mPrecRecall(labels=sorted(test_labels), n_scan=None, give_raw=True)
        
        all_dice_scores.append(m_meanDice)
        all_prec_scores.append(m_meanPrec)
        all_rec_scores.append(m_meanRec)
        
        print(f'Dice: {m_meanDice:.4f}, Precision: {m_meanPrec:.4f}, Recall: {m_meanRec:.4f}')
        
        # Log to wandb if run is provided
        if run is not None:
            wandb.log({
                f"eval/dice_scan_{_scan_ids[i]}": m_meanDice,
                f"eval/precision_scan_{_scan_ids[i]}": m_meanPrec,
                f"eval/recall_scan_{_scan_ids[i]}": m_meanRec,
                "eval/round": round
            })

    # Compute overall metrics
    overall_dice = np.mean(all_dice_scores)
    overall_prec = np.mean(all_prec_scores)
    overall_rec = np.mean(all_rec_scores)
    
    print(f"========== Overall Metrics ============")
    print(f'Overall Dice: {overall_dice:.4f}')
    print(f'Overall Precision: {overall_prec:.4f}')
    print(f'Overall Recall: {overall_rec:.4f}')
    
    # Log overall metrics to wandb
    if run is not None:
        wandb.log({
            "eval/overall_dice": overall_dice,
            "eval/overall_precision": overall_prec,
            "eval/overall_recall": overall_rec,
            "eval/round": round
        })
    
    model.train()
    return overall_dice, overall_prec, overall_rec


@ex.automain
def main(_run, _config, _log):
    # Initialize wandb
    wandb.init(
        project=_config.get('wandb_project', 'medical-segmentation'),
        name=_config.get('wandb_run_name', f"train_{_config['dataset']}_fold_{_config['eval_fold']}"),
        config=_config,
        tags=[_config['dataset'], f"fold_{_config['eval_fold']}"]
    )
    
    if _run.observers:
        os.makedirs(f'{_run.observers[0].dir}/snapshots', exist_ok=True)
        os.makedirs(f'{_run.observers[0].dir}/trainsnaps', exist_ok=True)
        for source_file, _ in _run.experiment_info['sources']:
            os.makedirs(os.path.dirname(f'{_run.observers[0].dir}/source/{source_file}'),
                        exist_ok=True)
            _run.observers[0].save_file(source_file, f'source/{source_file}')
        shutil.rmtree(f'{_run.observers[0].basedir}/_sources')
        with open("config.json", "w") as f:
            json.dump(_config, f, indent=4)

    set_seed(_config['seed'])
    cudnn.enabled = True
    cudnn.benchmark = True
    torch.cuda.set_device(device=_config['gpu_id'])
    torch.set_num_threads(1)

    _log.info('###### Create model ######')
    model = FewShotSeg(pretrained_path=None, cfg=_config['model'])
    model = model.cuda()
    model.train()

    # Watch model with wandb
    wandb.watch(model, log_freq=100)

    _log.info('###### Load data ######')
    data_name = _config['dataset']
    if data_name == 'CHAOST2_Superpix':
        baseset_name = 'CHAOST2'
    elif data_name == 'C0_Superpix':
        raise NotImplementedError
        baseset_name = 'C0'
    elif data_name == 'SABS':
        baseset_name = 'SABS'
    elif data_name == 'SABS_Superpix':
        baseset_name = 'SABS'
    elif data_name == 'FLARE22Train_Superpix':
        baseset_name = 'FLARE22Train'
    else:
        raise ValueError(f'Dataset: {data_name} not found')

    # Get test labels
    test_labels = DATASET_INFO[baseset_name]['LABEL_GROUP']['pa_all'] - DATASET_INFO[baseset_name]['LABEL_GROUP'][_config["label_sets"]]
    _log.info(f'###### Labels excluded in training : {[lb for lb in _config["exclude_cls_list"]]} ######')
    _log.info(f'###### Unseen labels evaluated in testing: {[lb for lb in test_labels]} ######')

    # Setup transforms
    tr_transforms = myaug.transform_with_label({'aug': myaug.augs[_config['which_aug']]})
    assert _config['scan_per_load'] < 0

    # Create training dataset
    dataset, tr_parent = med_fewshot(
        dataset_name=baseset_name,
        base_dir=_config['path'][data_name]['data_dir'],
        idx_split=_config['eval_fold'],
        mode='train',
        scan_per_load=_config['scan_per_load'],
        transforms=tr_transforms,
        act_labels=DATASET_INFO[baseset_name]['LABEL_GROUP']['pa_all'],
        n_ways=1,
        n_shots=1,
        nsup=_config['task']['n_shots'],
        fix_parent_len=_config["max_iters_per_load"] if _config['fix_length'] else None,
        max_iters_per_load=_config["max_iters_per_load"],
        min_fg=str(_config["min_fg_data"]),
        n_queries=1,
        exclude_list=_config["exclude_cls_list"],
        dataset_config=_config['DATASET_CONFIG'],
        client_eval=True if _config.get('client_eval', False) else False
    )
    dataset.norm_func = tr_parent.norm_func

    # Create training dataloader
    trainloader = DataLoader(
        dataset,
        batch_size=_config['batch_size'],
        shuffle=True,
        num_workers=_config['num_workers'],
        pin_memory=True,
        drop_last=True
    )

    # Setup evaluation dataset (reuse the same norm_func)
    _log.info('###### Setting up evaluation dataset ######')
    if baseset_name == 'SABS':
        norm_func = trainloader.dataset.norm_func
    else:
        norm_func = get_normalize_op(modality='MR', fids=None)

    te_dataset, te_parent = med_fewshot_val(
        dataset_name=baseset_name,
        base_dir=_config['path'][baseset_name]['data_dir'],
        idx_split=_config['eval_fold'],
        scan_per_load=_config['scan_per_load'],
        act_labels=test_labels,
        npart=_config['task']['npart'],
        nsup=_config['task']['n_shots'],
        extern_normalize_func=norm_func,
        client_eval=True
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

    my_weight = compose_wt_simple(_config["use_wce"], data_name)
    criterion = nn.CrossEntropyLoss(ignore_index=_config['ignore_label'], weight=my_weight)

    i_iter = 0
    epoch = 0
    n_sub_epoches = _config['n_steps'] // _config['max_iters_per_load']
    log_loss = {'loss': 0, 'align_loss': 0}
    
    # Evaluation frequency
    eval_frequency = 10  # Evaluate every 10 epochs
    best_dice = 0.0

    _log.info('###### Training ######')
    stime = time.time()
    
    for sub_epoch in range(n_sub_epoches):
        epoch = sub_epoch
        _log.info(f'###### This is epoch {sub_epoch} of {n_sub_epoches} epoches ######')
        
        for _, sample_batched in enumerate(trainloader):
            i_iter += 1
            
            # Prepare input
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
            
            # Forward pass
            query_pred, align_loss, debug_vis, assign_mats = model(support_images,
                                                                    support_fg_mask, 
                                                                    support_bg_mask, 
                                                                    query_images, 
                                                                    isval=False, val_wsize=None)

            # Compute loss
            query_loss = criterion(query_pred, query_labels) + get_tversky_loss(
                query_pred.argmax(dim=1, keepdim=True), query_labels[None, ...], 0.3, 0.7, 1.0)
            loss = query_loss + align_loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Log loss
            query_loss_val = query_loss.detach().data.cpu().numpy()
            align_loss_val = align_loss.detach().data.cpu().numpy() if align_loss != 0 else 0

            # Log to wandb
            wandb.log({
                "train/query_loss": query_loss_val,
                "train/align_loss": align_loss_val,
                "train/total_loss": query_loss_val + align_loss_val,
                "train/learning_rate": scheduler.get_last_lr()[0],
                "train/epoch": epoch,
                "train/iteration": i_iter
            })

            _run.log_scalar('loss', query_loss_val)
            _run.log_scalar('align_loss', align_loss_val)
            log_loss['loss'] += query_loss_val
            log_loss['align_loss'] += align_loss_val

            # Print loss and take snapshots
            if (i_iter + 1) % _config['print_interval'] == 0:
                nt = time.time()

                loss_avg = log_loss['loss'] / _config['print_interval']
                align_loss_avg = log_loss['align_loss'] / _config['print_interval']

                log_loss['loss'] = 0
                log_loss['align_loss'] = 0

                # Create training visualization
                fig, ax = plt.subplots(1, 2, figsize=(12, 6))
                
                si = (support_images[0][0][0].cpu() * std[0] + mean[0]).numpy().transpose((1, 2, 0))
                si = (si - si.min()) / (si.max() - si.min() + 1e-6)
                ax[0].imshow(si)
                
                sm = support_fg_mask[0][0][0].cpu().numpy()
                ax[0].imshow(np.stack([np.zeros(sm.shape), sm, np.zeros(sm.shape)], axis=2), alpha=0.3)
                ax[0].set_title('Support Image + Mask')

                qi = (query_images[0][0].cpu() * std[0] + mean[0]).numpy().transpose((1, 2, 0))
                qi = (qi - qi.min()) / (qi.max() - qi.min() + 1e-6)
                ax[1].imshow(qi)
                
                qm = query_labels[0].cpu().numpy()
                ax[1].imshow(np.stack([np.zeros(qm.shape), qm, np.zeros(qm.shape)], axis=2), alpha=0.3)
                
                qp = query_pred.argmax(dim=1).float().cpu().numpy()[0]
                ax[1].imshow(np.stack([qp, np.zeros(qp.shape), np.zeros(qp.shape)], axis=2), alpha=0.2)
                ax[1].set_title('Query Image + GT + Pred')

                # Log image to wandb
                wandb.log({
                    "train/visualization": wandb.Image(fig),
                    "train/iteration": i_iter
                })

                plt.savefig(os.path.join(f'{_run.observers[0].dir}/trainsnaps', f'{i_iter + 1}.png'), 
                           bbox_inches='tight')
                plt.close(fig)

                print(f'step {i_iter+1}: loss: {loss_avg:.4f}, align_loss: {align_loss_avg:.4f}, time: {(nt-stime)/60:.2f} mins')

            # Save snapshots
            if (i_iter + 1) % _config['save_snapshot_every'] == 0:
                _log.info('###### Taking snapshot ######')
                checkpoint = {
                    'model': model.state_dict(),
                    'opt': optimizer.state_dict(),
                    'sch': scheduler.state_dict(),
                    'iteration': i_iter,
                    'epoch': epoch
                }
                torch.save(checkpoint, os.path.join(f'{_run.observers[0].dir}/snapshots', f'{i_iter + 1}.pth'))

            # Reload dataset if needed
            if data_name == 'C0_Superpix' or data_name == 'CHAOST2_Superpix':
                if (i_iter + 1) % _config['max_iters_per_load'] == 0:
                    _log.info('###### Reloading dataset ######')
                    update_loader_dset(trainloader, tr_parent)
                    print(f'###### New dataset with {len(trainloader.dataset)} slices has been loaded ######')

            if (i_iter - 2) > _config['n_steps']:
                break
        
        # Evaluate every eval_frequency epochs
        if (epoch + 1) % eval_frequency == 0:
            _log.info(f'###### Evaluating at epoch {epoch + 1} ######')
            print(f"################### BUILDING EVALUATION DATASET ######################")
            print(f"############## VALIDATING {baseset_name} ###################")
            
            # Run evaluation
            overall_dice, overall_prec, overall_rec = evaluate_model(
                model=model,
                max_label=max_labels[baseset_name],
                client_data=baseset_name,
                te_dataset=te_dataset,
                te_parent=te_parent,
                test_labels=test_labels,
                _config=_config,
                run=wandb,
                round=epoch + 1,
                client=True,
                save_client_preds=_config.get('save_plots', False)
            )
            
            # Save best model
            if overall_dice > best_dice:
                best_dice = overall_dice
                _log.info(f'###### New best dice score: {best_dice:.4f} ######')
                
                # Save best model
                best_checkpoint = {
                    'model': model.state_dict(),
                    'opt': optimizer.state_dict(),
                    'sch': scheduler.state_dict(),
                    'iteration': i_iter,
                    'epoch': epoch,
                    'best_dice': best_dice
                }
                torch.save(best_checkpoint, os.path.join(f'{_run.observers[0].dir}/snapshots', 'best_model.pth'))
                
                # Log to wandb
                wandb.log({
                    "eval/best_dice": best_dice,
                    "eval/best_epoch": epoch + 1
                })

    # Final evaluation
    _log.info('###### Final evaluation ######')
    final_dice, final_prec, final_rec = evaluate_model(
        model=model,
        max_label=max_labels[baseset_name],
        client_data=baseset_name,
        te_dataset=te_dataset,
        te_parent=te_parent,
        test_labels=test_labels,
        _config=_config,
        run=wandb,
        round=n_sub_epoches,
        client=True,
        save_client_preds=_config.get('save_plots', False)
    )
    
    # Log final metrics
    wandb.log({
        "final/dice": final_dice,
        "final/precision": final_prec,
        "final/recall": final_rec
    })
    
    # Save final model
    final_checkpoint = {
        'model': model.state_dict(),
        'opt': optimizer.state_dict(),
        'sch': scheduler.state_dict(),
        'iteration': i_iter,
        'epoch': n_sub_epoches,
        'final_dice': final_dice
    }
    torch.save(final_checkpoint, os.path.join(f'{_run.observers[0].dir}/snapshots', 'final_model.pth'))
    
    wandb.finish()
    return 1
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
from dataloaders.dev_customized_med import med_fewshot, update_loader_dset, med_fewshot_val 
from dataloaders.GenericSuperDatasetv2 import SuperpixelDataset
from dataloaders.dataset_utils import DATASET_INFO
import dataloaders.augutils as myaug

from util.utils import set_seed, t2n, to01, compose_wt_simple, get_tversky_loss
from util.metric import Metric

from config_ssl_upload import ex
import tqdm
import json, copy, ast
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
        with open("config.json", "w") as f:
            json.dump(_config, f, indent=4)


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
    ### Training set
    data_name = _config['dataset']
    if data_name == 'CHAOST2_Superpix':
        baseset_name = 'CHAOST2'
    elif data_name == 'C0_Superpix':
        raise NotImplementedError
        baseset_name = 'C0'
    elif data_name == 'SABS':  # ADD THIS FOR SUPERVISED LEARNING
        baseset_name = 'SABS'
    elif data_name == 'SABS_Superpix':
        baseset_name = 'SABS'
    elif data_name=='FLARE22Train_Superpix':
        baseset_name='FLARE22Train'
    else:
        raise ValueError(f'Dataset: {data_name} not found')
    

    ###================== Transforms for data augmentation =============================== ###
    tr_transforms = myaug.transform_with_label({'aug': myaug.augs[_config['which_aug']]})
    ### ================================================================================== ###
    assert _config['scan_per_load'] < 0 # by default we load the entire dataset directly

    test_labels = DATASET_INFO[baseset_name]['LABEL_GROUP']['pa_all'] - DATASET_INFO[baseset_name]['LABEL_GROUP'][_config["label_sets"]]
    _log.info(f'###### Labels excluded in training : {[lb for lb in _config["exclude_cls_list"]]} ######')
    _log.info(f'###### Unseen labels evaluated in testing: {[lb for lb in test_labels]} ######')

    

    tr_transforms_supervised = None 

    # Create supervised dataset
    dataset, tr_parent = med_fewshot(
    dataset_name = baseset_name,
    base_dir=_config['path'][data_name]['data_dir'],
    idx_split = _config['eval_fold'],
    mode='train',
    scan_per_load = _config['scan_per_load'],
    transforms = tr_transforms,
    act_labels = DATASET_INFO[baseset_name]['LABEL_GROUP']['pa_all'],
    n_ways = 1,
    n_shots = 1,
    nsup = _config['task']['n_shots'],
    fix_parent_len = _config["max_iters_per_load"] if _config['fix_length'] else None,
    max_iters_per_load=_config["max_iters_per_load"],
    min_fg=str(_config["min_fg_data"]),
    n_queries=1,
    exclude_list = _config["exclude_cls_list"],
    dataset_config = _config['DATASET_CONFIG'],
    client_eval = True if _config.get('client_eval', False) else False
)
    dataset.norm_func = tr_parent.norm_func

# Update the dataloader creation
    trainloader = DataLoader(
    dataset,  # Use the paired dataset instead of tr_parent
    batch_size=_config['batch_size'],
    shuffle=True,
    num_workers=_config['num_workers'],
    pin_memory=True,
    drop_last=True
)

    '''tr_parent = SuperpixelDataset( # base dataset
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
        #num_workers=_config['num_workers'],
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )'''

    _log.info('###### Set optimizer ######')
    if _config['optim_type'] == 'sgd':
        print("Using SGD optimizer")
        optimizer = torch.optim.SGD(model.parameters(), **_config['optim'])
    elif _config['optim_type'] == 'adam':
        print("Using ADAM optimizer")
        optimizer = torch.optim.Adam(model.parameters(), **_config['optim'])
    else:
        raise NotImplementedError

    # optimizer.load_state_dict(torch.load(_config['reload_model_path'])['opt'])

    scheduler = MultiStepLR(optimizer, milestones=_config['lr_milestones'],  gamma = _config['lr_step_gamma'])
    # scheduler.load_state_dict(torch.load(_config['reload_model_path'])['sch'])

    my_weight = compose_wt_simple(_config["use_wce"], data_name)
    criterion = nn.CrossEntropyLoss(ignore_index=_config['ignore_label'], weight = my_weight)

    i_iter = 0 # total number of iteration
    n_sub_epoches = _config['n_steps'] // _config['max_iters_per_load'] # number of times for reloading

    log_loss = {'loss': 0, 'align_loss': 0}

    _log.info('###### Training ######')
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
            # FIXME: in the model definition, filter out the failure case where 
            # pseudolabel falls outside of image or too small to calculate a prototype
            # try:
            

            mean = [m for m in sample_batched["mean"]]
            std = [s for s in sample_batched["std"]]
            ########################################################################
            query_pred, align_loss, debug_vis, assign_mats = model(support_images,
                                                                    support_fg_mask, 
                                                                    support_bg_mask, 
                                                                    query_images, 
                                                                    isval = False, val_wsize = None)
            ########################################################################
            # except:
                # print('Faulty batch detected, skip')
                # continue

            query_loss = criterion(query_pred, query_labels) + get_tversky_loss(query_pred.argmax(dim = 1, keepdim = True), query_labels[None, ...], 0.3, 0.7 ,1.0)
            loss = query_loss + align_loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Log loss
            query_loss = query_loss.detach().data.cpu().numpy()
            align_loss = align_loss.detach().data.cpu().numpy() if align_loss != 0 else 0

            _run.log_scalar('loss', query_loss)
            _run.log_scalar('align_loss', align_loss)
            log_loss['loss'] += query_loss
            log_loss['align_loss'] += align_loss

            # print loss and take snapshots
            if (i_iter + 1) % _config['print_interval'] == 0:

                nt = time.time()

                loss = log_loss['loss'] / _config['print_interval']
                align_loss = log_loss['align_loss'] / _config['print_interval']

                log_loss['loss'] = 0
                log_loss['align_loss'] = 0

                fig, ax = plt.subplots(1,2)
                # print(mean.shape, std.shape, mean, std)            
                # print(support_images[0][0][0].min(), support_images[0][0][0].max())
                si = (support_images[0][0][0].cpu()*std[0]+mean[0]).numpy().transpose((1,2,0))
                si = (si - si.min())/(si.max() - si.min() + 1e-6)
                # print(si.shape)
                ax[0].imshow(si)
                # print(support_fg_mask[0][0][0].shape)
                sm = support_fg_mask[0][0][0].cpu().numpy()
                # print(sm.shape)
                ax[0].imshow(np.stack([np.zeros(sm.shape),sm,np.zeros(sm.shape)], axis = 2), alpha = 0.3)

                # print(query_pred[0].shape)
                qi = (query_images[0][0].cpu()*std[0]+mean[0]).numpy().transpose((1,2,0))
                qi = (qi - qi.min())/(qi.max() - qi.min() + 1e-6)
                ax[1].imshow(qi)
                qm = query_labels[0].cpu().numpy()
                # print(sm.shape, qm.shape)
                ax[1].imshow(np.stack([np.zeros(qm.shape),qm,np.zeros(qm.shape)], axis = 2), alpha = 0.3)
                qp = query_pred.argmax(dim = 1).float().cpu().numpy()[0] #(query_pred > 0.5)[0][0].cpu().numpy() #.transpose((1,2,0))
                # print(qp.shape)
                ax[1].imshow(np.stack([qp,np.zeros(qp.shape),np.zeros(qp.shape)], axis = 2), alpha = 0.2)

                plt.savefig(os.path.join(f'{_run.observers[0].dir}/trainsnaps', f'{i_iter + 1}.png'), bbox_inches='tight')
                plt.close(fig)

                print(f'step {i_iter+1}: loss: {loss}, align_loss: {align_loss}, time: {(nt-stime)/60} mins')

            if (i_iter + 1) % _config['save_snapshot_every'] == 0:
                _log.info('###### Taking snapshot ######')
                torch.save({'model':model.state_dict(),'opt':optimizer.state_dict(),'sch':scheduler.state_dict()},
                           os.path.join(f'{_run.observers[0].dir}/snapshots', f'{i_iter + 1}.pth'))
                

            if data_name == 'C0_Superpix' or data_name == 'CHAOST2_Superpix':
                if (i_iter + 1) % _config['max_iters_per_load'] == 0:
                    _log.info('###### Reloading dataset ######')
                    #trainloader.dataset.reload_buffer()
                    update_loader_dset(trainloader, tr_parent)
                    print(f'###### New dataset with {len(trainloader.dataset)} slices has been loaded ######')

            if (i_iter - 2) > _config['n_steps']:
                return 1 # finish up
                """
