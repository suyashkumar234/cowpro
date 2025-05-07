"""
Validation script
"""
import os
import shutil
import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import torch.backends.cudnn as cudnn
from torchvision import transforms
import numpy as np
import pandas as pd
import time
from itertools import product

from models.grid_proto_fewshot_updated import IterativeAdaptiveRefinement,FewShotSeg

from dataloaders.dev_customized_med import med_fewshot_val
from dataloaders.ManualAnnoDatasetv2 import ManualAnnoDataset
from dataloaders.GenericSuperDatasetv2 import SuperpixelDataset
from dataloaders.dataset_utils import DATASET_INFO, get_normalize_op
from dataloaders.niftiio import convert_to_sitk
import dataloaders.augutils as myaug

from util.metric import Metric

from config_ssl_upload import ex

import matplotlib.pyplot as plt
from PIL import Image
import imageio

import tqdm
import SimpleITK as sitk
from torchvision.utils import make_grid

# config pre-trained model caching path
os.environ['TORCH_HOME'] = "./pretrained_model"

def run_parameter_tuning(model, testloader, test_dataset, te_parent, test_labels, output_csv):
    """
    Run a parameter tuning experiment to test different refinement configurations
    
    Args:
        model: Model to test
        testloader: Test dataloader
        test_dataset: Test dataset
        te_parent: Parent test dataset
        test_labels: Test class labels
        output_csv: Path to save results CSV
    """
    print("\n=== Starting Parameter Tuning Experiment ===")
    print(f"Results will be saved to: {output_csv}")
    
    
    # Define parameter grid to search
    param_grid = {
        "refinement_iterations": [2, 3, 5, 7],
        "use_feedback": [True, False],
        "small_object_threshold": [0.005, 0.01, 0.02],
        "large_object_threshold": [0.15, 0.2, 0.25],
        "small_kernel_size": [3, 5],
        "medium_kernel_size": [3, 5],
        "large_kernel_size": [5, 7]
    }
    
    # Save original parameters for later restoration
    original_params = {
        "refinement_iterations": model.max_iterations if hasattr(model, 'max_iterations') else 3,
        "use_feedback": model.refinement.use_feedback if hasattr(model, 'refinement') else True,
        "small_object_threshold": model.refinement.small_object_threshold if hasattr(model, 'refinement') else 0.01,
        "large_object_threshold": model.refinement.large_object_threshold if hasattr(model, 'refinement') else 0.2,
        "small_kernel_size": model.refinement.small_kernel_size if hasattr(model, 'refinement') else 3,
        "medium_kernel_size": model.refinement.medium_kernel_size if hasattr(model, 'refinement') else 3,
        "large_kernel_size": model.refinement.large_kernel_size if hasattr(model, 'refinement') else 5
    }
    
    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(product(*param_values))
    
    # Store results
    results = []
    
    # First, evaluate baseline model (with original parameters)
    baseline_metrics = evaluate_parameter_set(model, testloader, test_dataset, te_parent, test_labels)
    baseline_row = {**original_params, **baseline_metrics, "config": "baseline"}
    results.append(baseline_row)
    
    # Save initial results
    pd.DataFrame([baseline_row]).to_csv(output_csv, index=False)
    
    print(f"Baseline Mean Dice: {baseline_metrics['mean_dice']:.4f}")
    
    # Test each parameter combination
    for i, combination in enumerate(param_combinations):
        # Set parameters for this run
        params = {name: value for name, value in zip(param_names, combination)}
        
        print(f"\nTesting parameter combination {i+1}/{len(param_combinations)}:")
        for param_name, param_value in params.items():
            print(f"  {param_name}: {param_value}")
            
            # Update model parameters
            if param_name == "refinement_iterations":
                model.max_iterations = param_value
                model.refinement.max_iterations = param_value
            elif param_name == "use_feedback":
                model.refinement.use_feedback = param_value
            elif param_name == "small_object_threshold":
                model.refinement.small_object_threshold = param_value
            elif param_name == "large_object_threshold":
                model.refinement.large_object_threshold = param_value
            elif param_name == "small_kernel_size":
                model.refinement.small_kernel_size = param_value
            elif param_name == "medium_kernel_size":
                model.refinement.medium_kernel_size = param_value
            elif param_name == "large_kernel_size":
                model.refinement.large_kernel_size = param_value
        
        # Evaluate with current parameters
        metrics = evaluate_parameter_set(model, testloader, test_dataset, te_parent, test_labels)
        
        # Record results
        result_row = {**params, **metrics, "config": f"combo_{i+1}"}
        results.append(result_row)
        
        # Save after each run
        pd.DataFrame(results).to_csv(output_csv, index=False)
        
        # Show comparison with baseline
        dice_diff = metrics['mean_dice'] - baseline_metrics['mean_dice']
        print(f"Mean Dice: {metrics['mean_dice']:.4f} (Diff from baseline: {dice_diff:+.4f})")
    
    # Find and report best parameters
    df = pd.DataFrame(results)
    best_idx = df["mean_dice"].idxmax()
    best_row = df.iloc[best_idx]
    
    print("\n=== Parameter Tuning Complete ===")
    print(f"Best configuration: {best_row['config']}")
    print(f"Best Mean Dice Score: {best_row['mean_dice']:.4f}")
    print("Best Parameters:")
    for param in param_names:
        if param in best_row:
            print(f"  {param}: {best_row[param]}")
    
    # Restore original parameters
    model.max_iterations = original_params["refinement_iterations"]
    model.refinement.max_iterations = original_params["refinement_iterations"]
    model.refinement.use_feedback = original_params["use_feedback"]
    model.refinement.small_object_threshold = original_params["small_object_threshold"]
    model.refinement.large_object_threshold = original_params["large_object_threshold"]
    model.refinement.small_kernel_size = original_params["small_kernel_size"]
    model.refinement.medium_kernel_size = original_params["medium_kernel_size"]
    model.refinement.large_kernel_size = original_params["large_kernel_size"]
    
    return best_row

def evaluate_parameter_set(model, testloader, test_dataset, te_parent, test_labels):
    """
    Evaluate model with current parameter set
    
    Args:
        model: Model to evaluate
        testloader: Test dataloader
        test_dataset: Test dataset
        te_parent: Parent test dataset
        test_labels: Test class labels
        
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Metrics for each class
    class_dices = {label: [] for label in test_labels}
    all_dices = []
    
    start_time = time.time()
    
    # Support scan IDs
    _scan_ids = te_parent.scan_ids
    
    # Sample only a subset of test data for faster parameter tuning
    max_samples_per_class = 20
    
    with torch.no_grad():
        # Select first support for simplicity
        sup_idx = 0
        
        for curr_lb in test_labels:
            print(f"  Testing class: {curr_lb}")
            
            # Set current class
            test_dataset.set_curr_cls(curr_lb)
            
            # Get support data
            support_batched = te_parent.get_support(
                curr_class=curr_lb, 
                class_idx=[curr_lb], 
                scan_idx=[sup_idx], 
                npart=3  # Number of parts for support
            )
            
            # Prepare support data
            support_images = [[shot.cuda() for shot in way]
                             for way in support_batched['support_images']]
            suffix = 'mask'
            support_fg_mask = [[shot[f'fg_{suffix}'].float().cuda() for shot in way]
                              for way in support_batched['support_mask']]
            support_bg_mask = [[shot[f'bg_{suffix}'].float().cuda() for shot in way]
                              for way in support_batched['support_mask']]
            
            # Count samples processed
            samples_count = 0
            
            # Process each query sample
            for sample_batched in testloader:
                # Skip support scans
                _scan_id = sample_batched["scan_id"][0]
                if _scan_id in te_parent.potential_support_sid:
                    continue
                
                # Limit number of samples per class for faster evaluation
                samples_count += 1
                if samples_count > max_samples_per_class:
                    break
                
                # Get query part assignment
                q_part = sample_batched["part_assign"]
                
                # Prepare query data
                query_images = [sample_batched['image'].cuda()]
                query_labels = torch.cat([sample_batched['label'].cuda()], dim=0)
                
                # Select appropriate support based on part
                sup_img_part = [[shot_tensor.unsqueeze(0) for shot_tensor in support_images[0][q_part]]]
                sup_fgm_part = [[shot_tensor.unsqueeze(0) for shot_tensor in support_fg_mask[0][q_part]]]
                sup_bgm_part = [[shot_tensor.unsqueeze(0) for shot_tensor in support_bg_mask[0][q_part]]]
                
                # Forward pass
                query_pred, _, _, _ = model(
                    sup_img_part, 
                    sup_fgm_part, 
                    sup_bgm_part, 
                    query_images, 
                    isval=True, 
                    val_wsize=2
                )
                
                # Convert prediction to binary mask
                pred_mask = query_pred.argmax(dim=1).float()
                
                # Calculate Dice coefficient
                intersection = (pred_mask * query_labels).sum().item()
                union = pred_mask.sum().item() + query_labels.sum().item()
                if union > 0:
                    dice = (2 * intersection) / union
                else:
                    dice = 1.0 if intersection == 0 else 0.0
                
                # Record metrics
                all_dices.append(dice)
                class_dices[curr_lb].append(dice)
    
    # Calculate metrics
    processing_time = time.time() - start_time
    
    metrics = {
        "mean_dice": np.mean(all_dices),
        "median_dice": np.median(all_dices),
        "min_dice": np.min(all_dices),
        "max_dice": np.max(all_dices),
        "std_dice": np.std(all_dices),
        "processing_time": processing_time,
        "samples_per_second": len(all_dices) / processing_time if processing_time > 0 else 0
    }
    
    # Add class-specific metrics
    for cls in class_dices:
        if class_dices[cls]:
            metrics[f"class_{cls}_dice"] = np.mean(class_dices[cls])
    
    return metrics

@ex.automain
def main(_run, _config, _log):
    if _run.observers:
        os.makedirs(f'{_run.observers[0].dir}/interm_preds', exist_ok=True)
        for source_file, _ in _run.experiment_info['sources']:
            os.makedirs(os.path.dirname(f'{_run.observers[0].dir}/source/{source_file}'),
                        exist_ok=True)
            _run.observers[0].save_file(source_file, f'source/{source_file}')
        shutil.rmtree(f'{_run.observers[0].basedir}/_sources')

    # Add these lines to register the new parameters if they don't exist
    
    cudnn.enabled = True
    cudnn.benchmark = True
    torch.cuda.set_device(device=_config['gpu_id'])
    torch.set_num_threads(1)

    _log.info(f'###### Reload model {_config["reload_model_path"]} ######')
    model = FewShotSeg(pretrained_path=None, cfg=_config['model'])
    model.load_state_dict(torch.load(_config['reload_model_path'])['model'], strict=False)
    model.use_refinement = True
    model.max_iterations = 3
    model.refinement = IterativeAdaptiveRefinement(
    max_iterations=3,
    use_feedback=True,
    small_object_threshold=0.01,
    large_object_threshold=0.2,
    small_kernel_size=3,
    medium_kernel_size=3,
    large_kernel_size=5,
    confidence_threshold=0.5
)

    model = model.cuda()
    model.eval()

    _log.info('###### Load data ######')
    ### Training set
    data_name = _config['dataset']
    if data_name == 'CHAOST2_Superpix':
        baseset_name = 'CHAOST2'
        max_label = 4
    elif data_name == 'C0_Superpix':
        raise NotImplementedError
        baseset_name = 'C0'
        max_label = 3
    elif data_name == 'SABS_Superpix':
        baseset_name = 'SABS'
        max_label = 13
    else:
        raise ValueError(f'Dataset: {data_name} not found')

    test_labels = DATASET_INFO[baseset_name]['LABEL_GROUP']['pa_all'] - DATASET_INFO[baseset_name]['LABEL_GROUP'][_config["label_sets"]]

    ### Transforms for data augmentation
    te_transforms = None

    assert _config['scan_per_load'] < 0 # by default we load the entire dataset directly

    _log.info(f'###### Labels excluded in training : {[lb for lb in _config["exclude_cls_list"]]} ######')
    _log.info(f'###### Unseen labels evaluated in testing: {[lb for lb in test_labels]} ######')

    if baseset_name == 'CHAOST2': # for CT we need to know statistics of 
        tr_parent = SuperpixelDataset( # base dataset
            which_dataset=baseset_name,
            base_dir=_config['path'][data_name]['data_dir'],
            idx_split=_config['eval_fold'],
            mode='train',
            min_fg=str(_config["min_fg_data"]), # dummy entry for superpixel dataset
            transform_param_limits=myaug.augs[_config['which_aug']],
            nsup=_config['task']['n_shots'],
            scan_per_load=_config['scan_per_load'],
            exclude_list=_config["exclude_cls_list"],
            superpix_scale=_config["superpix_scale"],
            fix_length=_config["max_iters_per_load"] if (data_name == 'C0_Superpix') or (data_name == 'CHAOST2_Superpix') else None,
            dataset_config=_config['DATASET_CONFIG']
            )
        norm_func = tr_parent.norm_func
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
        extern_normalize_func=norm_func
    )

    ### dataloaders
    testloader = DataLoader(
        te_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False
    )

    # Check if parameter tuning is requested
    if _config.get('run_param_tuning', True):
        _log.info('###### Running parameter tuning experiment ######')
        os.makedirs(f'{_run.observers[0].dir}/param_tuning', exist_ok=True)
        output_csv = f'{_run.observers[0].dir}/param_tuning/refinement_results.csv'
        
        best_params = run_parameter_tuning(
            model,
            testloader,
            te_dataset,
            te_parent,
            test_labels,
            output_csv
        )
        
        # Apply best parameters if requested
        if _config.get('use_best_params', False):
            _log.info(f'###### Applying best parameters from tuning ######')
            model.max_iterations = best_params["refinement_iterations"]
            model.refinement.max_iterations = best_params["refinement_iterations"]
            model.refinement.use_feedback = best_params["use_feedback"]
            model.refinement.small_object_threshold = best_params["small_object_threshold"]
            model.refinement.large_object_threshold = best_params["large_object_threshold"]
            model.refinement.small_kernel_size = best_params["small_kernel_size"]
            model.refinement.medium_kernel_size = best_params["medium_kernel_size"]
            model.refinement.large_kernel_size = best_params["large_kernel_size"]

    _log.info('###### Set validation nodes ######')
    _scan_ids = te_parent.scan_ids
    mar_val_metric_nodes = {}

    _log.info('###### Starting validation ######')
    model.eval()

    with torch.no_grad():
        save_pred_buffer_ = {} # indexed by class
        for sup_idx in range(len(_scan_ids)):
            _log.info(f'###### Set validation nodes for support id {_scan_ids[sup_idx]} ######')
            mar_val_metric_nodes[_scan_ids[sup_idx]] = Metric(max_label=max_label, n_scans=len(te_dataset.dataset.pid_curr_load) - _config['task']['n_shots'])
            mar_val_metric_nodes[_scan_ids[sup_idx]].reset()

            save_pred_buffer = {}

            for curr_lb in test_labels:
                te_dataset.set_curr_cls(curr_lb)
                support_batched = te_parent.get_support(curr_class=curr_lb, 
                                                        class_idx=[curr_lb], 
                                                        scan_idx=[sup_idx], 
                                                        npart=_config['task']['npart'])

                # way(1 for now) x part x shot x 3 x H x W] #
                support_images = [[shot.cuda() for shot in way]
                                    for way in support_batched['support_images']] # way x part x [shot x C x H x W]
                suffix = 'mask'
                support_fg_mask = [[shot[f'fg_{suffix}'].float().cuda() for shot in way]
                                    for way in support_batched['support_mask']]
                support_bg_mask = [[shot[f'bg_{suffix}'].float().cuda() for shot in way]
                                    for way in support_batched['support_mask']]

                curr_scan_count = -1 # counting for current scan
                _lb_buffer = {} # indexed by scan

                last_qpart = 0 # used as indicator for adding result to buffer

                for sample_batched in testloader:
                    _scan_id = sample_batched["scan_id"][0] # we assume batch size for query is 1
                    if _scan_id in te_parent.potential_support_sid: # skip the support scan, don't include that to query
                        continue
                    if sample_batched["is_start"]:
                        ii = 0
                        curr_scan_count += 1
                        _scan_id = sample_batched["scan_id"][0]
                        outsize = te_dataset.dataset.info_by_scan[_scan_id]["array_size"]
                        outsize = (256, 256, outsize[0]) # original image read by itk: Z, H, W, in prediction we use H, W, Z
                        _pred = np.zeros(outsize)
                        _pred.fill(np.nan)
                        _labels = np.zeros(outsize)
                        _qimgs = np.zeros((3, 256, 256, outsize[-1])) #outsize)
                        supimgs = np.zeros((3, 256, 256, outsize[-1])) #outsize)
                        supfgs = np.zeros((256, 256, outsize[-1])) #outsize)
                        _means = np.zeros((1, 1, 1, outsize[-1]))
                        _stds = np.zeros((1, 1, 1, outsize[-1]))

                    q_part = sample_batched["part_assign"] # the chunck of query, for assignment with support
                    query_images = [sample_batched['image'].cuda()]
                    query_labels = torch.cat([sample_batched['label'].cuda()], dim=0)

                    # [way, [part, [shot x C x H x W]]] ->
                    sup_img_part = [[shot_tensor.unsqueeze(0) for shot_tensor in support_images[0][q_part]]]   # way(1) x shot x [B(1) x C x H x W]
                    sup_fgm_part = [[shot_tensor.unsqueeze(0) for shot_tensor in support_fg_mask[0][q_part]]]
                    sup_bgm_part = [[shot_tensor.unsqueeze(0) for shot_tensor in support_bg_mask[0][q_part]]]

                    query_pred, _, _, _ = model(sup_img_part, sup_fgm_part, sup_bgm_part, query_images, isval=True, val_wsize=_config["val_wsize"])

                    query_pred = query_pred.argmax(dim=1).float().cpu().numpy() 
                    _pred[..., ii] = query_pred[0].copy()
                    _labels[..., ii] = query_labels[0].detach().cpu().clone().numpy()
                    _qimgs[..., ii] = query_images[0].detach().cpu().clone().numpy()
                    _means[..., ii] = sample_batched["mean"][0]
                    _stds[..., ii] = sample_batched["std"][0]

                    supimgs[..., ii] = sup_img_part[0][0][0].cpu().numpy()
                    supfgs[..., ii] = sup_fgm_part[0][0][0].cpu().numpy()

                    if (sample_batched["z_id"] - sample_batched["z_max"] <= _config['z_margin']) and (sample_batched["z_id"] - sample_batched["z_min"] >= -1 * _config['z_margin']):
                        mar_val_metric_nodes[_scan_ids[sup_idx]].record(_pred[..., ii], _labels[..., ii], labels=[curr_lb], n_scan=curr_scan_count) 
                    else:
                        pass

                    ii += 1
                    # now check data format
                    if sample_batched["is_end"]:
                        if _config['dataset'] != 'C0':
                            _lb_buffer[_scan_id] = [_pred.transpose(2, 0, 1),
                                                    _labels.transpose(2, 0, 1),
                                                    _qimgs.transpose(3, 1, 2, 0), 
                                                    _means.transpose(3, 1, 2, 0), 
                                                    _stds.transpose(3, 1, 2, 0), 
                                                    supimgs.transpose(3, 1, 2, 0),
                                                    supfgs.transpose(2, 0, 1)] # H, W, Z -> to Z H W
                        else:
                            _lb_buffer[_scan_id] = [_pred, _labels, _qimgs, _means, _stds, sup_img_part, sup_fgm_part]

                save_pred_buffer[str(curr_lb)] = _lb_buffer

            save_pred_buffer_[_scan_ids[sup_idx]] = save_pred_buffer

        ### save results
        for _sup_id, saved_pred in save_pred_buffer_.items():
            for curr_lb, _preds in saved_pred.items():
                c = 0
                for _scan_id, item in _preds.items():
                    gif_frames = []
                    _pred, _label, _qimg, _mean, _std, _supimg, _supfg = item
                    print(_pred.shape, _qimg.shape, _label.shape, _mean.shape, _std.shape, _supimg.shape, _supfg.shape)
                    _pred = 255*(_pred - _pred.min(axis=(1, 2), keepdims=True))/ (1.e-6 + _pred.max(axis=(1, 2), keepdims=True) - _pred.min(axis=(1, 2), keepdims=True))
                    _pred = _pred.astype(np.uint8)
                    _label = 255*(_label).astype(np.uint8)
                    _supfg = (255*_supfg).astype(np.uint8)
                    
                    for i in range(_pred.shape[0]):
                        fid = os.path.join(f'{_run.observers[0].dir}/interm_preds', f'scan_{_sup_id}_{_scan_id}_label_{curr_lb}_{i}.png')
                        qim = _qimg[i]*_std[i] + _mean[i]
                        qim = 255*(qim - qim.min())/(qim.max() - qim.min() + 1e-6)
                        qim = qim.astype(np.uint8)
                        qim = Image.fromarray(qim, 'RGB').convert("RGBA")

                        im1 = np.stack([_pred[i]*0.8, np.zeros(_pred[i].shape), np.zeros(_pred[i].shape)], axis=2)
                        im1 = im1.astype(np.uint8)
                        im1 = Image.fromarray(im1, 'RGB').convert("RGBA")

                        im2 = np.stack([np.zeros(_label[i].shape), _label[i]*0.8, np.zeros(_label[i].shape)], axis=2)
                        im2 = im2.astype(np.uint8)
                        im2 = Image.fromarray(im2, 'RGB').convert("RGBA")
                        
                        qim = Image.blend(qim, im1, alpha=0.5)
                        qim = Image.blend(qim, im2, alpha=0.3)

                        supimg = _supimg[i]*_std[i] + _mean[i]
                        supimg = (supimg - supimg.min())/(supimg.max() - supimg.min())*255
                        supimg = supimg.astype(np.uint8)
                        supimg = Image.fromarray(supimg, 'RGB').convert("RGBA")

                        supfg = _supfg[i]
                        supfg = np.stack([np.zeros(supfg.shape), supfg, np.zeros(supfg.shape)], axis=2)
                        supfg = supfg.astype(np.uint8)
                        supfg = Image.fromarray(supfg, 'RGB').convert("RGBA")

                        supimg = Image.blend(supimg, supfg, alpha=0.4)

                        new_image = Image.new('RGB', (2*256, 256), (250, 250, 250))
                        new_image.paste(supimg, (0, 0))
                        new_image.paste(qim, (256, 0))

                        gif_frames.append(new_image)
                        new_image.save(fid)
                        
                    fid = os.path.join(f'{_run.observers[0].dir}/interm_preds', f'scan_{_sup_id}_{_scan_id}_label_{curr_lb}.gif')
                    gif_frames[0].save(fid, format="GIF", append_images=gif_frames, save_all=True, duration=500, loop=0)
                    _log.info(f'###### {fid} has been saved #####')

        del save_pred_buffer

    del sample_batched, support_images, support_bg_mask, query_images, query_labels, query_pred

    # compute dice scores by scan
    for i in range(len(_scan_ids)):
        print(f"========== Metrics for Support Scan {_scan_ids[i]} ============")
        m_classDice, _, m_meanDice, _, m_rawDice = mar_val_metric_nodes[_scan_ids[i]].get_mDice(labels=sorted(test_labels), n_scan=None, give_raw=True)

        m_classPrec, _, m_meanPrec, _, m_classRec, _, m_meanRec, _, m_rawPrec, m_rawRec = mar_val_metric_nodes[_scan_ids[i]].get_mPrecRecall(labels=sorted(test_labels), n_scan=None, give_raw=True)

        mar_val_metric_nodes[_scan_ids[i]].reset() # reset this calculation node

        # write validation result to log file
        _run.log_scalar('mar_val_batches_classDice', m_classDice.tolist())
        _run.log_scalar('mar_val_batches_meanDice', m_meanDice.tolist())
        _run.log_scalar('mar_val_batches_rawDice', m_rawDice.tolist())

        _run.log_scalar('mar_val_batches_classPrec', m_classPrec.tolist())
        _run.log_scalar('mar_val_batches_meanPrec', m_meanPrec.tolist())
        _run.log_scalar('mar_val_batches_rawPrec', m_rawPrec.tolist())

        _run.log_scalar('mar_val_batches_classRec', m_classRec.tolist())
        _run.log_scalar('mar_val_al_batches_meanRec', m_meanRec.tolist())
        _run.log_scalar('mar_val_al_batches_rawRec', m_rawRec.tolist())

        _log.info(f'mar_val batches classDice: {m_classDice}')
        _log.info(f'mar_val batches meanDice: {m_meanDice}')

        _log.info(f'mar_val batches classDice: {m_classDice}')
        _log.info(f'mar_val batches meanDice: {m_meanDice}')

        _log.info(f'mar_val batches classPrec: {m_classPrec}')
        _log.info(f'mar_val batches meanPrec: {m_meanPrec}')

        _log.info(f'mar_val batches classRec: {m_classRec}')
        _log.info(f'mar_val batches meanRec: {m_meanRec}')

        print(f"============ Completed Support Scan {_scan_ids[i]} ============")

    # If parameter tuning was run, also log the best parameters
    if _config.get('run_param_tuning', False):
        output_csv = f'{_run.observers[0].dir}/param_tuning/refinement_results.csv'
        if os.path.exists(output_csv):
            try:
                df = pd.read_csv(output_csv)
                best_row = df.loc[df['mean_dice'].idxmax()]
                
                _log.info(f'###### Best parameter combination from tuning ######')
                for param, value in best_row.items():
                    if param not in ['mean_dice', 'median_dice', 'min_dice', 'max_dice', 'std_dice', 
                                    'processing_time', 'samples_per_second', 'config']:
                        _log.info(f'  {param}: {value}')
                _log.info(f'  mean_dice: {best_row["mean_dice"]:.4f}')
                
                # Create visualization of parameter effects if matplotlib is available
                try:
                    import matplotlib
                    matplotlib.use('Agg')  # Use non-interactive backend
                    import matplotlib.pyplot as plt
                    import seaborn as sns
                    
                    param_names = ['refinement_iterations', 'use_feedback', 'small_object_threshold', 
                                   'large_object_threshold', 'small_kernel_size', 'medium_kernel_size', 
                                   'large_kernel_size']
                    
                    plt.figure(figsize=(15, 10))
                    for i, param in enumerate(param_names):
                        if param in df.columns:
                            plt.subplot(2, 4, i+1)
                            if df[param].dtype == bool:
                                # For boolean parameters
                                sns.boxplot(x=param, y='mean_dice', data=df)
                            else:
                                # For numeric parameters
                                sns.regplot(x=param, y='mean_dice', data=df, scatter=True)
                            plt.title(f'Effect of {param}')
                            plt.xticks(rotation=45)
                    
                    plt.tight_layout()
                    plot_path = f'{_run.observers[0].dir}/param_tuning/parameter_effects.png'
                    plt.savefig(plot_path)
                    _log.info(f'  Parameter effects visualization saved to {plot_path}')
                    plt.close()
                except ImportError:
                    _log.info("Could not create visualization. Make sure matplotlib and seaborn are installed.")
            except Exception as e:
                _log.info(f"Error reading parameter tuning results: {e}")

    _log.info(f'End of validation')
    return 1
