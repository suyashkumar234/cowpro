import os
import shutil
import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import torch.backends.cudnn as cudnn
import numpy as np

from models.grid_proto_fewshotsa import FewShotSeg
from dataloaders.dev_customized_med import med_fewshot
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

def visualize_attention_across_channels(model, i_iter, run_dir, query_img=None):
    """
    Visualize the average over all channels for the output of the attention module
    
    Args:
        model: The FewShotSeg model
        i_iter: Current iteration number
        run_dir: Directory for saving outputs
        query_img: Optional normalized query image for reference
    """
    try:
        import os
        import numpy as np
        import matplotlib.pyplot as plt
        
        # Create output directory
        attention_dir = os.path.join(run_dir, "channel_averaged_attention")
        os.makedirs(attention_dir, exist_ok=True)
        
        # Check if model has necessary attributes
        if not hasattr(model, 'features_before_attention') or model.features_before_attention is None:
            print("No feature data available for visualization")
            return
        
        # Get data from model
        features_before = model.features_before_attention.detach()
        features_after = model.features_after_attention.detach()
        feature_delta = model.feature_delta.detach()
        raw_attention = model.raw_attention_contribution.detach()
        
        # Get gamma value
        gamma_value = model.self_attention.gamma.item()
        
        # For the query image (usually the last in batch)
        b_idx = features_before.shape[0] - 1 if features_before.shape[0] > 1 else 0
        
        # Compute average over all channels
        # This is the main visualization requested: average over all channels
        before_avg = features_before[b_idx].mean(dim=0).cpu().numpy()
        after_avg = features_after[b_idx].mean(dim=0).cpu().numpy()
        delta_avg = feature_delta[b_idx].mean(dim=0).cpu().numpy()
        attention_avg = raw_attention[b_idx].mean(dim=0).cpu().numpy()
        
        # Create figure for channel-averaged visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        
        # 1. Average activation before self-attention
        img1 = axes[0, 0].imshow(before_avg, cmap='viridis')
        plt.colorbar(img1, ax=axes[0, 0])
        axes[0, 0].set_title(f"Avg. Activation Before Self-Attention\nMean: {before_avg.mean():.4f}")
        axes[0, 0].axis('off')
        
        # 2. Average activation after self-attention
        img2 = axes[0, 1].imshow(after_avg, cmap='viridis')
        plt.colorbar(img2, ax=axes[0, 1])
        axes[0, 1].set_title(f"Avg. Activation After Self-Attention\nMean: {after_avg.mean():.4f}")
        axes[0, 1].axis('off')
        
        # 3. Average raw attention contribution
        img3 = axes[1, 0].imshow(attention_avg, cmap='viridis')
        plt.colorbar(img3, ax=axes[1, 0])
        axes[1, 0].set_title(f"Avg. Attention Output\nMean: {attention_avg.mean():.4f}")
        axes[1, 0].axis('off')
        
        # 4. Average delta (difference) visualization with diverging colormap
        vmax = np.max(np.abs(delta_avg))
        img4 = axes[1, 1].imshow(delta_avg, cmap='coolwarm', vmin=-vmax, vmax=vmax)
        plt.colorbar(img4, ax=axes[1, 1])
        axes[1, 1].set_title(f"Avg. Change from Self-Attention\nMean: {delta_avg.mean():.4f}")
        axes[1, 1].axis('off')
        
        plt.suptitle(f"Channel-Averaged Self-Attention Analysis\nIteration {i_iter} - Gamma: {gamma_value:.6f}", fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(attention_dir, f"channel_avg_attention_{i_iter}.png"), bbox_inches='tight')
        plt.close(fig)
        
        # Create a side-by-side visualization with query image for reference
        if query_img is not None:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # Query image
            axes[0].imshow(query_img)
            axes[0].set_title("Query Image")
            axes[0].axis('off')
            
            # Channel-averaged attention output
            im = axes[1].imshow(attention_avg, cmap='hot')
            plt.colorbar(im, ax=axes[1])
            axes[1].set_title(f"Channel-Averaged Attention Output\nMean: {attention_avg.mean():.4f}")
            axes[1].axis('off')
            
            # Channel-averaged delta
            im = axes[2].imshow(delta_avg, cmap='coolwarm', vmin=-vmax, vmax=vmax)
            plt.colorbar(im, ax=axes[2])
            axes[2].set_title(f"Channel-Averaged Activation Change\nMean: {delta_avg.mean():.4f}")
            axes[2].axis('off')
            
            plt.suptitle(f"Attention Analysis (Gamma: {gamma_value:.6f}) - Iteration {i_iter}", fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(attention_dir, f"attention_with_query_{i_iter}.png"), bbox_inches='tight')
            plt.close(fig)
        
        # Track gamma values over time
        gamma_file = os.path.join(attention_dir, "gamma_values.txt")
        
        # Append current iteration and gamma value
        with open(gamma_file, "a+") as f:
            f.write(f"{i_iter},{gamma_value}\n")
        
        # Plot gamma evolution if we have enough iterations
        if i_iter % 1000 == 0:
            try:
                iterations = []
                gamma_values = []
                
                with open(gamma_file, "r") as f:
                    for line in f:
                        if line.strip():
                            it, gamma = line.strip().split(",")
                            iterations.append(int(it))
                            gamma_values.append(float(gamma))
                
                if len(iterations) >= 2:
                    plt.figure(figsize=(10, 6))
                    plt.plot(iterations, gamma_values, 'o-')
                    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
                    plt.grid(alpha=0.3)
                    plt.xlabel('Iteration')
                    plt.ylabel('Gamma Value')
                    plt.title('Evolution of Self-Attention Gamma Parameter')
                    plt.savefig(os.path.join(attention_dir, "gamma_evolution.png"))
                    plt.close()
            except Exception as e:
                print(f"Error plotting gamma evolution: {e}")
        
        # Visualize channel-wise attention statistics for a better understanding
        mean_channel_before = features_before[b_idx].mean(dim=(1,2)).cpu().numpy()
        mean_channel_after = features_after[b_idx].mean(dim=(1,2)).cpu().numpy()
        mean_channel_delta = feature_delta[b_idx].mean(dim=(1,2)).cpu().numpy()
        
        # Plot distribution of channel means
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.hist(mean_channel_before, bins=30, alpha=0.5, label='Before')
        plt.hist(mean_channel_after, bins=30, alpha=0.5, label='After')
        plt.legend()
        plt.title('Distribution of Channel Mean Activations')
        plt.xlabel('Mean Activation')
        plt.ylabel('Number of Channels')
        
        plt.subplot(1, 2, 2)
        plt.hist(mean_channel_delta, bins=30, color='green')
        plt.title('Distribution of Channel Mean Changes')
        plt.xlabel('Mean Activation Change')
        plt.ylabel('Number of Channels')
        
        plt.tight_layout()
        plt.savefig(os.path.join(attention_dir, f"channel_distributions_{i_iter}.png"))
        plt.close()
        
    except Exception as e:
        print(f"Error in channel-averaged attention visualization: {e}")

@ex.automain
def main(_run, _config, _log):
    if _run.observers:
        os.makedirs(f'{_run.observers[0].dir}/snapshots', exist_ok=True)
        os.makedirs(f'{_run.observers[0].dir}/trainsnaps', exist_ok=True)
        os.makedirs(f'{_run.observers[0].dir}/attention_maps', exist_ok=True)
        os.makedirs(f'{_run.observers[0].dir}/channel_averaged_attention', exist_ok=True)
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
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
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

    my_weight = compose_wt_simple(_config["use_wce"], data_name)
    criterion = nn.CrossEntropyLoss(ignore_index=_config['ignore_label'], weight=my_weight)

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

            mean = [m for m in sample_batched["mean"]]
            std = [s for s in sample_batched["std"]]
            
            # Forward pass through the model with self-attention
            query_pred, align_loss, debug_vis, assign_mats, attention_maps = model(
                support_images,
                support_fg_mask, 
                support_bg_mask, 
                query_images, 
                isval=False, 
                val_wsize=None,
                show_viz=True  # Enable visualization
            )

            # Calculate losses
            query_loss = criterion(query_pred, query_labels) + get_tversky_loss(
                query_pred.argmax(dim=1, keepdim=True), 
                query_labels[None, ...], 
                0.3, 0.7, 1.0
            )
            loss = query_loss + align_loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Log loss
            query_loss_value = query_loss.detach().data.cpu().numpy()
            align_loss_value = align_loss.detach().data.cpu().numpy() if align_loss != 0 else 0

            _run.log_scalar('loss', query_loss_value)
            _run.log_scalar('align_loss', align_loss_value)
            log_loss['loss'] += query_loss_value
            log_loss['align_loss'] += align_loss_value

            # Print loss and visualization at intervals
            if (i_iter + 1) % _config['print_interval'] == 0:
                nt = time.time()

                loss_avg = log_loss['loss'] / _config['print_interval']
                align_loss_avg = log_loss['align_loss'] / _config['print_interval']

                log_loss['loss'] = 0
                log_loss['align_loss'] = 0

                # Create visualization
                try:
                    fig, ax = plt.subplots(2, 2, figsize=(12, 10))
                    
                    # Support image and mask
                    si = (support_images[0][0][0].cpu()*std[0]+mean[0]).numpy().transpose((1,2,0))
                    si = (si - si.min())/(si.max() - si.min() + 1e-6)
                    ax[0, 0].imshow(si)
                    sm = support_fg_mask[0][0][0].cpu().numpy()
                    ax[0, 0].imshow(np.stack([np.zeros(sm.shape), sm, np.zeros(sm.shape)], axis=2), alpha=0.3)
                    ax[0, 0].set_title("Support Image & Mask")
                    ax[0, 0].axis('off')

                    # Query image, ground truth, and prediction
                    qi = (query_images[0][0].cpu()*std[0]+mean[0]).numpy().transpose((1,2,0))
                    qi = (qi - qi.min())/(qi.max() - qi.min() + 1e-6)
                    ax[0, 1].imshow(qi)
                    qm = query_labels[0].cpu().numpy()
                    ax[0, 1].imshow(np.stack([np.zeros(qm.shape), qm, np.zeros(qm.shape)], axis=2), alpha=0.3)
                    qp = query_pred.argmax(dim=1).float().cpu().numpy()[0]
                    ax[0, 1].imshow(np.stack([qp, np.zeros(qp.shape), np.zeros(qp.shape)], axis=2), alpha=0.2)
                    ax[0, 1].set_title("Query Image, GT & Pred")
                    ax[0, 1].axis('off')
                    
                    # Try to visualize attention map
                    if attention_maps is not None:
                        try:
                            # Get attention map for visualization
                            att_map = attention_maps[0].detach().cpu().numpy()
                            ax[1, 0].imshow(att_map, cmap='hot')
                            ax[1, 0].set_title("Self-Attention Map")
                            ax[1, 0].axis('off')
                        except Exception as e:
                            print(f"Error visualizing attention map: {e}")
                            ax[1, 0].text(0.5, 0.5, "Attention visualization error", 
                                         ha='center', va='center')
                            ax[1, 0].axis('off')
                    else:
                        ax[1, 0].text(0.5, 0.5, "No attention map available", 
                                     ha='center', va='center')
                        ax[1, 0].axis('off')
                    
                    # Display gamma value
                    try:
                        gamma_value = model.self_attention.gamma.item()
                        ax[1, 1].text(0.5, 0.5, f"Self-Attention Gamma: {gamma_value:.6f}", 
                                     ha='center', va='center', fontsize=12)
                        ax[1, 1].axis('off')
                    except Exception as e:
                        print(f"Error displaying gamma value: {e}")
                        ax[1, 1].text(0.5, 0.5, "Could not retrieve gamma", 
                                     ha='center', va='center')
                        ax[1, 1].axis('off')
                
                    plt.tight_layout()
                    plt.savefig(os.path.join(f'{_run.observers[0].dir}/trainsnaps', f'{i_iter + 1}.png'), bbox_inches='tight')
                    plt.close(fig)
                except Exception as e:
                    print(f"Error creating main visualization: {e}")
                
                # Generate channel-averaged attention visualization
                try:
                    # Prepare normalized query image for reference
                    qi_normalized = (query_images[0][0].cpu()*std[0]+mean[0]).numpy().transpose((1,2,0))
                    qi_normalized = (qi_normalized - qi_normalized.min())/(qi_normalized.max() - qi_normalized.min() + 1e-6)
                    
                    # Call the attention visualization function
                    visualize_attention_across_channels(
                        model=model, 
                        i_iter=i_iter + 1, 
                        run_dir=f'{_run.observers[0].dir}',
                        query_img=qi_normalized
                    )
                except Exception as e:
                    print(f"Error creating attention analysis: {e}")
                
                print(f'step {i_iter+1}: loss: {loss_avg}, align_loss: {align_loss_avg}, time: {(nt-stime)/60} mins')
                
                # Print gamma value if available
                try:
                    gamma_value = model.self_attention.gamma.item()
                    print(f'Self-attention gamma value: {gamma_value:.6f}')
                    
                    # Log gamma value
                    _run.log_scalar('gamma_value', gamma_value)
                except Exception as e:
                    print(f'Could not retrieve gamma value: {e}')

            # Save model snapshots
            if (i_iter + 1) % _config['save_snapshot_every'] == 0:
                _log.info('###### Taking snapshot ######')
                torch.save({'model': model.state_dict(), 'opt': optimizer.state_dict(), 'sch': scheduler.state_dict()},
                           os.path.join(f'{_run.observers[0].dir}/snapshots', f'{i_iter + 1}.pth'))

            # Reload dataset if needed
            if data_name == 'C0_Superpix' or data_name == 'CHAOST2_Superpix':
                if (i_iter + 1) % _config['max_iters_per_load'] == 0:
                    _log.info('###### Reloading dataset ######')
                    trainloader.dataset.reload_buffer()
                    print(f'###### New dataset with {len(trainloader.dataset)} slices has been loaded ######')

            # Check if we've reached the maximum number of steps
            if (i_iter - 2) > _config['n_steps']:
                return 1  # finish up
