"""
Fixed Validation script for Supervised CoWPro Model
Includes proper prediction saving and visualization like original validation.py
"""
import os
import shutil
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
import json
import glob
import re
import copy

from models.grid_proto_fewshot import FewShotSeg
from dataloaders.dataset_utils import DATASET_INFO, get_normalize_op, read_nii_bysitk
from dataloaders.common import BaseDataset
from dataloaders.niftiio import convert_to_sitk
from util.utils import CircularList
from util.metric import Metric

from config_ssl_upload import ex

import matplotlib.pyplot as plt
from PIL import Image
import imageio
import SimpleITK as sitk

# config pre-trained model caching path
os.environ['TORCH_HOME'] = "./pretrained_model"

class SupervisedValidationDataset(BaseDataset):
    """
    Validation dataset that mimics the structure of the original validation dataset
    """
    def __init__(self, which_dataset, base_dir, idx_split, nsup=1, extern_normalize_func=None):
        super(SupervisedValidationDataset, self).__init__(base_dir)
        
        self.img_modality = DATASET_INFO[which_dataset]['MODALITY']
        self.sep = DATASET_INFO[which_dataset]['_SEP']
        self.label_name = DATASET_INFO[which_dataset]['REAL_LABEL_NAME']
        self.all_label_names = self.label_name
        self.base_dir = base_dir
        self.nsup = nsup
        
        # Find all scan IDs
        self.img_pids = [re.findall('\d+', fid)[-1] for fid in glob.glob(self.base_dir + "/image_*.nii.gz")]
        self.img_pids = CircularList(sorted(self.img_pids, key=lambda x: int(x)))
        
        # Get validation scan IDs
        val_ids = copy.deepcopy(self.img_pids[self.sep[idx_split]: self.sep[idx_split + 1] + self.nsup])
        self.scan_ids = val_ids
        self.pid_curr_load = val_ids
        self.potential_support_sid = val_ids[-self.nsup:]  # Last nsup scans as potential support
        
        print(f"Validation scans: {self.scan_ids}")
        print(f"Potential support scans: {self.potential_support_sid}")
        
        # Set up normalization
        if extern_normalize_func is not None:
            self.norm_func = extern_normalize_func
        else:
            img_fids = [os.path.join(self.base_dir, f'image_{scan_id}.nii.gz') for scan_id in self.scan_ids]
            self.norm_func = get_normalize_op(self.img_modality, img_fids)
        
        # Load data
        self.actual_dataset = self.read_dataset()
        self.overall_slice_by_cls = self.read_classfiles()
        
        # Add current class tracking (needed for original validation interface)
        self.__curr_cls = None
        
    def set_curr_cls(self, curr_cls):
        """Set current class for validation (original interface)"""
        self.__curr_cls = curr_cls

    def get_curr_cls(self):
        """Get current class (original interface)"""
        return self.__curr_cls

    def safe_read_nii(self, file_path, peel_info=False):
        """Safely read NII file and handle tuple returns"""
        try:
            result = read_nii_bysitk(file_path, peel_info=peel_info)
            if isinstance(result, tuple):
                if peel_info:
                    return result[0], result[1] if len(result) > 1 else None
                else:
                    return result[0]
            else:
                if peel_info:
                    return result, None
                else:
                    return result
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            raise
    
    def read_dataset(self):
        """Read all validation scans into memory"""
        out_list = []
        self.scan_z_idx = {}
        self.info_by_scan = {}
        glb_idx = 0
        
        for scan_id in self.scan_ids:
            print(f"Loading validation scan {scan_id}...")
            
            img_fid = os.path.join(self.base_dir, f'image_{scan_id}.nii.gz')
            lb_fid = os.path.join(self.base_dir, f'label_{scan_id}.nii.gz')
            
            # Read image
            img, _info = self.safe_read_nii(img_fid, peel_info=True)
            img = img.transpose(1, 2, 0)  # Z,H,W -> H,W,Z
            self.info_by_scan[scan_id] = _info
            
            img = np.float32(img)
            norm_result = self.norm_func(img)
            if isinstance(norm_result, tuple):
                img = norm_result[0]
            else:
                img = norm_result
                
            # Read label
            lb = self.safe_read_nii(lb_fid, peel_info=False)
            lb = lb.transpose(1, 2, 0)  # Z,H,W -> H,W,Z
            lb = np.float32(lb)
            
            # Crop to 256x256
            img = img[:256, :256, :]
            lb = lb[:256, :256, :]
            
            # Ensure same number of slices
            min_slices = min(img.shape[-1], lb.shape[-1])
            img = img[:, :, :min_slices]
            lb = lb[:, :, :min_slices]
            
            self.scan_z_idx[scan_id] = [-1 for _ in range(img.shape[-1])]
            
            # Store all slices
            for ii in range(img.shape[-1]):
                out_list.append({
                    "img": img[..., ii: ii + 1],
                    "lb": lb[..., ii: ii + 1],
                    "is_start": ii == 0,
                    "is_end": ii == img.shape[-1] - 1,
                    "nframe": img.shape[-1],
                    "scan_id": scan_id,
                    "z_id": ii,
                    "part_assign": 1,  # Default part assignment
                    "z_min": 0,
                    "z_max": img.shape[-1] - 1
                })
                
                self.scan_z_idx[scan_id][ii] = glb_idx
                glb_idx += 1
                
            print(f"  Loaded {img.shape[-1]} slices")
            
        print(f"Total validation dataset size: {len(out_list)} slices")
        return out_list
    
    def read_classfiles(self):
        """Load class-slice indexing files"""
        classmap_path = os.path.join(self.base_dir, 'classmap_1.json')
        if not os.path.exists(classmap_path):
            print(f"Warning: Classmap file not found: {classmap_path}")
            return {}
            
        with open(classmap_path, 'r') as f:
            cls_map = json.load(f)
        
        # Also store as tp1_cls_map for compatibility
        self.tp1_cls_map = cls_map
        
        return cls_map
    
    def label_strip(self, label):
        """Mask unrelated labels out (original interface)"""
        if self.__curr_cls is None:
            raise Exception("Please initialize current class first")
            
        out = torch.where(label == self.__curr_cls,
                          torch.ones_like(label), torch.zeros_like(label))
        return out

    def __getitem__(self, idx):
        """Get item with original interface compatibility"""
        sample = self.actual_dataset[idx]
        
        # Convert to tensors
        img = torch.from_numpy(np.transpose(sample["img"], (2, 0, 1)))  # (1, H, W)
        lb = torch.from_numpy(sample["lb"].squeeze(-1))  # (H, W)
        
        # Tile image channels to 3
        img = img.repeat([3, 1, 1])  # (3, H, W)
        
        # Apply label stripping if current class is set
        if self.__curr_cls is not None:
            lb = self.label_strip(lb)
            
        # Add metadata for compatibility with original validation
        sample_out = {
            "image": img,
            "label": lb,
            "is_start": sample["is_start"],
            "is_end": sample["is_end"],
            "nframe": sample["nframe"],
            "scan_id": sample["scan_id"],
            "z_id": sample["z_id"],
            "part_assign": sample["part_assign"],
            "z_min": sample["z_min"],
            "z_max": sample["z_max"],
            "mean": torch.tensor([0.0]),  # Dummy values for compatibility
            "std": torch.tensor([1.0])
        }
        
        return sample_out
    
    def __len__(self):
        return len(self.actual_dataset)
    
    def get_support(self, curr_class, class_idx, scan_idx, npart=3):
        """Get support data for a specific class and scan (original interface)"""
        support_scan_id = self.scan_ids[scan_idx[0]]  # scan_idx is a list
        class_name = self.label_name[curr_class]
        
        if class_name not in self.overall_slice_by_cls or support_scan_id not in self.overall_slice_by_cls[class_name]:
            print(f"Warning: No slices found for class {class_name} in scan {support_scan_id}")
            return {'class_ids': [[curr_class]], 'support_images': [[]], 'support_mask': [[]]}
            
        # Get slices containing this class
        class_slices = self.overall_slice_by_cls[class_name][support_scan_id]
        if not class_slices:
            return {'class_ids': [[curr_class]], 'support_images': [[]], 'support_mask': [[]]}
            
        # Create support data for multiple parts
        out_buffer = []
        
        if npart == 1:
            pcts = [0.5]
        else:
            half_part = 1 / (npart * 2)
            part_interval = (1.0 - 1.0 / npart) / (npart - 1)
            pcts = [half_part + part_interval * ii for ii in range(npart)]
            
        for _part in range(npart):
            # Get slice for this part
            _zid = class_slices[int(pcts[_part] * len(class_slices))]
            _glb_idx = self.scan_z_idx[support_scan_id][_zid]
            
            support_data = self.actual_dataset[_glb_idx]
            
            # Convert to proper format
            support_img = support_data["img"]  # (H, W, 1)
            support_lb = support_data["lb"]    # (H, W, 1)
            
            # Create binary mask for target class
            support_mask = np.float32(support_lb == curr_class)  # (H, W, 1)
            
            # Convert to tensors with proper dimensions
            support_img = torch.from_numpy(np.transpose(support_img, (2, 0, 1)))  # (1, H, W)
            support_mask = torch.from_numpy(support_mask.squeeze(-1))  # (H, W)
            
            # Tile image channels to 3
            support_img = support_img.repeat([3, 1, 1])  # (3, H, W)
            
            out_buffer.append({
                "image": support_img,
                "label": support_mask,
            })
        
        # Format for model (way x part x shot)
        support_images = []
        support_mask = []
        support_class = []
        
        for itm in out_buffer:
            support_images.append(itm["image"])
            support_class.append(curr_class)
            
            # Create FG/BG masks
            fg_mask = torch.where(itm["label"] == 1, torch.ones_like(itm["label"]), torch.zeros_like(itm["label"]))
            bg_mask = torch.where(itm["label"] == 0, torch.ones_like(itm["label"]), torch.zeros_like(itm["label"]))
            
            support_mask.append({
                'fg_mask': fg_mask,
                'bg_mask': bg_mask
            })

        return {
            'class_ids': [support_class],
            'support_images': [support_images],
            'support_mask': [support_mask],
        }

class ValidationDatasetWrapper:
    """Wrapper to mimic the original validation dataset interface"""
    def __init__(self, dataset, test_classes, npart):
        self.dataset = dataset
        self.test_classes = test_classes
        self.npart = npart
        
    def set_curr_cls(self, curr_cls):
        self.dataset.set_curr_cls(curr_cls)
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        return self.dataset[idx]

@ex.automain
def main(_run, _config, _log):
    if _run.observers:
        os.makedirs(f'{_run.observers[0].dir}/interm_preds', exist_ok=True)
        for source_file, _ in _run.experiment_info['sources']:
            os.makedirs(os.path.dirname(f'{_run.observers[0].dir}/source/{source_file}'),
                        exist_ok=True)
            _run.observers[0].save_file(source_file, f'source/{source_file}')
        shutil.rmtree(f'{_run.observers[0].basedir}/_sources')

    cudnn.enabled = True
    cudnn.benchmark = True
    torch.cuda.set_device(device=_config['gpu_id'])
    torch.set_num_threads(1)

    _log.info(f'###### Reload model {_config["reload_model_path"]} ######')
    model = FewShotSeg(pretrained_path=None, cfg=_config['model'])
    
    # Load the trained model
    if not os.path.exists(_config['reload_model_path']):
        raise FileNotFoundError(f"Model file not found: {_config['reload_model_path']}")
        
    checkpoint = torch.load(_config['reload_model_path'], map_location='cuda')
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'], strict=False)
        _log.info(f'Loaded model from iteration {checkpoint.get("iter", "unknown")}')
    else:
        model.load_state_dict(checkpoint, strict=False)
        _log.info('Loaded model (no iteration info)')

    model = model.cuda()
    model.eval()

    _log.info('###### Load data ######')
    
    # Handle different dataset naming conventions
    data_name = _config['dataset']
    if data_name.endswith('_Supervised'):
        baseset_name = data_name.replace('_Supervised', '')
    else:
        baseset_name = data_name
        
    if baseset_name == 'SABS':
        max_label = 13
    elif baseset_name == 'CHAOST2':
        max_label = 4
    elif baseset_name == 'C0':
        max_label = 3
    else:
        raise ValueError(f'Dataset: {data_name} not found')

    test_labels = DATASET_INFO[baseset_name]['LABEL_GROUP']['pa_all'] - \
                  DATASET_INFO[baseset_name]['LABEL_GROUP'][_config["label_sets"]]

    _log.info(f'###### Labels excluded in training : {[lb for lb in _config["exclude_cls_list"]]} ######')
    _log.info(f'###### Unseen labels evaluated in testing: {[lb for lb in test_labels]} ######')

    # Create validation dataset
    te_parent = SupervisedValidationDataset(
        which_dataset=baseset_name,
        base_dir=_config['path'][baseset_name]['data_dir'],
        idx_split=_config['eval_fold'],
        nsup=_config['task']['n_shots']
    )
    
    te_dataset = ValidationDatasetWrapper(te_parent, test_labels, _config['task']['npart'])

    _log.info('###### Set validation nodes ######')
    _scan_ids = te_parent.scan_ids
    mar_val_metric_nodes = {}

    _log.info('###### Starting validation ######')
    model.eval()

    with torch.no_grad():
        save_pred_buffer_ = {}  # indexed by support scan
        
        # Test with different support scans
        support_indices = _config.get('support_idx', [len(_scan_ids) - 1])
        
        for sup_idx in support_indices:
            support_scan_id = _scan_ids[sup_idx]
            _log.info(f'###### Set validation nodes for support id {support_scan_id} ######')
            
            mar_val_metric_nodes[support_scan_id] = Metric(
                max_label=max_label, 
                n_scans=len(te_parent.pid_curr_load) - _config['task']['n_shots']
            )
            mar_val_metric_nodes[support_scan_id].reset()

            save_pred_buffer = {}

            for curr_lb in test_labels:
                class_name = DATASET_INFO[baseset_name]["REAL_LABEL_NAME"][curr_lb]
                _log.info(f'Processing class {curr_lb} ({class_name})')
                
                te_dataset.set_curr_cls(curr_lb)
                support_batched = te_parent.get_support(
                    curr_class=curr_lb, 
                    class_idx=[curr_lb], 
                    scan_idx=[sup_idx], 
                    npart=_config['task']['npart']
                )

                # Format support data for model
                support_images = [[shot.cuda() for shot in way]
                                for way in support_batched['support_images']]
                support_fg_mask = [[shot[f'fg_mask'].float().cuda() for shot in way]
                                 for way in support_batched['support_mask']]
                support_bg_mask = [[shot[f'bg_mask'].float().cuda() for shot in way]
                                 for way in support_batched['support_mask']]

                curr_scan_count = -1
                _lb_buffer = {}

                # Process each query slice
                for idx in range(len(te_dataset)):
                    sample_batched = te_dataset[idx]
                    
                    _scan_id = sample_batched["scan_id"]
                    
                    # Skip support scan
                    if _scan_id in te_parent.potential_support_sid:
                        continue
                        
                    if sample_batched["is_start"]:
                        ii = 0
                        curr_scan_count += 1
                        _scan_id = sample_batched["scan_id"]
                        outsize = te_parent.info_by_scan[_scan_id]["array_size"]
                        outsize = (256, 256, outsize[0])
                        _pred = np.zeros(outsize)
                        _pred.fill(np.nan)
                        _labels = np.zeros(outsize)
                        _qimgs = np.zeros((3, 256, 256, outsize[-1]))
                        supimgs = np.zeros((3, 256, 256, outsize[-1]))
                        supfgs = np.zeros((256, 256, outsize[-1]))
                        _means = np.zeros((1, 1, 1, outsize[-1]))
                        _stds = np.zeros((1, 1, 1, outsize[-1]))

                    q_part = sample_batched["part_assign"]
                    query_images = [sample_batched['image'].cuda().unsqueeze(0)]  # Add batch dimension
                    query_labels = sample_batched['label'].cuda().unsqueeze(0)   # Add batch dimension

                    # Select appropriate support part
                    if q_part < len(support_images[0]):
                        sup_img_part = [[support_images[0][q_part].unsqueeze(0)]]
                        sup_fgm_part = [[support_fg_mask[0][q_part].unsqueeze(0)]]
                        sup_bgm_part = [[support_bg_mask[0][q_part].unsqueeze(0)]]
                    else:
                        # Fallback to first part if q_part is out of range
                        sup_img_part = [[support_images[0][0].unsqueeze(0)]]
                        sup_fgm_part = [[support_fg_mask[0][0].unsqueeze(0)]]
                        sup_bgm_part = [[support_bg_mask[0][0].unsqueeze(0)]]

                    # Forward pass
                    query_pred, _, _, _ = model(
                        sup_img_part, sup_fgm_part, sup_bgm_part, query_images, 
                        isval=True, val_wsize=_config["val_wsize"]
                    )

                    # Process predictions
                    query_pred = query_pred.argmax(dim=1).float().cpu().numpy()
                    
                    _pred[..., ii] = query_pred[0].copy()
                    _labels[..., ii] = query_labels[0].detach().cpu().clone().numpy()
                    _qimgs[..., ii] = query_images[0][0].detach().cpu().clone().numpy()
                    _means[..., ii] = sample_batched["mean"][0]
                    _stds[..., ii] = sample_batched["std"][0]

                    supimgs[..., ii] = sup_img_part[0][0][0].cpu().numpy()
                    supfgs[..., ii] = sup_fgm_part[0][0][0].cpu().numpy()

                    # Record metrics within valid range
                    z_id = sample_batched["z_id"]
                    z_min = sample_batched["z_min"]
                    z_max = sample_batched["z_max"]
                    z_margin = _config.get('z_margin', 0)
                    
                    if (z_id - z_max <= z_margin) and (z_id - z_min >= -1 * z_margin):
                        mar_val_metric_nodes[support_scan_id].record(
                            _pred[..., ii], _labels[..., ii], 
                            labels=[curr_lb], n_scan=curr_scan_count
                        )

                    ii += 1
                    
                    # Save scan when finished
                    if sample_batched["is_end"]:
                        _lb_buffer[_scan_id] = [
                            _pred.transpose(2, 0, 1),    # Z, H, W
                            _labels.transpose(2, 0, 1),  # Z, H, W
                            _qimgs.transpose(3, 1, 2, 0),  # Z, H, W, C
                            _means.transpose(3, 1, 2, 0),  # Z, H, W, C
                            _stds.transpose(3, 1, 2, 0),   # Z, H, W, C
                            supimgs.transpose(3, 1, 2, 0), # Z, H, W, C
                            supfgs.transpose(2, 0, 1)      # Z, H, W
                        ]

                save_pred_buffer[str(curr_lb)] = _lb_buffer

            save_pred_buffer_[support_scan_id] = save_pred_buffer

        ### Save results and create visualizations ###
        for _sup_id, saved_pred in save_pred_buffer_.items():
            for curr_lb, _preds in saved_pred.items():
                for _scan_id, item in _preds.items():
                    gif_frames = []
                    _pred, _label, _qimg, _mean, _std, _supimg, _supfg = item
                    
                    print(f"Processing scan {_scan_id}, class {curr_lb}")
                    print(f"Shapes - pred: {_pred.shape}, qimg: {_qimg.shape}, label: {_label.shape}")
                    
                    # Normalize predictions and labels for visualization
                    _pred_vis = 255 * (_pred - _pred.min(axis=(1, 2), keepdims=True)) / (
                        1.e-6 + _pred.max(axis=(1, 2), keepdims=True) - _pred.min(axis=(1, 2), keepdims=True)
                    )
                    _pred_vis = _pred_vis.astype(np.uint8)
                    
                    _label_vis = (255 * _label).astype(np.uint8)
                    _supfg_vis = (255 * _supfg).astype(np.uint8)
                    
                    # Create visualizations for each slice
                    for i in range(_pred.shape[0]):
                        # Process query image
                        qim = _qimg[i] * _std[i] + _mean[i]
                        qim = 255 * (qim - qim.min()) / (qim.max() - qim.min() + 1e-6)
                        qim = qim.astype(np.uint8)
                        qim = Image.fromarray(qim, 'RGB').convert("RGBA")

                        # Create prediction overlay (red)
                        im1 = np.stack([_pred_vis[i] * 0.8, np.zeros(_pred_vis[i].shape), np.zeros(_pred_vis[i].shape)], axis=2)
                        im1 = im1.astype(np.uint8)
                        im1 = Image.fromarray(im1, 'RGB').convert("RGBA")

                        # Create ground truth overlay (green)
                        im2 = np.stack([np.zeros(_label_vis[i].shape), _label_vis[i] * 0.8, np.zeros(_label_vis[i].shape)], axis=2)
                        im2 = im2.astype(np.uint8)
                        im2 = Image.fromarray(im2, 'RGB').convert("RGBA")
                        
                        # Blend query image with overlays
                        qim = Image.blend(qim, im1, alpha=0.5)
                        qim = Image.blend(qim, im2, alpha=0.3)

                        # Process support image
                        supimg = _supimg[i] * _std[i] + _mean[i]
                        supimg = (supimg - supimg.min()) / (supimg.max() - supimg.min()) * 255
                        supimg = supimg.astype(np.uint8)
                        supimg = Image.fromarray(supimg, 'RGB').convert("RGBA")

                        # Create support mask overlay (green)
                        supfg = _supfg_vis[i]
                        supfg = np.stack([np.zeros(supfg.shape), supfg, np.zeros(supfg.shape)], axis=2)
                        supfg = supfg.astype(np.uint8)
                        supfg = Image.fromarray(supfg, 'RGB').convert("RGBA")

                        # Blend support image with mask
                        supimg = Image.blend(supimg, supfg, alpha=0.4)

                        # Create combined image (support | query)
                        new_image = Image.new('RGB', (2 * 256, 256), (250, 250, 250))
                        new_image.paste(supimg, (0, 0))
                        new_image.paste(qim, (256, 0))

                        gif_frames.append(new_image)
                        
                        # Save individual frame
                        fid = os.path.join(f'{_run.observers[0].dir}/interm_preds', 
                                         f'scan_{_sup_id}_{_scan_id}_label_{curr_lb}_{i}.png')
                        new_image.save(fid)
                        
                    # Save as GIF
                    if gif_frames:
                        fid = os.path.join(f'{_run.observers[0].dir}/interm_preds', 
                                         f'scan_{_sup_id}_{_scan_id}_label_{curr_lb}.gif')
                        gif_frames[0].save(fid, format="GIF", append_images=gif_frames, 
                                         save_all=True, duration=500, loop=0)
                        _log.info(f'###### {fid} has been saved #####')

        del save_pred_buffer_

    # Compute and log final metrics
    for i, support_scan_id in enumerate([_scan_ids[idx] for idx in support_indices]):
        print(f"========== Metrics for Support Scan {support_scan_id} ============")
        
        # Get metrics from the metric node
        m_classDice, _, m_meanDice, _, m_rawDice = mar_val_metric_nodes[support_scan_id].get_mDice(
            labels=sorted(test_labels), n_scan=None, give_raw=True
        )

        m_classPrec, _, m_meanPrec, _, m_classRec, _, m_meanRec, _, m_rawPrec, m_rawRec = mar_val_metric_nodes[support_scan_id].get_mPrecRecall(
            labels=sorted(test_labels), n_scan=None, give_raw=True
        )

        mar_val_metric_nodes[support_scan_id].reset()

        # Log results
        _run.log_scalar('mar_val_batches_classDice', m_classDice.tolist())
        _run.log_scalar('mar_val_batches_meanDice', m_meanDice.tolist())
        _run.log_scalar('mar_val_batches_rawDice', m_rawDice.tolist())

        _run.log_scalar('mar_val_batches_classPrec', m_classPrec.tolist())
        _run.log_scalar('mar_val_batches_meanPrec', m_meanPrec.tolist())
        _run.log_scalar('mar_val_batches_rawPrec', m_rawPrec.tolist())

        _run.log_scalar('mar_val_batches_classRec', m_classRec.tolist())
        _run.log_scalar('mar_val_batches_meanRec', m_meanRec.tolist())
        _run.log_scalar('mar_val_batches_rawRec', m_rawRec.tolist())

        _log.info(f'mar_val batches classDice: {m_classDice}')
        _log.info(f'mar_val batches meanDice: {m_meanDice}')

        _log.info(f'mar_val batches classPrec: {m_classPrec}')
        _log.info(f'mar_val batches meanPrec: {m_meanPrec}')

        _log.info(f'mar_val batches classRec: {m_classRec}')
        _log.info(f'mar_val batches meanRec: {m_meanRec}')

        print(f"============ Completed Support Scan {support_scan_id} ============")

    _log.info(f'End of validation')
    return 1
