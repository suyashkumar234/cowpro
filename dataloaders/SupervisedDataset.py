"""
Final Fixed Supervised Dataset for Training CoWPro Model
Properly handles all tuple return cases from read_nii_bysitk
"""
import glob
import numpy as np
import torch
import random
import os
import copy
import json
import re
from dataloaders.common import BaseDataset
from dataloaders.dataset_utils import *
from util.utils import CircularList
import dataloaders.augutils as myaug
import dataloaders.image_transforms as myit

class SupervisedDataset(BaseDataset):
    def __init__(self, 
                which_dataset, 
                base_dir, 
                idx_split, 
                mode, 
                transform_param_limits, 
                scan_per_load, 
                nsup=1, 
                fix_length=None, 
                tile_z_dim=3, 
                exclude_list=[], 
                min_slice_distance=4,
                max_distance_ratio=1/6,
                **kwargs):
        """
        Supervised dataset for training with real ground truth masks
        """
        super(SupervisedDataset, self).__init__(base_dir)
        
        self.img_modality = DATASET_INFO[which_dataset]['MODALITY']
        self.sep = DATASET_INFO[which_dataset]['_SEP']
        self.label_name = DATASET_INFO[which_dataset]['REAL_LABEL_NAME']
        
        self.transform_param_limits = transform_param_limits
        self.is_train = True if mode == 'train' else False
        self.fix_length = fix_length
        self.nclass = len(self.label_name)
        self.tile_z_dim = tile_z_dim
        
        # Distance constraints for support-query pairs
        self.min_slice_distance = min_slice_distance
        self.max_distance_ratio = max_distance_ratio
        
        # Find scans in the data folder
        self.nsup = nsup
        self.base_dir = base_dir
        self.img_pids = [re.findall('\d+', fid)[-1] for fid in glob.glob(self.base_dir + "/image_*.nii.gz")]
        self.img_pids = CircularList(sorted(self.img_pids, key=lambda x: int(x)))
        
        # Experiment configs
        self.exclude_lbs = exclude_list
        if len(exclude_list) > 0:
            print(f'###### Dataset: the following classes has been excluded {exclude_list}######')
            
        self.idx_split = idx_split
        self.scan_ids = self.get_scanids(mode, idx_split)
        self.scan_per_load = scan_per_load
        
        self.info_by_scan = None
        self.img_lb_fids = self.organize_sample_fids()
        self.norm_func = get_normalize_op(self.img_modality, 
                                        [fid_pair['img_fid'] for _, fid_pair in self.img_lb_fids.items()])
        
        if self.is_train:
            if scan_per_load > 0:
                self.pid_curr_load = np.random.choice(self.scan_ids, replace=False, size=self.scan_per_load)
            else:
                self.pid_curr_load = self.scan_ids
        elif mode == 'val':
            self.pid_curr_load = self.scan_ids
        else:
            raise Exception
            
        self.actual_dataset = self.read_dataset()
        self.size = len(self.actual_dataset)
        self.overall_slice_by_cls = self.read_classfiles()
        
        print("###### Initial scans loaded: ######")
        print(self.pid_curr_load)
        
        # Setup augmentation
        print("#### Setting up augmentation population ####")
        self.affine = self.transform_param_limits.get('affine', 0)
        self.alpha = self.transform_param_limits.get('elastic', {'alpha': 0})['alpha']
        self.sigma = self.transform_param_limits.get('elastic', {'sigma': 0})['sigma']
        self.gamma_range = self.transform_param_limits['gamma_range']
        
        self.randomaffine = myit.RandomAffine(
            self.affine.get('rotate'),
            self.affine.get('shift'),
            self.affine.get('shear'),
            self.affine.get('scale'),
            self.affine.get('scale_iso', True),
            order=3
        )
        
        self.elastic = myit.ElasticTransform(self.alpha, self.sigma)
        
    def get_scanids(self, mode, idx_split):
        """Load scans by train-test split"""
        val_ids = copy.deepcopy(self.img_pids[self.sep[idx_split]: self.sep[idx_split + 1] + self.nsup])
        if mode == 'train':
            return [ii for ii in self.img_pids if ii not in val_ids]
        elif mode == 'val':
            return val_ids
            
    def organize_sample_fids(self):
        """Organize file paths for images and labels"""
        out_list = {}
        for curr_id in self.scan_ids:
            curr_dict = {}
            _img_fid = os.path.join(self.base_dir, f'image_{curr_id}.nii.gz')
            _lb_fid = os.path.join(self.base_dir, f'label_{curr_id}.nii.gz')
            
            curr_dict["img_fid"] = _img_fid
            curr_dict["lbs_fid"] = _lb_fid
            out_list[str(curr_id)] = curr_dict
        return out_list

    def safe_read_nii(self, file_path, peel_info=False):
        """
        Safely read NII file and always return just the image array
        """
        try:
            print(f"    DEBUG: Reading {file_path} with peel_info={peel_info}")
            result = read_nii_bysitk(file_path, peel_info=peel_info)
            print(f"    DEBUG: read_nii_bysitk returned type: {type(result)}")
            
            # Always handle as if it might be a tuple
            if isinstance(result, tuple):
                print(f"    DEBUG: Result is tuple with length: {len(result)}")
                # If it's a tuple, extract the image (first element)
                img = result[0]
                info = result[1] if len(result) > 1 else None
                print(f"    DEBUG: Extracted img type: {type(img)}")
                
                # Ensure img is numpy array
                if not isinstance(img, np.ndarray):
                    raise ValueError(f"Expected numpy array, got {type(img)}")
                    
                if peel_info:
                    return img, info
                else:
                    return img
            else:
                print(f"    DEBUG: Result is not tuple, type: {type(result)}")
                # If it's not a tuple, it's just the image
                if not isinstance(result, np.ndarray):
                    raise ValueError(f"Expected numpy array, got {type(result)}")
                    
                if peel_info:
                    return result, None
                else:
                    return result
                    
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            raise
        
    def read_dataset(self):
        """Read images and labels into memory and build slice indexing"""
        out_list = []
        self.scan_z_idx = {}
        self.info_by_scan = {}
        self.scan_slice_counts = {}  # Track number of slices per scan
        glb_idx = 0
        
        for scan_id, itm in self.img_lb_fids.items():
            if scan_id not in self.pid_curr_load:
                continue
                
            print(f"Loading scan {scan_id}...")
            
            try:
                # Read image with metadata
                print(f"  DEBUG: About to read image...")
                img_result = self.safe_read_nii(itm["img_fid"], peel_info=True)
                print(f"  DEBUG: img_result type: {type(img_result)}")
                
                if isinstance(img_result, tuple):
                    img, _info = img_result
                    print(f"  DEBUG: Unpacked tuple - img type: {type(img)}, info type: {type(_info)}")
                else:
                    img = img_result
                    _info = None
                    print(f"  DEBUG: Direct assignment - img type: {type(img)}")
                    
                print(f"  Image shape: {img.shape}")
                print(f"  Image type: {type(img)}")
                
                # Verify img is numpy array
                if not isinstance(img, np.ndarray):
                    print(f"  ERROR: Image is not numpy array, type: {type(img)}")
                    continue
                    
                img = img.transpose(1, 2, 0)
                print(f"  After transpose: {img.shape}")
                
                self.info_by_scan[scan_id] = _info
                
                img = np.float32(img)
                normalized_result = self.norm_func(img)
                print(f"  DEBUG: norm_func returned type: {type(normalized_result)}")
                
                # Handle case where norm_func might return tuple
                if isinstance(normalized_result, tuple):
                    img = normalized_result[0]
                    print(f"  DEBUG: Extracted from norm_func tuple - img type: {type(img)}")
                else:
                    img = normalized_result
                    print(f"  DEBUG: Direct assignment from norm_func - img type: {type(img)}")
                    
                print(f"  DEBUG: After normalization - img type: {type(img)}, shape: {img.shape}")
                
                # Read label
                print(f"  DEBUG: About to read label...")
                lb_result = self.safe_read_nii(itm["lbs_fid"], peel_info=False)
                print(f"  DEBUG: lb_result type: {type(lb_result)}")
                
                if isinstance(lb_result, tuple):
                    lb = lb_result[0]
                    print(f"  DEBUG: Extracted from tuple - lb type: {type(lb)}")
                else:
                    lb = lb_result
                    print(f"  DEBUG: Direct assignment - lb type: {type(lb)}")
                    
                print(f"  Label shape: {lb.shape}")
                print(f"  Label type: {type(lb)}")
                
                # Verify lb is numpy array
                if not isinstance(lb, np.ndarray):
                    print(f"  ERROR: Label is not numpy array, type: {type(lb)}")
                    continue
                    
                lb = lb.transpose(1, 2, 0)
                print(f"  Label after transpose: {lb.shape}")
                lb = np.float32(lb)
                print(f"  DEBUG: Before shape check - img type: {type(img)}, lb type: {type(lb)}")
                
                # Ensure consistent shape and crop to 256x256
                if len(img.shape) != 3 or len(lb.shape) != 3:
                    print(f"  ERROR: Invalid shapes - img: {img.shape}, lb: {lb.shape}")
                    continue
                    
            except Exception as e:
                print(f"  ERROR loading scan {scan_id}: {e}")
                continue
                
            # Crop to 256x256
            img = img[:256, :256, :]
            lb = lb[:256, :256, :]
            
            if img.shape[-1] != lb.shape[-1]:
                print(f"  WARNING: Image and label have different number of slices: {img.shape[-1]} vs {lb.shape[-1]}")
                min_slices = min(img.shape[-1], lb.shape[-1])
                img = img[:, :, :min_slices]
                lb = lb[:, :, :min_slices]
                print(f"  Truncated to {min_slices} slices")
            
            # Store slice count for distance calculations
            self.scan_slice_counts[scan_id] = img.shape[-1]
            
            self.scan_z_idx[scan_id] = [-1 for _ in range(img.shape[-1])]
            
            # Process all slices
            for ii in range(img.shape[-1]):
                is_start = (ii == 0)
                is_end = (ii == img.shape[-1] - 1)
                
                out_list.append({
                    "img": img[..., ii: ii + 1],
                    "lb": lb[..., ii: ii + 1],
                    "is_start": is_start,
                    "is_end": is_end,
                    "nframe": img.shape[-1],
                    "scan_id": scan_id,
                    "z_id": ii
                })
                
                self.scan_z_idx[scan_id][ii] = glb_idx
                glb_idx += 1
                
            print(f"  Loaded {img.shape[-1]} slices from scan {scan_id}")
                
        print(f"Total dataset size: {len(out_list)} slices")
        return out_list
        
    def read_classfiles(self):
        """Load class-slice indexing files"""
        classmap_path = os.path.join(self.base_dir, 'classmap_1.json')
        print(f"Loading classmap from: {classmap_path}")
        
        if not os.path.exists(classmap_path):
            print(f"ERROR: Classmap file not found: {classmap_path}")
            raise FileNotFoundError(f"Classmap file not found: {classmap_path}")
            
        with open(classmap_path, 'r') as fopen:
            cls_map = json.load(fopen)
            fopen.close()
            
        print(f"Loaded classmap with {len(cls_map)} classes")
        for class_name, scan_data in cls_map.items():
            print(f"  {class_name}: {len(scan_data)} scans")
            
        return cls_map
        
    def get_valid_support_indices(self, query_scan_id, query_z_id, target_class):
        """Get valid support slice indices based on distance constraints"""
        if target_class not in self.overall_slice_by_cls:
            return []
            
        if query_scan_id not in self.overall_slice_by_cls[target_class]:
            return []
            
        # Get all slices containing the target class in the same scan
        class_slices = self.overall_slice_by_cls[target_class][query_scan_id]
        total_slices = self.scan_slice_counts[query_scan_id]
        
        # Calculate distance constraints
        max_distance = int(total_slices * self.max_distance_ratio)
        max_distance = max(max_distance, self.min_slice_distance)  # Ensure minimum distance
        
        valid_indices = []
        for slice_idx in class_slices:
            distance = abs(slice_idx - query_z_id)
            if self.min_slice_distance <= distance <= max_distance:
                valid_indices.append(slice_idx)
                
        return valid_indices
        
    def sample_support_query_pair(self):
        """Sample a valid support-query pair based on distance constraints"""
        max_attempts = 100
        attempts = 0
        
        while attempts < max_attempts:
            # Sample a random query slice
            query_idx = random.randint(0, len(self.actual_dataset) - 1)
            query_data = self.actual_dataset[query_idx]
            query_scan_id = query_data["scan_id"]
            query_z_id = query_data["z_id"]
            
            # Get available classes in this slice (excluding background)
            query_labels = np.unique(query_data["lb"])
            foreground_classes = [int(lb) for lb in query_labels if lb > 0 and lb not in self.exclude_lbs]
            
            if not foreground_classes:
                attempts += 1
                continue
                
            # Sample a target class
            target_class = random.choice(foreground_classes)
            class_name = self.label_name[target_class]
            
            # Get valid support indices
            valid_support_indices = self.get_valid_support_indices(
                query_scan_id, query_z_id, class_name
            )
            
            if valid_support_indices:
                # Sample a support slice
                support_z_id = random.choice(valid_support_indices)
                support_global_idx = self.scan_z_idx[query_scan_id][support_z_id]
                support_data = self.actual_dataset[support_global_idx]
                
                return support_data, query_data, target_class
                
            attempts += 1
            
        # Fallback: return a random pair if no valid pair found
        print("Warning: Could not find valid support-query pair with distance constraints, using fallback")
        query_idx = random.randint(0, len(self.actual_dataset) - 1)
        
        # Try to find a support from the same scan but different slice
        query_data = self.actual_dataset[query_idx]
        query_scan_id = query_data["scan_id"]
        query_z_id = query_data["z_id"]
        
        # Find other slices from the same scan
        same_scan_indices = [i for i, data in enumerate(self.actual_dataset) 
                           if data["scan_id"] == query_scan_id and data["z_id"] != query_z_id]
        
        if same_scan_indices:
            support_idx = random.choice(same_scan_indices)
            support_data = self.actual_dataset[support_idx]
        else:
            support_idx = random.randint(0, len(self.actual_dataset) - 1)
            support_data = self.actual_dataset[support_idx]
        
        # Get a random foreground class from query
        query_labels = np.unique(query_data["lb"])
        foreground_classes = [int(lb) for lb in query_labels if lb > 0 and lb not in self.exclude_lbs]
        target_class = random.choice(foreground_classes) if foreground_classes else 1
        
        return support_data, query_data, target_class
        
    def gamma_transform(self, img):
        """Apply gamma transformation for intensity augmentation"""
        if isinstance(self.gamma_range, tuple):
            gamma = np.random.rand() * (self.gamma_range[1] - self.gamma_range[0]) + self.gamma_range[0]
            cmin = img.min()
            irange = (img.max() - cmin + 1e-5)
            
            img = img - cmin + 1e-5
            img = irange * np.power(img * 1.0 / irange, gamma)
            img = img + cmin
            
        return img, gamma
        
    def transform_img_lb(self, comp, c_label, c_img, nclass):
        """Apply augmentation transforms to image and label pair"""
        comp = copy.deepcopy(comp)
        assert c_img + 1 == comp.shape[-1], "only allow single slice 2D label"
        
        # Convert label to one-hot
        _label = comp[..., c_img]
        _h_label = np.float32(np.arange(nclass) == (_label[..., None]))
        comp = np.concatenate([comp[..., :c_img], _h_label], -1)
        
        # Affine transformations
        affine_params = self.randomaffine.build_M(comp.shape[:2])
        comp = self.randomaffine(comp, affine_params)
        
        # Elastic transformation
        comp, dx_params, dy_params = self.elastic(comp)
        
        # Extract transformed components
        t_label_h = comp[..., c_img:]
        t_label_h = np.rint(t_label_h)
        assert t_label_h.max() <= 1
        t_img = comp[..., 0:c_img]
        
        # Intensity transform
        t_img, gamma = self.gamma_transform(t_img)
        
        # Convert back to compact label
        t_label = np.expand_dims(np.argmax(t_label_h, axis=-1), -1)
        
        return t_img, t_label
        
    def __getitem__(self, index):
        """Get a training sample with support-query pair"""
        # Sample valid support-query pair
        support_data, query_data, target_class = self.sample_support_query_pair()
        
        # Create binary masks for the target class
        support_mask = np.float32(support_data["lb"] == target_class)
        query_mask = np.float32(query_data["lb"] == target_class)
        
        # Apply augmentations
        support_comp = np.concatenate([support_data["img"], support_mask], axis=-1)
        support_img, support_lb = self.transform_img_lb(
            support_comp, c_img=1, c_label=1, nclass=2
        )
        
        query_comp = np.concatenate([query_data["img"], query_mask], axis=-1)
        query_img, query_lb = self.transform_img_lb(
            query_comp, c_img=1, c_label=1, nclass=2
        )
        
        # Convert to tensors
        support_img = torch.from_numpy(np.transpose(support_img, (2, 0, 1)))
        support_lb = torch.from_numpy(support_lb.squeeze(-1))
        query_img = torch.from_numpy(np.transpose(query_img, (2, 0, 1)))
        query_lb = torch.from_numpy(query_lb.squeeze(-1))
        
        # Tile along channel dimension if needed
        if self.tile_z_dim:
            support_img = support_img.repeat([self.tile_z_dim, 1, 1])
            query_img = query_img.repeat([self.tile_z_dim, 1, 1])
            
        # Create masks for few-shot learning format
        support_fg_mask = torch.where(support_lb == 1, 
                                    torch.ones_like(support_lb), 
                                    torch.zeros_like(support_lb))
        support_bg_mask = torch.where(support_lb == 0,
                                    torch.ones_like(support_lb),
                                    torch.zeros_like(support_lb))
        
        return {
            'class_ids': [[target_class]],
            'support_images': [[support_img]],
            'support_mask': [[{
                'fg_mask': support_fg_mask,
                'bg_mask': support_bg_mask
            }]],
            'query_images': [query_img],
            'query_labels': [query_lb],
            'support_scan_id': support_data["scan_id"],
            'support_z_id': support_data["z_id"],
            'query_scan_id': query_data["scan_id"],
            'query_z_id': query_data["z_id"],
            'target_class': target_class
        }
        
    def __len__(self):
        """Return dataset length"""
        if self.fix_length is not None:
            return self.fix_length
        else:
            return len(self.actual_dataset) * 2  # Allow multiple epochs over the data
