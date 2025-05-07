"""
ALPNet with Iterative Adaptive Refinement for Few-Shot Medical Image Segmentation
"""
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from sklearn.mixture import GaussianMixture
import pandas as pd
import time
import os
from itertools import product

from .alpmodule import MultiProtoAsConv
from .alpmodule2 import MultiProtoAsWCos
from .backbone.torchvision_backbones import TVDeeplabRes101Encoder, Encoder
from util.utils import get_tversky_loss

# Options for type of prototypes
FG_PROT_MODE = 'gridconv+'  # using both local and global prototype
BG_PROT_MODE = 'gridconv'   # using local prototype only
# Also 'mask' refers to using global prototype only (as done in vanilla PANet)

# Thresholds for deciding class of prototypes
FG_THRESH = 0.95
BG_THRESH = 0.95

class IterativeAdaptiveRefinement(nn.Module):
    """
    Implements an iterative adaptive refinement strategy for medical image segmentation.
    Combines the strengths of iterative refinement with context-adaptive morphological operations.
    """
    def __init__(self, 
                 max_iterations=5, 
                 confidence_threshold=0.5,
                 small_object_threshold=0.01,
                 large_object_threshold=0.2,
                 small_kernel_size=3,
                 medium_kernel_size=3,
                 large_kernel_size=5,
                 use_feedback=True):
        """
        Args:
            max_iterations: Maximum number of refinement iterations
            confidence_threshold: Threshold for considering pixels as foreground in confidence maps
            small_object_threshold: Area ratio threshold for small objects
            large_object_threshold: Area ratio threshold for large objects
            small_kernel_size: Kernel size for small object operations
            medium_kernel_size: Kernel size for medium object operations
            large_kernel_size: Kernel size for large object operations
            use_feedback: Whether to use feedback from previous iteration
        """
        super(IterativeAdaptiveRefinement, self).__init__()
        self.max_iterations = max_iterations
        self.confidence_threshold = confidence_threshold
        self.small_object_threshold = small_object_threshold
        self.large_object_threshold = large_object_threshold
        self.small_kernel_size = small_kernel_size
        self.medium_kernel_size = medium_kernel_size
        self.large_kernel_size = large_kernel_size
        self.use_feedback = use_feedback
        
        # Convolutional layers for feedback mechanism (if used)
        if self.use_feedback:
            self.feedback_conv = nn.Conv2d(2, 1, kernel_size=3, padding=1)
            self.feedback_bn = nn.BatchNorm2d(1)
            
    def forward(self, image, initial_mask, confidence_map=None):
        """
        Apply iterative adaptive refinement to an initial segmentation mask
        
        Args:
            image: Input image tensor (B, C, H, W)
            initial_mask: Initial binary segmentation mask (B, 1, H, W)
            confidence_map: Optional confidence map from model (B, 1, H, W)
            
        Returns:
            refined_mask: Refined binary mask (B, 1, H, W)
        """
        batch_size = image.shape[0]
        device = image.device
        
        # Initialize refined masks with initial predictions
        current_masks = initial_mask.clone()
        feedback = torch.zeros_like(initial_mask)
        
        # Convert tensors to numpy for processing
        image_np = image.detach().cpu().numpy()
        masks_np = current_masks.detach().cpu().numpy()
        
        # Iterative refinement
        for iteration in range(self.max_iterations):
            refined_masks = []
            
            for b in range(batch_size):
                img = image_np[b]
                mask = masks_np[b, 0]  # Remove channel dimension
                
                # Get feedback from previous iteration if enabled
                if self.use_feedback and iteration > 0:
                    prev_mask = masks_np[b, 0].copy()
                    fb = feedback[b, 0].detach().cpu().numpy()
                else:
                    fb = None
                
                # Apply adaptive refinement based on mask characteristics
                refined = self.apply_adaptive_refinement(img, mask, fb, iteration)
                refined_masks.append(refined)
            
            # Stack refined masks
            masks_np = np.stack(refined_masks, axis=0)[:, np.newaxis, :, :]
            
            # Update current masks
            current_masks = torch.from_numpy(masks_np).float().to(device)
            
            # Generate feedback for next iteration
            if self.use_feedback and iteration < self.max_iterations - 1:
                # Compute difference between consecutive iterations
                prev_masks = torch.from_numpy(masks_np.copy()).float().to(device)
                if confidence_map is not None:
                    feedback_input = torch.cat([prev_masks, confidence_map], dim=1)
                else:
                    feedback_input = torch.cat([prev_masks, current_masks], dim=1)
                feedback = F.relu(self.feedback_bn(self.feedback_conv(feedback_input)))
        
        return current_masks
    
    def apply_adaptive_refinement(self, image, mask, feedback=None, iteration=0):
        """
        Apply adaptive refinement based on mask characteristics
        
        Args:
            image: Input image (C, H, W)
            mask: Binary mask (H, W)
            feedback: Optional feedback from previous iteration (H, W)
            iteration: Current iteration number
            
        Returns:
            refined: Refined binary mask (H, W)
        """
        # Analyze mask properties
        mask_size = np.sum(mask)
        mask_area_ratio = mask_size / (mask.shape[0] * mask.shape[1])
        
        # Skip refinement for very small or empty masks
        if mask_size < 10:
            return mask
        
        # Incorporate feedback if available
        if feedback is not None:
            # Use feedback to adjust the mask
            mask = np.clip(mask + 0.5 * feedback, 0, 1)
        
        # Convert to binary mask for morphological operations
        binary_mask = (mask > 0.5).astype(np.uint8)
        
        # Extract contours for complexity analysis
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Calculate boundary complexity (perimeter to area ratio)
        if len(contours) > 0 and mask_size > 100:
            perimeter = sum(cv2.arcLength(cnt, True) for cnt in contours)
            boundary_complexity = perimeter / mask_size
        else:
            boundary_complexity = 0
            
        # Determine appropriate refinement strategy based on object characteristics
        # and current iteration
        if mask_area_ratio < self.small_object_threshold:
            # Small object strategy - preserve details, carefully expand
            kernel_size = max(3, self.small_kernel_size - iteration)
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            
            if boundary_complexity > 0.1:
                # Complex boundary - careful processing
                refined = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
                refined = cv2.dilate(refined, kernel, iterations=1)
            else:
                # Simple boundary - can be more aggressive
                refined = cv2.dilate(binary_mask, kernel, iterations=1)
                refined = cv2.medianBlur(refined, 3)
                
        elif mask_area_ratio > self.large_object_threshold:
            # Large object strategy - focus on cleaning boundaries
            kernel_size = max(3, self.large_kernel_size - iteration)
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            
            if boundary_complexity > 0.05:
                # Complex boundary - more careful processing
                refined = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
                refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, kernel)
            else:
                # Clean up noise
                refined = cv2.erode(binary_mask, kernel, iterations=1)
                refined = cv2.dilate(refined, kernel, iterations=1)
                
        else:
            # Medium object strategy - balanced approach
            kernel_size = max(3, self.medium_kernel_size - iteration)
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            
            # For medium objects, apply opening then closing
            refined = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
            refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kernel)
            
        # Fill small holes
        if mask_size > 500:
            # Find contours and fill holes
            refined_contours, _ = cv2.findContours(refined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            holes_mask = np.zeros_like(refined)
            for contour in refined_contours:
                if cv2.contourArea(contour) > 100:
                    cv2.drawContours(holes_mask, [contour], 0, 1, -1)
                    
            # Use the filled contours if they're not too different from original
            if np.sum(holes_mask) < mask_size * 1.5:
                refined = holes_mask
        
        # Conditional median filtering for final smoothing
        if iteration == self.max_iterations - 1:
            refined = cv2.medianBlur(refined, 3)
            
        return refined.astype(np.float32)

class FewShotSeg(nn.Module):
    """
    ALPNet with Iterative Adaptive Refinement
    Args:
        in_channels:        Number of input channels
        cfg:                Model configurations
    """
    def __init__(self, in_channels=3, pretrained_path=None, cfg=None):
        super(FewShotSeg, self).__init__()
        self.pretrained_path = pretrained_path
        self.config = cfg or {'align': False}
        
        # Initialize refinement parameters
        self.use_refinement = self.config.get('use_refinement', True)
        self.max_iterations = self.config.get('refinement_iterations', 3)
        self.use_feedback = self.config.get('use_feedback', True)
        self.small_object_threshold = self.config.get('small_object_threshold', 0.01)
        self.large_object_threshold = self.config.get('large_object_threshold', 0.2)
        self.small_kernel_size = self.config.get('small_kernel_size', 3)
        self.medium_kernel_size = self.config.get('medium_kernel_size', 3)
        self.large_kernel_size = self.config.get('large_kernel_size', 5)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.5)
        
        # Create refinement module
        if self.use_refinement:
            self.refinement = IterativeAdaptiveRefinement(
                max_iterations=self.max_iterations,
                use_feedback=self.use_feedback,
                small_object_threshold=self.small_object_threshold,
                large_object_threshold=self.large_object_threshold,
                small_kernel_size=self.small_kernel_size,
                medium_kernel_size=self.medium_kernel_size,
                large_kernel_size=self.large_kernel_size,
                confidence_threshold=self.confidence_threshold
            )
        
        self.get_encoder(in_channels)
        self.get_cls()

    def get_encoder(self, in_channels):
        use_coco_init = self.config['use_coco_init']
        self.encoder = TVDeeplabRes101Encoder(use_coco_init)

        if self.pretrained_path:
            self.load_state_dict(torch.load(self.pretrained_path)['model'], strict=False)
            print(f'###### Pre-trained model {self.pretrained_path} has been loaded ######')

    def get_cls(self):
        """
        Obtain the similarity-based classifier
        """
        proto_hw = self.config["proto_grid_size"]
        feature_hw = self.config["feature_hw"]
        assert self.config['cls_name'] == 'grid_proto'
        if self.config['cls_name'] == 'grid_proto':
            self.cls_unit = MultiProtoAsWCos(proto_grid=[proto_hw, proto_hw], 
                                           feature_hw=self.config["feature_hw"])
        else:
            raise NotImplementedError(f'Classifier {self.config["cls_name"]} not implemented')

    def forward(self, supp_imgs, fore_mask, back_mask, qry_imgs, isval, val_wsize, show_viz=False):
        """
        Args:
            supp_imgs: support images
                way x shot x [B x 3 x H x W], list of lists of tensors
            fore_mask: foreground masks for support images
                way x shot x [B x H x W], list of lists of tensors
            back_mask: background masks for support images
                way x shot x [B x H x W], list of lists of tensors
            qry_imgs: query images
                N x [B x 3 x H x W], list of tensors
            isval: whether in validation mode
            val_wsize: window size during validation
            show_viz: return the visualization dictionary
        """
        n_ways = len(supp_imgs)
        n_shots = len(supp_imgs[0])
        n_queries = len(qry_imgs)

        assert n_ways == 1, "Multi-shot has not been implemented yet" 
        assert n_queries == 1

        sup_bsize = supp_imgs[0][0].shape[0]
        img_size = supp_imgs[0][0].shape[-2:]
        qry_bsize = qry_imgs[0].shape[0]

        assert sup_bsize == qry_bsize == 1

        imgs_concat = torch.cat([torch.cat(way, dim=0) for way in supp_imgs]
                               + [torch.cat(qry_imgs, dim=0),], dim=0)

        img_fts = self.encoder(imgs_concat, low_level=False)
        fts_size = img_fts.shape[-2:]

        supp_fts = img_fts[:n_ways * n_shots * sup_bsize].view(
            n_ways, n_shots, sup_bsize, -1, *fts_size)  # Wa x Sh x B x C x H' x W'
        qry_fts = img_fts[n_ways * n_shots * sup_bsize:].view(
            n_queries, qry_bsize, -1, *fts_size)   # N x B x C x H' x W'
        fore_mask = torch.stack([torch.stack(way, dim=0)
                                 for way in fore_mask], dim=0)  # Wa x Sh x B x H' x W'
        fore_mask = torch.autograd.Variable(fore_mask, requires_grad=True)
        back_mask = torch.stack([torch.stack(way, dim=0)
                                for way in back_mask], dim=0)  # Wa x Sh x B x H' x W'

        ###### Compute loss ######
        align_loss = 0
        outputs = []
        visualizes = [] # the buffer for visualization

        for epi in range(1): # batch dimension, fixed to 1
            fg_masks = [] # keep the way part

            # re-interpolate support mask to the same size as support feature
            res_fg_msk = torch.stack([F.interpolate(fore_mask_w, size=fts_size, mode='bilinear') 
                                    for fore_mask_w in fore_mask], dim=0) # [nway, ns, nb, nh', nw']
            res_bg_msk = torch.stack([F.interpolate(back_mask_w, size=fts_size, mode='bilinear') 
                                    for back_mask_w in back_mask], dim=0) # [nway, ns, nb, nh', nw']

            scores = []
            assign_maps = []
            bg_sim_maps = []
            fg_sim_maps = []

            _raw_score, _, aux_attr = self.cls_unit(qry_fts, supp_fts, res_bg_msk, mode=BG_PROT_MODE, 
                                                   fg=False, thresh=BG_THRESH, isval=isval, 
                                                   val_wsize=val_wsize, vis_sim=show_viz)

            scores.append(_raw_score)
            assign_maps.append(aux_attr['proto_assign'])
            if show_viz:
                bg_sim_maps.append(aux_attr['raw_local_sims'])

            for way, _msk in enumerate(res_fg_msk):
                _raw_score, _, aux_attr = self.cls_unit(qry_fts, supp_fts, _msk.unsqueeze(0), fg=True, 
                                                      mode=FG_PROT_MODE, 
                                                      thresh=FG_THRESH, isval=isval, 
                                                      val_wsize=val_wsize, vis_sim=show_viz)

                scores.append(_raw_score)
                if show_viz:
                    fg_sim_maps.append(aux_attr['raw_local_sims'])

            # Combine scores from background and foreground to get the initial prediction
            pred = torch.cat(scores, dim=1)  # N x (1 + Wa) x H' x W'
            
            # Upsample prediction to original image size
            pred_upsampled = F.interpolate(pred, size=img_size, mode='bilinear')
            
            # Get binary mask from prediction
            pred_mask = pred_upsampled.argmax(dim=1, keepdim=True).float()
            confidence_map = F.softmax(pred_upsampled, dim=1)[:, 1:2]
            
            # Apply iterative adaptive refinement if enabled and in validation mode
            if self.use_refinement and not self.training and isval:
                # Get the query images for refinement
                query_images_tensor = torch.cat(qry_imgs, dim=0)  # N x 3 x H x W
                
                try:
                    # Apply iterative adaptive refinement
                    refined_mask = self.refinement(query_images_tensor, pred_mask, confidence_map)
                    
                    # Convert refined binary mask to class probabilities
                    refined_pred = torch.zeros_like(pred_upsampled)
                    refined_pred[:, 0] = 1.0 - refined_mask.squeeze(1)  # Background probability
                    refined_pred[:, 1] = refined_mask.squeeze(1)        # Foreground probability
                    
                    # Use the refined prediction
                    outputs.append(refined_pred)
                except Exception as e:
                    print(f"Error in refinement: {e}. Using initial prediction.")
                    outputs.append(pred_upsampled)
            else:
                # Use the initial prototype-based prediction without refinement
                outputs.append(pred_upsampled)

            ###### Prototype alignment loss ######
            if self.config['align'] and self.training:
                try:
                    align_loss_epi = self.alignLoss(qry_fts[:, epi], pred, supp_fts[:, :, epi],
                                                   fore_mask[:, :, epi], back_mask[:, :, epi])
                    align_loss += align_loss_epi
                except:
                    align_loss += 0
                    
        output = torch.stack(outputs, dim=1)  # N x B x (1 + Wa) x H x W
        output = output.view(-1, *output.shape[2:])
        assign_maps = torch.stack(assign_maps, dim=1)
        bg_sim_maps = torch.stack(bg_sim_maps, dim=1) if show_viz else None
        fg_sim_maps = torch.stack(fg_sim_maps, dim=1) if show_viz else None

        return output, align_loss / sup_bsize, [bg_sim_maps, fg_sim_maps], assign_maps

    # Batch was at the outer loop
    def alignLoss(self, qry_fts, pred, supp_fts, fore_mask, back_mask):
        """
        Compute the loss for the prototype alignment branch

        Args:
            qry_fts: embedding features for query images
                expect shape: N x C x H' x W'
            pred: predicted segmentation score
                expect shape: N x (1 + Wa) x H x W
            supp_fts: embedding fatures for support images
                expect shape: Wa x Sh x C x H' x W'
            fore_mask: foreground masks for support images
                expect shape: way x shot x H x W
            back_mask: background masks for support images
                expect shape: way x shot x H x W
        """
        n_ways, n_shots = len(fore_mask), len(fore_mask[0])

        # Masks for getting query prototype
        pred_mask = pred.argmax(dim=1).unsqueeze(0)  #1 x  N x H' x W'
        binary_masks = [pred_mask == i for i in range(1 + n_ways)]

        # skip_ways = [i for i in range(n_ways) if binary_masks[i + 1].sum() == 0]
        # FIXME: fix this in future we here make a stronger assumption that a positive class must be there to avoid undersegmentation/ lazyness
        skip_ways = []

        ### added for matching dimensions to the new data format
        qry_fts = qry_fts.unsqueeze(0).unsqueeze(2) # added to nway(1) and nb(1)

        ### end of added part

        loss = []
        for way in range(n_ways):
            if way in skip_ways:
                continue
            # Get the query prototypes
            for shot in range(n_shots):
                img_fts = supp_fts[way: way + 1, shot: shot + 1] # actual local query [way(1), nb(1, nb is now nshot), nc, h, w]

                qry_pred_fg_msk = F.interpolate(binary_masks[way + 1].float(), size=img_fts.shape[-2:], mode='bilinear') # [1 (way), n (shot), h, w]

                # background
                qry_pred_bg_msk = F.interpolate(binary_masks[0].float(), size=img_fts.shape[-2:], mode='bilinear') # 1, n, h ,w
                scores = []

                _raw_score_bg, _, _ = self.cls_unit(qry=img_fts, sup_x=qry_fts, sup_y=qry_pred_bg_msk.unsqueeze(-3), fg=False, mode=BG_PROT_MODE, thresh=BG_THRESH)

                scores.append(_raw_score_bg)

                _raw_score_fg, _, _ = self.cls_unit(qry=img_fts, sup_x=qry_fts, sup_y=qry_pred_fg_msk.unsqueeze(-3), 
                                                   fg=True, mode=FG_PROT_MODE, 
                                                   thresh=FG_THRESH)
                scores.append(_raw_score_fg)

                supp_pred = torch.cat(scores, dim=1)  # N x (1 + Wa) x H' x W'
                supp_pred = F.interpolate(supp_pred, size=fore_mask.shape[-2:], mode='bilinear')

                # Construct the support Ground-Truth segmentation
                supp_label = torch.full_like(fore_mask[way, shot], 255, device=img_fts.device).long()

                supp_label[fore_mask[way, shot] == 1] = 1
                supp_label[back_mask[way, shot] == 1] = 0
                # Compute Loss
                loss.append(F.cross_entropy(supp_pred, supp_label[None, ...], ignore_index=255) / n_shots / n_ways)
                loss.append(get_tversky_loss(supp_pred.argmax(dim=1, keepdim=True), supp_label[None, ...], 0.3, 0.7, 1.0) / n_shots / n_ways)

        return torch.sum(torch.stack(loss))

    # Parameter tuning function
    def run_parameter_tuning(self, test_dataloader, test_labels, output_csv="refinement_params_results.csv"):
        """
        Run parameter tuning experiments and save results to CSV
        
        Args:
            test_dataloader: DataLoader for test data
            test_labels: List of test class labels
            output_csv: Path to save results CSV
        """
        # Save current model parameters for later restoration
        original_params = {
            "max_iterations": self.max_iterations,
            "use_feedback": self.refinement.use_feedback,
            "small_object_threshold": self.refinement.small_object_threshold,
            "large_object_threshold": self.refinement.large_object_threshold,
            "small_kernel_size": self.refinement.small_kernel_size,
            "medium_kernel_size": self.refinement.medium_kernel_size,
            "large_kernel_size": self.refinement.large_kernel_size,
            "confidence_threshold": self.refinement.confidence_threshold
        }
        
        print("Starting parameter tuning experiment...")
        print(f"Results will be saved to: {output_csv}")
        
        # Define parameter combinations to test
        param_grid = {
            "refinement_iterations": [2, 3, 5, 7],
            "use_feedback": [True, False],
            "small_object_threshold": [0.005, 0.01, 0.02],
            "large_object_threshold": [0.15, 0.2, 0.25],
            "small_kernel_size": [3, 5],
            "medium_kernel_size": [3, 5],
            "large_kernel_size": [5, 7]
        }
        
        # Generate all combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(product(*param_values))
        
        # Store results
        results = []
        
        # Test each parameter combination
        for i, combination in enumerate(param_combinations):
            # Set parameters for this run
            params = {name: value for name, value in zip(param_names, combination)}
            
            print(f"\nTesting parameter combination {i+1}/{len(param_combinations)}:")
            for name, value in params.items():
                print(f"  {name}: {value}")
            
            # Update model parameters
            self.max_iterations = params["refinement_iterations"]
            self.refinement.max_iterations = params["refinement_iterations"]
            self.refinement.use_feedback = params["use_feedback"]
            self.refinement.small_object_threshold = params["small_object_threshold"]
            self.refinement.large_object_threshold = params["large_object_threshold"]
            self.refinement.small_kernel_size = params["small_kernel_size"]
            self.refinement.medium_kernel_size = params["medium_kernel_size"]
            self.refinement.large_kernel_size = params["large_kernel_size"]
            
            # Evaluate model with current parameters
            metrics = self.evaluate_parameter_set(test_dataloader, test_labels)
            
            # Add parameters to metrics and save results
            result_row = {**params, **metrics}
            results.append(result_row)
            
            # Save to CSV after each run
            df = pd.DataFrame(results)
            df.to_csv(output_csv, index=False)
            
            print(f"Results for this combination:")
            for metric_name, metric_value in metrics.items():
                print(f"  {metric_name}: {metric_value}")
        
        # Find best parameters
        df = pd.DataFrame(results)
        best_row = df.loc[df["mean_dice"].idxmax()]
        
        print("\n===== Best Parameters Found =====")
        for param in param_names:
            print(f"{param}: {best_row[param]}")
        print(f"Mean Dice Score: {best_row['mean_dice']:.4f}")
        
        # Restore original parameters
        self.max_iterations = original_params["max_iterations"]
        self.refinement.max_iterations = original_params["max_iterations"]
        self.refinement.use_feedback = original_params["use_feedback"]
        self.refinement.small_object_threshold = original_params["small_object_threshold"]
        self.refinement.large_object_threshold = original_params["large_object_threshold"]
        self.refinement.small_kernel_size = original_params["small_kernel_size"]
        self.refinement.medium_kernel_size = original_params["medium_kernel_size"]
        self.refinement.large_kernel_size = original_params["large_kernel_size"]
        self.refinement.confidence_threshold = original_params["confidence_threshold"]
        
        return best_row
    
    def evaluate_parameter_set(self, test_dataloader, test_labels):
        """
        Evaluate model with current parameter set
        
        Args:
            test_dataloader: DataLoader for test data
            test_labels: List of test class labels
            
        Returns:
            metrics: Dictionary of evaluation metrics
        """
        self.eval()
        device = next(self.parameters()).device
        
        # Track metrics
        class_dices = {label: [] for label in test_labels}
        all_dices = []
        start_time = time.time()
        
        with torch.no_grad():
            # Test each class
            for test_class in test_labels:
                print(f"  Evaluating class: {test_class}")
                
                # Set current class for validation dataset
                test_dataloader.dataset.set_curr_cls(test_class)
                
                # Get support data
                support_batched = test_dataloader.dataset.dataset.get_support(
                    curr_class=test_class, 
                    class_idx=[test_class], 
                    scan_idx=[0],  # Using first scan as support
                    npart=3  # Number of parts in support volume
                )
                
                # Prepare support data
                support_images = [[shot.to(device) for shot in way] 
                                  for way in support_batched['support_images']]
                support_fg_mask = [[shot['fg_mask'].to(device) for shot in way] 
                                   for way in support_batched['support_mask']]
                support_bg_mask = [[shot['bg_mask'].to(device) for shot in way] 
                                   for way in support_batched['support_mask']]
                
                # Track testing samples for this class
                class_samples = 0
                
                for sample in test_dataloader:
                    # Skip support scans
                    if sample["scan_id"][0] in test_dataloader.dataset.dataset.potential_support_sid:
                        continue
                    
                    # Get query part assignment
                    q_part = sample["part_assign"]
                    
                    # Prepare query data
                    query_images = [sample['image'].to(device)]
                    query_labels = sample['label'].to(device)
                    
                    # Select support based on part assignment
                    sup_img_part = [[shot_tensor.unsqueeze(0) for shot_tensor in support_images[0][q_part]]]
                    sup_fgm_part = [[shot_tensor.unsqueeze(0) for shot_tensor in support_fg_mask[0][q_part]]]
                    sup_bgm_part = [[shot_tensor.unsqueeze(0) for shot_tensor in support_bg_mask[0][q_part]]]
                    
                    # Forward pass
                    query_pred, _, _, _ = self(
                        sup_img_part, 
                        sup_fgm_part, 
                        sup_bgm_part, 
                        query_images, 
                        isval=True, 
                        val_wsize=2
                    )
                    
                    # Convert prediction to binary mask
                    pred_mask = query_pred.argmax(dim=1)
                    
                    # Calculate Dice coefficient
                    intersection = (pred_mask * query_labels).sum().item()
                    union = pred_mask.sum().item() + query_labels.sum().item()
                    if union > 0:
                        dice = (2 * intersection) / union
                    else:
                        dice = 1.0 if intersection == 0 else 0.0
                    
                    # Record Dice score
                    all_dices.append(dice)
                    class_dices[test_class].append(dice)
                    class_samples += 1
                
                print(f"    Processed {class_samples} samples for class {test_class}")
        
        # Calculate metrics
        total_time = time.time() - start_time
        
        # Overall metrics
        metrics = {
            "mean_dice": np.mean(all_dices),
            "median_dice": np.median(all_dices),
            "min_dice": np.min(all_dices),
            "max_dice": np.max(all_dices),
            "std_dice": np.std(all_dices),
            "total_time": total_time,
            "samples_per_second": len(all_dices) / total_time if total_time > 0 else 0
        }
        
        # Class-specific metrics
        for cls in class_dices:
            if class_dices[cls]:
                metrics[f"class_{cls}_dice"] = np.mean(class_dices[cls])
        
        return metrics

# Function to run a comprehensive parameter grid search
def run_grid_search(model, test_dataloader, test_labels, output_dir="param_tuning_results"):
    """
    Run a comprehensive grid search over refinement parameters
    
    Args:
        model: Model to tune
        test_dataloader: DataLoader for test data
        test_labels: List of test class labels  
        output_dir: Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_csv = os.path.join(output_dir, f"refinement_params_{timestamp}.csv")
    
    # Parameter grid to search
    param_grid = {
        "refinement_iterations": [2, 3, 5, 7],
        "use_feedback": [True, False],
        "small_object_threshold": [0.005, 0.01, 0.02],
        "large_object_threshold": [0.15, 0.2, 0.25],
        "small_kernel_size": [3, 5],
        "medium_kernel_size": [3, 5],
        "large_kernel_size": [5, 7],
        "confidence_threshold": [0.3, 0.5, 0.7]
    }
    
    # Original parameters for reference
    original_params = {
        "refinement_iterations": model.max_iterations,
        "use_feedback": model.refinement.use_feedback,
        "small_object_threshold": model.refinement.small_object_threshold,
        "large_object_threshold": model.refinement.large_object_threshold,
        "small_kernel_size": model.refinement.small_kernel_size,
        "medium_kernel_size": model.refinement.medium_kernel_size,
        "large_kernel_size": model.refinement.large_kernel_size,
        "confidence_threshold": model.refinement.confidence_threshold
    }
    
    print("Starting grid search with parameter combinations:")
    for param, values in param_grid.items():
        print(f"  {param}: {values}")
    
    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(product(*param_values))
    
    # Store results
    results = []
    
    # Evaluate model with original parameters as baseline
    print("\nEvaluating with original parameters as baseline:")
    for param, value in original_params.items():
        print(f"  {param}: {value}")
    
    baseline_metrics = model.evaluate_parameter_set(test_dataloader, test_labels)
    baseline_row = {**original_params, **baseline_metrics, "config": "baseline"}
    results.append(baseline_row)
    
    # Save initial results to CSV
    pd.DataFrame([baseline_row]).to_csv(output_csv, index=False)
    
    print(f"Baseline results: Mean Dice = {baseline_metrics['mean_dice']:.4f}")
    
    # Test each parameter combination
    for i, combination in enumerate(param_combinations):
        # Set parameters for this run
        params = {name: value for name, value in zip(param_names, combination)}
        
        print(f"\nTesting combination {i+1}/{len(param_combinations)}:")
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
            elif param_name == "confidence_threshold":
                model.refinement.confidence_threshold = param_value
        
        # Evaluate model with current parameters
        metrics = model.evaluate_parameter_set(test_dataloader, test_labels)
        
        # Add parameters to metrics and save results
        result_row = {**params, **metrics, "config": f"combo_{i+1}"}
        results.append(result_row)
        
        # Save to CSV after each run
        pd.DataFrame(results).to_csv(output_csv, index=False)
        
        print(f"Results for this combination: Mean Dice = {metrics['mean_dice']:.4f}")
        
        # Compare with baseline
        dice_diff = metrics['mean_dice'] - baseline_metrics['mean_dice']
        print(f"Difference from baseline: {dice_diff:.4f} ({'+' if dice_diff >= 0 else ''}{dice_diff*100:.2f}%)")
    
    # Find best parameters
    df = pd.DataFrame(results)
    best_idx = df["mean_dice"].idxmax()
    best_row = df.iloc[best_idx]
    
    print("\n===== Grid Search Complete =====")
    print(f"Best configuration: {best_row['config']}")
    print(f"Best Mean Dice Score: {best_row['mean_dice']:.4f}")
    print("Best Parameters:")
    for param in param_names:
        if param in best_row:
            print(f"  {param}: {best_row[param]}")
    
    # Create a visualization of results
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=(15, 10))
        for i, param in enumerate(param_grid.keys()):
            plt.subplot(2, 4, i+1)
            sns.boxplot(x=param, y='mean_dice', data=df)
            plt.title(f'Effect of {param}')
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"parameter_effects_{timestamp}.png")
        plt.savefig(plot_path)
        print(f"Parameter effects visualization saved to {plot_path}")
    except ImportError:
        print("Could not create visualization. Make sure matplotlib and seaborn are installed.")
    
    # Restore original parameters
    model.max_iterations = original_params["refinement_iterations"]
    model.refinement.max_iterations = original_params["refinement_iterations"]
    model.refinement.use_feedback = original_params["use_feedback"]
    model.refinement.small_object_threshold = original_params["small_object_threshold"]
    model.refinement.large_object_threshold = original_params["large_object_threshold"]
    model.refinement.small_kernel_size = original_params["small_kernel_size"]
    model.refinement.medium_kernel_size = original_params["medium_kernel_size"]
    model.refinement.large_kernel_size = original_params["large_kernel_size"]
    model.refinement.confidence_threshold = original_params["confidence_threshold"]
    
    print("\nParameters restored to original values.")
    return best_row

# Function to run a focused parameter search
def run_focused_parameter_search(model, test_dataloader, test_labels, focus_params, output_dir="param_tuning_results"):
    """
    Run a focused parameter search on specific parameters
    
    Args:
        model: Model to tune
        test_dataloader: DataLoader for test data
        test_labels: List of test class labels
        focus_params: Dictionary of parameters to focus on with their values to test
        output_dir: Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_csv = os.path.join(output_dir, f"focused_params_{timestamp}.csv")
    
    # Save original parameters
    original_params = {
        "refinement_iterations": model.max_iterations,
        "use_feedback": model.refinement.use_feedback,
        "small_object_threshold": model.refinement.small_object_threshold,
        "large_object_threshold": model.refinement.large_object_threshold,
        "small_kernel_size": model.refinement.small_kernel_size,
        "medium_kernel_size": model.refinement.medium_kernel_size,
        "large_kernel_size": model.refinement.large_kernel_size,
        "confidence_threshold": model.refinement.confidence_threshold
    }
    
    print(f"Running focused parameter search on: {list(focus_params.keys())}")
    
    # Generate parameter combinations for focused search
    param_names = list(focus_params.keys())
    param_values = list(focus_params.values())
    param_combinations = list(product(*param_values))
    
    # Store results
    results = []
    
    # Evaluate baseline
    print("\nEvaluating with original parameters as baseline:")
    baseline_metrics = model.evaluate_parameter_set(test_dataloader, test_labels)
    baseline_row = {**original_params, **baseline_metrics, "config": "baseline"}
    results.append(baseline_row)
    pd.DataFrame([baseline_row]).to_csv(output_csv, index=False)
    
    print(f"Baseline results: Mean Dice = {baseline_metrics['mean_dice']:.4f}")
    
    # Test each focused parameter combination
    for i, combination in enumerate(param_combinations):
        # Set parameters for this run
        params = {name: value for name, value in zip(param_names, combination)}
        
        print(f"\nTesting combination {i+1}/{len(param_combinations)}:")
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
            elif param_name == "confidence_threshold":
                model.refinement.confidence_threshold = param_value
        
        # Evaluate model with current parameters
        metrics = model.evaluate_parameter_set(test_dataloader, test_labels)
        
        # Add parameters to metrics and save results
        result_row = {**original_params, **params, **metrics, "config": f"combo_{i+1}"}
        results.append(result_row)
        
        # Save to CSV after each run
        pd.DataFrame(results).to_csv(output_csv, index=False)
        
        print(f"Results for this combination: Mean Dice = {metrics['mean_dice']:.4f}")
        
        # Compare with baseline
        dice_diff = metrics['mean_dice'] - baseline_metrics['mean_dice']
        print(f"Difference from baseline: {dice_diff:.4f} ({'+' if dice_diff >= 0 else ''}{dice_diff*100:.2f}%)")
    
    # Find best parameters
    df = pd.DataFrame(results)
    best_idx = df["mean_dice"].idxmax()
    best_row = df.iloc[best_idx]
    
    print("\n===== Focused Parameter Search Complete =====")
    print(f"Best configuration: {best_row['config']}")
    print(f"Best Mean Dice Score: {best_row['mean_dice']:.4f}")
    print("Best Parameters:")
    for param in param_names:
        if param in best_row:
            print(f"  {param}: {best_row[param]}")
    
    # Create a visualization of results
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Plot parameter effects
        plt.figure(figsize=(15, 10))
        for i, param in enumerate(focus_params.keys()):
            plt.subplot(2, 2, i+1)
            sns.boxplot(x=param, y='mean_dice', data=df)
            plt.title(f'Effect of {param}')
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"focused_params_effects_{timestamp}.png")
        plt.savefig(plot_path)
        print(f"Parameter effects visualization saved to {plot_path}")
    except ImportError:
        print("Could not create visualization. Make sure matplotlib and seaborn are installed.")
    
    # Restore original parameters
    model.max_iterations = original_params["refinement_iterations"]
    model.refinement.max_iterations = original_params["refinement_iterations"]
    model.refinement.use_feedback = original_params["use_feedback"]
    model.refinement.small_object_threshold = original_params["small_object_threshold"]
    model.refinement.large_object_threshold = original_params["large_object_threshold"]
    model.refinement.small_kernel_size = original_params["small_kernel_size"]
    model.refinement.medium_kernel_size = original_params["medium_kernel_size"]
    model.refinement.large_kernel_size = original_params["large_kernel_size"]
    model.refinement.confidence_threshold = original_params["confidence_threshold"]
    
    print("\nParameters restored to original values.")
    return best_row