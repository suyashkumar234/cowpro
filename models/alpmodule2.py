"""
ALPModule with Self-Attention refinement
"""
import torch
import math
from torch import nn
from torch.nn import functional as F
import numpy as np

class MultiProtoAsWCos(nn.Module):
    def __init__(self, proto_grid, feature_hw, upsample_mode='bilinear'):
        """
        ALPModule
        Args:
            proto_grid:     Grid size when doing multi-prototyping. For a 32-by-32 feature map, a size of 16-by-16 leads to a pooling window of 2-by-2
            feature_hw:     Spatial size of input feature map
        """
        super(MultiProtoAsWCos, self).__init__()
        self.proto_grid = proto_grid
        self.upsample_mode = upsample_mode
        kernel_size = [ft_l // grid_l for ft_l, grid_l in zip(feature_hw, proto_grid)]
        self.avg_pool_op = nn.AvgPool2d(kernel_size)
        
        # Boundary enhancement with appropriate size padding
        self.boundary_detector = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, qry, sup_x, sup_y, mode, thresh, fg=True, isval=False, val_wsize=None, vis_sim=False, **kwargs):
        """
        Now supports
        Args:
            mode: 'mask'/ 'grid'. if mask, works as original prototyping
            qry: [way(1), nc, h, w]
            sup_x: [nb, nc, h, w]
            sup_y: [nb, 1, h, w]
            vis_sim: visualize raw similarities or not
        New
            mode:       'mask'/ 'grid'. if mask, works as original prototyping
            qry:        [way(1), nb(1), nc, h, w]
            sup_x:      [way(1), shot, nb(1), nc, h, w]
            sup_y:      [way(1), shot, nb(1), h, w]
            vis_sim:    visualize raw similarities or not
        """

        qry = qry.squeeze(1)  # [way(1), nb(1), nc, hw] -> [way(1), nc, h, w]
        sup_x = sup_x.squeeze(0).squeeze(1)  # [nshot, nc, h, w]
        sup_y = sup_y.squeeze(0)  # [nshot, 1, h, w]

        def safe_norm(x, p=2, dim=1, eps=1e-4):
            x_norm = torch.norm(x, p=p, dim=dim)
            x_norm = torch.max(x_norm, torch.ones_like(x_norm).cuda() * eps)
            x = x.div(x_norm.unsqueeze(1).expand_as(x))
            return x

        if mode == 'mask':  # class-level prototype only
            proto = torch.sum(sup_x * sup_y, dim=(-1, -2)) \
                / (sup_y.sum(dim=(-1, -2)) + 1e-5)  # nb x C

            proto = proto.mean(dim=0, keepdim=True)  # 1 X C, take the mean of everything
            pred_mask = F.cosine_similarity(qry, proto[..., None, None], dim=1, eps=1e-4) * 20.0  # [1, h, w]

            vis_dict = {'proto_assign': None}  # things to visualize
            if vis_sim:
                vis_dict['raw_local_sims'] = pred_mask
            return pred_mask.unsqueeze(1), [pred_mask], vis_dict  # just a placeholder. pred_mask returned as [1, way(1), h, w]

        # no need to merge with gridconv+
        elif mode == 'gridconv':  # using local prototypes only
            input_size = qry.shape
            nch = input_size[1]

            sup_nshot = sup_x.shape[0]

            # Apply pooling to get downsampled feature and mask 
            n_sup_x = F.avg_pool2d(sup_x, val_wsize) if isval else self.avg_pool_op(sup_x)
            sup_y_g = F.avg_pool2d(sup_y, val_wsize) if isval else self.avg_pool_op(sup_y)

            # Reshape for processing
            n_sup_x = n_sup_x.view(sup_nshot, nch, -1).permute(0, 2, 1).unsqueeze(0)  # way(1),nb, hw, nc
            n_sup_x = n_sup_x.reshape(1, -1, nch).unsqueeze(0)

            # Reshape masks
            sup_y_g = sup_y_g.view(sup_nshot, 1, -1).permute(1, 0, 2).view(1, -1).unsqueeze(0)

            # Select prototypes where mask value > threshold
            mask_threshold = thresh if fg else 0.5 * thresh  # Less strict threshold for background
            protos = n_sup_x[sup_y_g > mask_threshold, :]  # npro, nc
            
            # If no prototypes found, use a fallback approach
            if protos.shape[0] == 0:
                protos = n_sup_x[sup_y_g > 0.1, :]  # Use a much lower threshold
                if protos.shape[0] == 0:  # If still no prototypes, use all features
                    protos = n_sup_x.reshape(-1, nch)
            
            # Center the prototypes and query features
            protos = protos - protos.mean(dim=-1, keepdim=True)
            qry = qry - qry.mean(dim=1, keepdim=True)

            # Normalize features for cosine similarity
            pro_n = safe_norm(protos)  # npro x nc
            npro, nc = pro_n.shape
            qry_n = safe_norm(qry)  # 1 x nc x h x w
            _, nc, h, w = qry_n.shape

            # Compute similarity between query and prototypes 
            dists = F.conv2d(qry_n, pro_n[..., None, None]) * 20  # 1 x npro x h x w

            # Generate probability scores for each prototype
            qry_proto_prob = F.softmax(dists, dim=1).view(-1, npro, h*w)  # 1 x npro x h x w

            # Weighted prototype aggregation using attention-based weights
            ch_proto = torch.bmm(pro_n.unsqueeze(0).transpose(1, 2), qry_proto_prob).view(-1, nc, h, w)

            # Final prediction using weighted prototype similarity
            pred_grid = torch.sum(qry_n * ch_proto * 20, dim=1, keepdim=True)
            debug_assign = dists.argmax(dim=1).float().detach()

            vis_dict = {'proto_assign': debug_assign}  # things to visualize

            if vis_sim:  # return the similarity for visualization
                vis_dict['raw_local_sims'] = dists.clone().detach()

            return pred_grid, [debug_assign], vis_dict

        elif mode == 'gridconv+':  # local and global prototypes
            input_size = qry.shape
            nch = input_size[1]
            nb_q = input_size[0]

            # Get the number of support shots
            sup_nshot = sup_x.shape[0]  # FIX: Define sup_nshot for this mode

            # Apply pooling operations
            n_sup_x = F.avg_pool2d(sup_x, val_wsize) if isval else self.avg_pool_op(sup_x)
            sup_y_g = F.avg_pool2d(sup_y, val_wsize) if isval else self.avg_pool_op(sup_y)

            # Reshape for processing
            n_sup_x = n_sup_x.view(sup_nshot, nch, -1).permute(0, 2, 1).unsqueeze(0)
            n_sup_x = n_sup_x.reshape(1, -1, nch).unsqueeze(0)

            sup_y_g = sup_y_g.view(sup_nshot, 1, -1).permute(1, 0, 2).view(1, -1).unsqueeze(0)

            # Select local prototypes
            protos = n_sup_x[sup_y_g > thresh, :]
            
            # Compute global prototype
            glb_proto = torch.sum(sup_x * sup_y, dim=(-1, -2)) / (sup_y.sum(dim=(-1, -2)) + 1e-5)

            # Combine local and global prototypes
            protos = torch.cat([protos, glb_proto], dim=0)

            # Center prototypes and query features
            protos = protos - protos.mean(dim=-1, keepdim=True)
            qry = qry - qry.mean(dim=1, keepdim=True)

            # Normalize for cosine similarity
            pro_n = safe_norm(protos)  # npro x nc
            npro, nc = pro_n.shape
            qry_n = safe_norm(qry)  # 1 x nc x h x w
            _, nc, h, w = qry_n.shape

            # Compute similarity
            dists = F.conv2d(qry_n, pro_n[..., None, None]) * 20  # 1 x npro x h x w

            # Compute weighted prototype aggregation
            qry_proto_prob = F.softmax(dists, dim=1).view(-1, npro, h*w)  # 1 x npro x h x w
            ch_proto = torch.bmm(pro_n.unsqueeze(0).transpose(1, 2), qry_proto_prob).view(-1, nc, h, w)

            # Final prediction
            pred_grid = torch.sum(qry_n * ch_proto * 20, dim=1, keepdim=True)
            debug_assign = dists.argmax(dim=1).float()

            vis_dict = {'proto_assign': debug_assign}
            if vis_sim:
                vis_dict['raw_local_sims'] = dists.clone().detach()

            return pred_grid, [debug_assign], vis_dict

        else:
            # Other modes are unchanged
            B, C, H, W = qry.shape
            nch = C
            nb_q = B

            sup_nshot = sup_x.shape[0]  # FIX: Define sup_nshot for this mode
            n_sup_x = F.avg_pool2d(sup_x, val_wsize) if isval else self.avg_pool_op(sup_x)

            n_sup_x = n_sup_x.view(sup_nshot, nch, -1).permute(0, 2, 1).unsqueeze(0)
            n_sup_x = n_sup_x.reshape(1, -1, nch).unsqueeze(0)

            sup_y_g = F.avg_pool2d(sup_y, val_wsize) if isval else self.avg_pool_op(sup_y)
            sup_y_g = sup_y_g.view(sup_nshot, 1, -1).permute(1, 0, 2).view(1, -1).unsqueeze(0)

            protos = n_sup_x[sup_y_g > thresh, :]  # n_pro x n_c
            protos = protos.unsqueeze(0)
            
            if fg:
                glb_proto = torch.sum(sup_x * sup_y, dim=(-1, -2)) / (sup_y.sum(dim=(-1, -2)) + 1e-5)
                if F.avg_pool2d(sup_y, 4).max() >= 0.95:
                    protos = torch.cat([protos, glb_proto.unsqueeze(1)], dim=0)
                else:
                    protos = glb_proto.unsqueeze(1)

            qry_n = qry
            qry_n = qry_n.view(B, C, -1).transpose(1, 2)

            self.temperature = C**0.5

            # Compute attention-based similarity between prototypes and query features
            attn1 = torch.bmm(protos/self.temperature, qry_n.transpose(1, 2))
            attn1 = F.softmax(attn1/0.05, dim=-1)

            # Self-attention to refine the features
            attn = torch.bmm(attn1.transpose(1, 2), attn1)

            # Use attention to combine features
            pred_grid = torch.bmm(attn, qry_n) * 20
            
            # Reshape back to spatial dimensions
            pred_grid = pred_grid.transpose(1, 2).view(B, C, H, W).mean(dim=1, keepdim=True)

            raw_local_sims = attn.detach()
            debug_assign = attn.argmax(dim=1).float().view(B, H, W)

            vis_dict = {'proto_assign': debug_assign}
            if vis_sim:
                vis_dict['raw_local_sims'] = raw_local_sims

            return pred_grid, [debug_assign], vis_dict
