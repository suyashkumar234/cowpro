"""
ALPModule with UNet integration - Fixed Version
"""
import torch
import math
from torch import nn
from torch.nn import functional as F
import numpy as np
from pdb import set_trace
import matplotlib.pyplot as plt

class MultiProtoAsWCos(nn.Module):
    def __init__(self, proto_grid, feature_hw, upsample_mode = 'bilinear'):
        """
        ALPModule with enhanced correlation weighting for UNet features
        Args:
            proto_grid:     Grid size when doing multi-prototyping
            feature_hw:     Spatial size of input feature map
        """
        super(MultiProtoAsWCos, self).__init__()
        self.proto_grid = proto_grid
        self.upsample_mode = upsample_mode
        kernel_size = [ft_l // grid_l for ft_l, grid_l in zip(feature_hw, proto_grid)]
        self.avg_pool_op = nn.AvgPool2d(kernel_size)
        
        # Enhanced correlation weighting modules for UNet features
        self.correlation_conv = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128)
        )
        
        # Attention mechanism for prototype weighting
        self.attention_conv = nn.Sequential(
            nn.Conv2d(128, 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )

    def extract_prototypes_with_correlation(self, sup_x, sup_y, qry_x, thresh, fg=True):
        """
        Extract prototypes using correlation-based weighting
        """
        sup_nshot = sup_x.shape[0]
        nch = sup_x.shape[1]
        
        # Apply correlation weighting
        weighted_sup_x = self.correlation_conv(sup_x)
        weighted_qry_x = F.adaptive_avg_pool2d(self.correlation_conv(qry_x), sup_x.shape[-2:])
        
        # Compute attention weights
        attention_weights = self.attention_conv(weighted_sup_x)
        
        # Apply attention to support features
        attended_sup_x = sup_x * attention_weights
        
        # Pool features for prototype extraction
        n_sup_x = self.avg_pool_op(attended_sup_x)
        n_sup_x = n_sup_x.view(sup_nshot, nch, -1).permute(0,2,1).unsqueeze(0)
        n_sup_x = n_sup_x.reshape(1, -1, nch).unsqueeze(0)
        
        # Pool masks
        sup_y_g = self.avg_pool_op(sup_y)
        sup_y_g = sup_y_g.view(sup_nshot, 1, -1).permute(1, 0, 2).view(1, -1).unsqueeze(0)
        
        # Extract prototypes above threshold
        valid_mask = sup_y_g > thresh
        protos = n_sup_x[valid_mask, :]
        
        # Handle empty prototypes case
        if len(protos) == 0:
            # Create a fallback prototype using global average
            if fg:
                # For foreground, use global prototype
                glb_proto = torch.sum(attended_sup_x * sup_y, dim=(-1, -2)) / (sup_y.sum(dim=(-1, -2)) + 1e-5)
                protos = glb_proto.mean(dim=0, keepdim=True)  # Average across shots
            else:
                # For background, use inverse mask
                bg_mask = 1.0 - sup_y
                glb_proto = torch.sum(attended_sup_x * bg_mask, dim=(-1, -2)) / (bg_mask.sum(dim=(-1, -2)) + 1e-5)
                protos = glb_proto.mean(dim=0, keepdim=True)  # Average across shots
        elif fg and len(protos) > 0:
            # Add global prototype for foreground
            glb_proto = torch.sum(attended_sup_x * sup_y, dim=(-1, -2)) / (sup_y.sum(dim=(-1, -2)) + 1e-5)
            if F.avg_pool2d(sup_y, 4).max() >= 0.95:
                protos = torch.cat([protos, glb_proto], dim=0)
            else:
                protos = glb_proto.unsqueeze(0)
        
        return protos, attention_weights

    def correlate_query_with_prototypes(self, qry, protos):
        """
        Perform correlation-based weighting between query and prototypes
        """
        def safe_norm(x, p=2, dim=1, eps=1e-4):
            x_norm = torch.norm(x, p=p, dim=dim)
            x_norm = torch.max(x_norm, torch.ones_like(x_norm).cuda() * eps)
            x = x.div(x_norm.unsqueeze(1).expand_as(x))
            return x
        
        # Normalize features
        protos = protos - protos.mean(dim=-1, keepdim=True)
        qry = qry - qry.mean(dim=1, keepdim=True)
        
        pro_n = safe_norm(protos)
        qry_n = safe_norm(qry)
        
        npro, nc = pro_n.shape
        _, nc, h, w = qry_n.shape
        
        # Correlation-based similarity computation
        dists = F.conv2d(qry_n, pro_n[..., None, None]) * 20
        
        # Enhanced correlation weighting
        qry_proto_prob = F.softmax(dists, dim=1).view(-1, npro, h*w)
        
        # Weighted feature combination
        ch_proto = torch.bmm(pro_n.unsqueeze(0).transpose(1,2), qry_proto_prob).view(-1, nc, h, w)
        
        # Final correlation score
        pred_grid = torch.sum(qry_n * ch_proto * 20, dim=1, keepdim=True)
        
        return pred_grid, dists

    def forward(self, qry, sup_x, sup_y, mode, thresh, fg=True, isval=False, val_wsize=None, vis_sim=False, **kwargs):
        """
        Enhanced forward pass with UNet feature correlation
        """
        qry = qry.squeeze(1)  # [way(1), nb(1), nc, hw] -> [way(1), nc, h, w]
        sup_x = sup_x.squeeze(0).squeeze(1)  # [nshot, nc, h, w]
        sup_y = sup_y.squeeze(0)  # [nshot, 1, h, w]

        if mode == 'mask':  # class-level prototype only
            proto = torch.sum(sup_x * sup_y, dim=(-1, -2)) / (sup_y.sum(dim=(-1, -2)) + 1e-5)
            proto = proto.mean(dim=0, keepdim=True)
            pred_mask = F.cosine_similarity(qry, proto[..., None, None], dim=1, eps=1e-4) * 20.0
            
            vis_dict = {'proto_assign': None}
            if vis_sim:
                vis_dict['raw_local_sims'] = pred_mask
            return pred_mask.unsqueeze(1), [pred_mask], vis_dict

        elif mode == 'gridconv':  # using local prototypes only
            protos, attention_weights = self.extract_prototypes_with_correlation(
                sup_x, sup_y, qry, thresh, fg=False)
            
            pred_grid, dists = self.correlate_query_with_prototypes(qry, protos)
            debug_assign = dists.argmax(dim=1).float().detach()
            
            vis_dict = {'proto_assign': debug_assign, 'attention_weights': attention_weights}
            if vis_sim:
                vis_dict['raw_local_sims'] = dists.clone().detach()
            
            return pred_grid, [debug_assign], vis_dict

        elif mode == 'gridconv+':  # local and global prototypes
            protos, attention_weights = self.extract_prototypes_with_correlation(
                sup_x, sup_y, qry, thresh, fg=True)
            
            pred_grid, dists = self.correlate_query_with_prototypes(qry, protos)
            debug_assign = dists.argmax(dim=1).float()
            
            vis_dict = {'proto_assign': debug_assign, 'attention_weights': attention_weights}
            if vis_sim:
                vis_dict['raw_local_sims'] = dists.clone().detach()
            
            return pred_grid, [debug_assign], vis_dict

        else:
            # Enhanced attention-based correlation for other modes
            return self.enhanced_attention_forward(qry, sup_x, sup_y, thresh, fg, isval, val_wsize, vis_sim)

    def original_gridconv_forward(self, qry, sup_x, sup_y, thresh, isval, val_wsize, vis_sim):
        """Fallback to original gridconv implementation with empty prototype handling"""
        def safe_norm(x, p=2, dim=1, eps=1e-4):
            x_norm = torch.norm(x, p=p, dim=dim)
            x_norm = torch.max(x_norm, torch.ones_like(x_norm).cuda() * eps)
            x = x.div(x_norm.unsqueeze(1).expand_as(x))
            return x

        input_size = qry.shape
        nch = input_size[1]
        sup_nshot = sup_x.shape[0]

        n_sup_x = F.avg_pool2d(sup_x, val_wsize) if isval else self.avg_pool_op(sup_x)
        n_sup_x = n_sup_x.view(sup_nshot, nch, -1).permute(0,2,1).unsqueeze(0)
        n_sup_x = n_sup_x.reshape(1, -1, nch).unsqueeze(0)

        sup_y_g = F.avg_pool2d(sup_y, val_wsize) if isval else self.avg_pool_op(sup_y)
        sup_y_g = sup_y_g.view(sup_nshot, 1, -1).permute(1, 0, 2).view(1, -1).unsqueeze(0)

        valid_mask = sup_y_g > thresh
        protos = n_sup_x[valid_mask, :]
        
        # Handle empty prototypes case
        if len(protos) == 0:
            # Create fallback prototype using global average
            proto = torch.sum(sup_x * sup_y, dim=(-1, -2)) / (sup_y.sum(dim=(-1, -2)) + 1e-5)
            protos = proto.mean(dim=0, keepdim=True)  # [1, nch]
        
        protos = protos - protos.mean(dim=-1, keepdim=True)
        qry = qry - qry.mean(dim=1, keepdim=True)

        pro_n = safe_norm(protos)
        qry_n = safe_norm(qry)
        
        dists = F.conv2d(qry_n, pro_n[..., None, None]) * 20
        qry_proto_prob = F.softmax(dists, dim=1).view(-1, pro_n.shape[0], qry.shape[-2]*qry.shape[-1])
        ch_proto = torch.bmm(pro_n.unsqueeze(0).transpose(1,2), qry_proto_prob).view(-1, nch, qry.shape[-2], qry.shape[-1])
        pred_grid = torch.sum(qry_n * ch_proto * 20, dim=1, keepdim=True)
        debug_assign = dists.argmax(dim=1).float().detach()

        vis_dict = {'proto_assign': debug_assign}
        if vis_sim:
            vis_dict['raw_local_sims'] = dists.clone().detach()

        return pred_grid, [debug_assign], vis_dict

    def original_gridconv_plus_forward(self, qry, sup_x, sup_y, thresh, isval, val_wsize, vis_sim):
        """Fallback to original gridconv+ implementation with empty prototype handling"""
        def safe_norm(x, p=2, dim=1, eps=1e-4):
            x_norm = torch.norm(x, p=p, dim=dim)
            x_norm = torch.max(x_norm, torch.ones_like(x_norm).cuda() * eps)
            x = x.div(x_norm.unsqueeze(1).expand_as(x))
            return x

        input_size = qry.shape
        nch = input_size[1]
        sup_nshot = sup_x.shape[0]

        n_sup_x = F.avg_pool2d(sup_x, val_wsize) if isval else self.avg_pool_op(sup_x)
        n_sup_x = n_sup_x.view(sup_nshot, nch, -1).permute(0,2,1).unsqueeze(0)
        n_sup_x = n_sup_x.reshape(1, -1, nch).unsqueeze(0)

        sup_y_g = F.avg_pool2d(sup_y, val_wsize) if isval else self.avg_pool_op(sup_y)
        sup_y_g = sup_y_g.view(sup_nshot, 1, -1).permute(1, 0, 2).view(1, -1).unsqueeze(0)

        valid_mask = sup_y_g > thresh
        protos = n_sup_x[valid_mask, :]
        
        # Always add global prototype
        glb_proto = torch.sum(sup_x * sup_y, dim=(-1, -2)) / (sup_y.sum(dim=(-1, -2)) + 1e-5)
        
        if len(protos) == 0:
            # Use only global prototype if no local prototypes found
            protos = glb_proto.mean(dim=0, keepdim=True)  # [1, nch]
        else:
            # Concatenate local and global prototypes
            protos = torch.cat([protos, glb_proto], dim=0)

        protos = protos - protos.mean(dim=-1, keepdim=True)
        qry = qry - qry.mean(dim=1, keepdim=True)

        pro_n = safe_norm(protos)
        qry_n = safe_norm(qry)
        
        dists = F.conv2d(qry_n, pro_n[..., None, None]) * 20
        qry_proto_prob = F.softmax(dists, dim=1).view(-1, pro_n.shape[0], qry.shape[-2]*qry.shape[-1])
        ch_proto = torch.bmm(pro_n.unsqueeze(0).transpose(1,2), qry_proto_prob).view(-1, nch, qry.shape[-2], qry.shape[-1])
        pred_grid = torch.sum(qry_n * ch_proto * 20, dim=1, keepdim=True)
        debug_assign = dists.argmax(dim=1).float()

        vis_dict = {'proto_assign': debug_assign}
        if vis_sim:
            vis_dict['raw_local_sims'] = dists.clone().detach()

        return pred_grid, [debug_assign], vis_dict

    def enhanced_attention_forward(self, qry, sup_x, sup_y, thresh, fg, isval, val_wsize, vis_sim):
        """Enhanced attention-based forward pass"""
        B, C, H, W = qry.shape
        nch = C
        sup_nshot = sup_x.shape[0]

        # Extract prototypes with correlation weighting
        protos, attention_weights = self.extract_prototypes_with_correlation(
            sup_x, sup_y, qry, thresh, fg=fg)
        
        # Enhanced correlation computation
        qry_n = qry.view(B, C, -1).transpose(1, 2)
        self.temperature = C**0.5

        # Multi-head attention mechanism
        attn1 = torch.bmm(protos/self.temperature, qry_n.transpose(1, 2))
        attn1 = F.softmax(attn1/0.05, dim=-1)
        
        # Self-attention for better correlation
        attn = torch.bmm(attn1.transpose(1,2), attn1)
        
        # Final prediction with correlation weighting
        pred_grid = torch.bmm(attn, qry_n) * 20
        pred_grid = pred_grid.mean(dim=1).view(B, 1, H, W)

        debug_assign = attn.argmax(dim=1).float().view(B, H, W)

        vis_dict = {'proto_assign': debug_assign, 'attention_weights': attention_weights}
        if vis_sim:
            vis_dict['raw_local_sims'] = attn.clone().detach()

        return pred_grid, [debug_assign], vis_dict
