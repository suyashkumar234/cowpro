"""
ALPNet with Self-Attention - Fixed Version
"""
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

from .alpmodulesa import MultiProtoAsWCos  # Note the modified import
from .backbone.torchvision_backbones import TVDeeplabRes101Encoder, Encoder
from util.utils import get_tversky_loss
import numpy as np

# options for type of prototypes
FG_PROT_MODE = 'gridconv+' # using both local and global prototype
BG_PROT_MODE = 'gridconv' #gridconv  # using local prototype only. 
# Also 'mask' refers to using global prototype only (as done in vanilla PANet)

# thresholds for deciding class of prototypes
FG_THRESH = 0.95
BG_THRESH = 0.95

class SelfAttention(nn.Module):
    def __init__(self, in_dim, attention_dim=None):
        super(SelfAttention, self).__init__()
        if attention_dim is None:
            attention_dim = in_dim // 8
        
        # Add layer normalization
        self.layer_norm = nn.LayerNorm([in_dim])
        
        self.query_conv = nn.Conv2d(in_dim, attention_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_dim, attention_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        
        self.gamma = nn.Parameter(torch.ones(1) * 0.01)
        self.forward_count = 0
        
    def forward(self, x):
        self.forward_count += 1
        batch_size, C, height, width = x.size()
        
        # Store input features for tracking changes
        input_features = x.clone()
        
        # Compute self-attention
        proj_query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, width * height)
        
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        
        self.height = height
        self.width = width
        
        proj_value = self.value_conv(x).view(batch_size, -1, width * height)
        
        attention_output = torch.bmm(proj_value, attention.permute(0, 2, 1))
        attention_output = attention_output.view(batch_size, C, height, width)
        
        self_attention_contribution = attention_output
        weighted_attention = self.gamma * attention_output
        feature_delta = weighted_attention
        
        # Add residual connection
        pre_norm_output = weighted_attention + x
        
        # Apply layer normalization after residual (need to reshape)
        output_reshaped = pre_norm_output.permute(0, 2, 3, 1)  # [B, H, W, C]
        normalized_output = self.layer_norm(output_reshaped)
        out = normalized_output.permute(0, 3, 1, 2)  # [B, C, H, W]
        
        if attention.numel() == 1:
            attention = attention.repeat(1, 4, 4)
            
        if self.forward_count % 100 == 0:
            print(f"Gamma value: {self.gamma.item():.6f}")
            print(f"Input features mean: {input_features.mean().item():.6f}, std: {input_features.std().item():.6f}")
            print(f"Attention contribution mean: {weighted_attention.mean().item():.6f}, std: {weighted_attention.std().item():.6f}")
            print(f"Pre-norm output mean: {pre_norm_output.mean().item():.6f}, std: {pre_norm_output.std().item():.6f}")
            print(f"Normalized output mean: {out.mean().item():.6f}, std: {out.std().item():.6f}")
            
        return out, attention, feature_delta, self_attention_contribution

class FewShotSeg(nn.Module):
    """
    ALPNet
    Args:
        in_channels:        Number of input channels
        cfg:                Model configurations
    """
    def __init__(self, in_channels=3, pretrained_path=None, cfg=None):
        super(FewShotSeg, self).__init__()
        self.pretrained_path = pretrained_path
        self.config = cfg or {'align': False}
        self.get_encoder(in_channels)
        self.get_cls()
        
        # Add self-attention module for feature refinement
        self.self_attention = SelfAttention(256)  # 256 is the feature dimension from the encoder
        
    def get_encoder(self, in_channels):
        use_coco_init = self.config['use_coco_init']
        self.encoder = TVDeeplabRes101Encoder(use_coco_init)

        if self.pretrained_path:
            self.load_state_dict(torch.load(self.pretrained_path)['model'], strict = False)
            print(f'###### Pre-trained model {self.pretrained_path} has been loaded ######')

    def get_cls(self):
        """
        Obtain the similarity-based classifier
        """
        proto_hw = self.config["proto_grid_size"]
        feature_hw = self.config["feature_hw"]
        assert self.config['cls_name'] == 'grid_proto'
        if self.config['cls_name'] == 'grid_proto':
            self.cls_unit = MultiProtoAsWCos(proto_grid = [proto_hw, proto_hw], 
                                            feature_hw =  self.config["feature_hw"]) # when treating it as ordinary prototype
        else:
            raise NotImplementedError(f'Classifier {self.config["cls_name"]} not implemented')

    def forward(self, supp_imgs, fore_mask, back_mask, qry_imgs, isval, val_wsize, show_viz = False):
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
            show_viz: return the visualization dictionary
        """
        n_ways = len(supp_imgs)
        n_shots = len(supp_imgs[0])
        n_queries = len(qry_imgs)

        assert n_ways == 1, "Multi-shot has not been implemented yet" 
        # NOTE: actual shot in support goes in batch dimension
        assert n_queries == 1

        sup_bsize = supp_imgs[0][0].shape[0]
        img_size = supp_imgs[0][0].shape[-2:]
        qry_bsize = qry_imgs[0].shape[0]

        assert sup_bsize == qry_bsize == 1

        imgs_concat = torch.cat([torch.cat(way, dim=0) for way in supp_imgs]
                                + [torch.cat(qry_imgs, dim=0),], dim=0)

        img_fts = self.encoder(imgs_concat, low_level = False)
        
        # Apply self-attention to refine the features with error handling
        try:
    # Store features before self-attention for visualization
            features_before_attention = img_fts.clone()
    
    # Enhanced self-attention returns more info for analysis
            img_fts, attention_maps, feature_delta, raw_attention_contribution = self.self_attention(img_fts)
    
    # Save these for visualization later
            self.features_before_attention = features_before_attention
            self.features_after_attention = img_fts
            self.feature_delta = feature_delta
            self.raw_attention_contribution = raw_attention_contribution
    
        except Exception as e:
            print(f"Error in self-attention module: {e}")
    # Fallback: use original features and create dummy tensors
            self.features_before_attention = img_fts.clone()
            self.features_after_attention = img_fts
            self.feature_delta = torch.zeros_like(img_fts)
            self.raw_attention_contribution = torch.zeros_like(img_fts)
            attention_maps = torch.ones((1, 1, 1), device=img_fts.device)
        
        fts_size = img_fts.shape[-2:]

        supp_fts = img_fts[:n_ways * n_shots * sup_bsize].view(
            n_ways, n_shots, sup_bsize, -1, *fts_size)  # Wa x Sh x B x C x H' x W'
        qry_fts = img_fts[n_ways * n_shots * sup_bsize:].view(
            n_queries, qry_bsize, -1, *fts_size)   # N x B x C x H' x W'
        fore_mask = torch.stack([torch.stack(way, dim=0)
                                 for way in fore_mask], dim=0)  # Wa x Sh x B x H' x W'
        fore_mask = torch.autograd.Variable(fore_mask, requires_grad = True)
        back_mask = torch.stack([torch.stack(way, dim=0)
                                 for way in back_mask], dim=0)  # Wa x Sh x B x H' x W'

        # Re-interpolate support mask to the same size as support feature
        res_fg_msk = torch.stack([F.interpolate(fore_mask_w, size=fts_size, mode='bilinear') 
                                 for fore_mask_w in fore_mask], dim=0)  # [nway, ns, nb, nh', nw']
        res_bg_msk = torch.stack([F.interpolate(back_mask_w, size=fts_size, mode='bilinear') 
                                 for back_mask_w in back_mask], dim=0)  # [nway, ns, nb, nh', nw']

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

        pred = torch.cat(scores, dim=1)  # N x (1 + Wa) x H' x W'
        outputs = [F.interpolate(pred, size=img_size, mode='bilinear')]

        ###### Prototype alignment loss ######
        align_loss = 0
        if self.config['align'] and self.training:
            for epi in range(1):  # batch dimension, fixed to 1
                try:
                    align_loss_epi = self.alignLoss(qry_fts[:, epi], pred, supp_fts[:, :, epi],
                                                   fore_mask[:, :, epi], back_mask[:, :, epi])
                    align_loss += align_loss_epi
                except Exception as e:
                    print(f"Error in alignment loss: {e}")
                    align_loss += 0
                    
        output = torch.stack(outputs, dim=1)  # N x B x (1 + Wa) x H x W
        output = output.view(-1, *output.shape[2:])
        assign_maps = torch.stack(assign_maps, dim=1)
        bg_sim_maps = torch.stack(bg_sim_maps, dim=1) if show_viz else None
        fg_sim_maps = torch.stack(fg_sim_maps, dim=1) if show_viz else None

        return output, align_loss / sup_bsize, [bg_sim_maps, fg_sim_maps], assign_maps, attention_maps


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

        # FIXME: fix this in future we here make a stronger assumption that a positive class must be there to avoid undersegmentation/ lazyness
        skip_ways = []

        ### added for matching dimensions to the new data format
        qry_fts = qry_fts.unsqueeze(0).unsqueeze(2) # added to nway(1) and nb(1)

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
