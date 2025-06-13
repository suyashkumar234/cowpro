"""
ALPNet with UNet Encoder - Fixed Version
"""
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

from .alpmodule import MultiProtoAsConv
from .alpmodule2unet2 import MultiProtoAsWCos
from .backbone.unet_backbone import UNetEncoder, SimpleUNetDecoder
# DEBUG
from util.utils import get_tversky_loss
from pdb import set_trace

import pickle
import torchvision

# options for type of prototypes
FG_PROT_MODE = 'gridconv+' # using both local and global prototype
BG_PROT_MODE = 'gridconv' #gridconv  # using local prototype only. 
# Also 'mask' refers to using global prototype only (as done in vanilla PANet)

# thresholds for deciding class of prototypes
FG_THRESH = 0.95
BG_THRESH = 0.95

class FewShotSeg(nn.Module):
    """
    ALPNet with UNet Encoder
    Args:
        in_channels:        Number of input channels
        cfg:                Model configurations
    """
    def __init__(self, in_channels=3, pretrained_path=None, cfg=None):
        super(FewShotSeg, self).__init__()
        self.pretrained_path = pretrained_path
        self.config = cfg or {'align': False}
        self.get_encoder(in_channels)
        self.get_decoder()
        self.get_cls()

    def get_encoder(self, in_channels):
        """
        Initialize UNet encoder instead of ResNet
        """
        use_pretrained = self.config.get('use_pretrained', True)
        self.encoder = UNetEncoder(in_channels=in_channels, 
                                 features=[64, 128, 256, 512],
                                 use_pretrained=use_pretrained)

        if self.pretrained_path:
            self.load_state_dict(torch.load(self.pretrained_path)['model'], strict=False)
            print(f'###### Pre-trained model {self.pretrained_path} has been loaded ######')

    def get_decoder(self):
        """
        Initialize UNet decoder for final mask prediction
        """
        # Use simplified decoder to avoid channel mismatch
        encoder_channels = [64, 64, 128, 256, 512]  # ResNet34 channels
        self.decoder = SimpleUNetDecoder(encoder_channels=encoder_channels, n_classes=2)

    def get_cls(self):
        """
        Obtain the similarity-based classifier
        """
        proto_hw = self.config["proto_grid_size"]
        feature_hw = self.config["feature_hw"]
        assert self.config['cls_name'] == 'grid_proto'
        if self.config['cls_name'] == 'grid_proto':
            self.cls_unit = MultiProtoAsWCos(proto_grid = [proto_hw, proto_hw], 
                                            feature_hw =  self.config["feature_hw"])
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
        assert n_queries == 1

        sup_bsize = supp_imgs[0][0].shape[0]
        img_size = supp_imgs[0][0].shape[-2:]
        qry_bsize = qry_imgs[0].shape[0]

        assert sup_bsize == qry_bsize == 1

        imgs_concat = torch.cat([torch.cat(way, dim=0) for way in supp_imgs]
                                + [torch.cat(qry_imgs, dim=0),], dim=0)

        # Get encoder features - UNet encoder returns multiple feature maps
        encoder_features = self.encoder(imgs_concat)
        # Remove debug prints by commenting out
        # print(f"Encoder output shapes: {[f.shape for f in encoder_features]}")
        
        # Use the deepest feature map for prototype computation
        img_fts = encoder_features[-1]  # [B, 512, H/16, W/16]
        fts_size = img_fts.shape[-2:]

        # Separate support and query features
        supp_fts = img_fts[:n_ways * n_shots * sup_bsize].view(
            n_ways, n_shots, sup_bsize, -1, *fts_size)  # Wa x Sh x B x C x H' x W'
        qry_fts = img_fts[n_ways * n_shots * sup_bsize:].view(
            n_queries, qry_bsize, -1, *fts_size)   # N x B x C x H' x W'
        
        # Get query encoder features for decoder
        qry_encoder_features = [feat[n_ways * n_shots * sup_bsize:] for feat in encoder_features]
        
        fore_mask = torch.stack([torch.stack(way, dim=0)
                                 for way in fore_mask], dim=0)
        fore_mask = torch.autograd.Variable(fore_mask, requires_grad = True)
        back_mask = torch.stack([torch.stack(way, dim=0)
                                 for way in back_mask], dim=0)

        ###### Compute loss ######
        align_loss = 0
        outputs = []
        visualizes = []

        for epi in range(1):
            # Interpolate masks to feature size
            res_fg_msk = torch.stack([F.interpolate(fore_mask_w, size = fts_size, mode = 'bilinear') 
                                    for fore_mask_w in fore_mask], dim = 0)
            res_bg_msk = torch.stack([F.interpolate(back_mask_w, size = fts_size, mode = 'bilinear') 
                                    for back_mask_w in back_mask], dim = 0)

            scores = []
            assign_maps = []
            bg_sim_maps = []
            fg_sim_maps = []

            # Extract background prototypes
            _raw_score, _, aux_attr = self.cls_unit(qry_fts, supp_fts, res_bg_msk, mode = BG_PROT_MODE, 
                                                    fg = False, thresh = BG_THRESH, isval = isval, 
                                                    val_wsize = val_wsize, vis_sim = show_viz)

            scores.append(_raw_score)
            assign_maps.append(aux_attr['proto_assign'])
            if show_viz:
                bg_sim_maps.append(aux_attr['raw_local_sims'])

            # Extract foreground prototypes
            for way, _msk in enumerate(res_fg_msk):
                _raw_score, _, aux_attr = self.cls_unit(qry_fts, supp_fts, _msk.unsqueeze(0), fg = True, 
                                                        mode = FG_PROT_MODE,
                                                        thresh = FG_THRESH, isval = isval, 
                                                        val_wsize = val_wsize, vis_sim = show_viz)

                scores.append(_raw_score)
                if show_viz:
                    fg_sim_maps.append(aux_attr['raw_local_sims'])

            # Concatenate prototype scores
            proto_scores = torch.cat(scores, dim=1)  # N x (1 + Wa) x H' x W'
            
            # Use UNet decoder to generate final prediction
            final_pred = self.decoder(qry_encoder_features, proto_scores)
            outputs.append(F.interpolate(final_pred, size=img_size, mode='bilinear'))

            ###### Prototype alignment loss ######
            if self.config['align'] and self.training:
                try:
                    align_loss_epi = self.alignLoss(qry_fts[:, epi], proto_scores, supp_fts[:, :, epi],
                                                    fore_mask[:, :, epi], back_mask[:, :, epi])
                    align_loss += align_loss_epi
                except:
                    align_loss += 0

        output = torch.stack(outputs, dim=1)
        output = output.view(-1, *output.shape[2:])
        assign_maps = torch.stack(assign_maps, dim = 1)
        bg_sim_maps = torch.stack(bg_sim_maps, dim = 1) if show_viz else None
        fg_sim_maps = torch.stack(fg_sim_maps, dim = 1) if show_viz else None

        return output, align_loss / sup_bsize, [bg_sim_maps, fg_sim_maps], assign_maps

    def alignLoss(self, qry_fts, pred, supp_fts, fore_mask, back_mask):
        """
        Compute the loss for the prototype alignment branch
        """
        n_ways, n_shots = len(fore_mask), len(fore_mask[0])

        # Masks for getting query prototype
        pred_mask = pred.argmax(dim=1).unsqueeze(0)
        binary_masks = [pred_mask == i for i in range(1 + n_ways)]

        skip_ways = []

        ### added for matching dimensions to the new data format
        qry_fts = qry_fts.unsqueeze(0).unsqueeze(2)

        loss = []
        for way in range(n_ways):
            if way in skip_ways:
                continue
            # Get the query prototypes
            for shot in range(n_shots):
                img_fts = supp_fts[way: way + 1, shot: shot + 1]

                qry_pred_fg_msk = F.interpolate(binary_masks[way + 1].float(), size = img_fts.shape[-2:], mode = 'bilinear')

                # background
                qry_pred_bg_msk = F.interpolate(binary_masks[0].float(), size = img_fts.shape[-2:], mode = 'bilinear')
                scores = []

                _raw_score_bg, _, _ = self.cls_unit(qry = img_fts, sup_x = qry_fts, sup_y = qry_pred_bg_msk.unsqueeze(-3), 
                                                  fg = False, mode = BG_PROT_MODE, thresh = BG_THRESH)

                scores.append(_raw_score_bg)

                _raw_score_fg, _, _ = self.cls_unit(qry = img_fts, sup_x = qry_fts, sup_y = qry_pred_fg_msk.unsqueeze(-3), 
                                                    fg = True, mode = FG_PROT_MODE,
                                                    thresh = FG_THRESH)
                scores.append(_raw_score_fg)

                supp_pred = torch.cat(scores, dim=1)
                supp_pred = F.interpolate(supp_pred, size=fore_mask.shape[-2:], mode='bilinear')

                # Construct the support Ground-Truth segmentation
                supp_label = torch.full_like(fore_mask[way, shot], 255, device=img_fts.device).long()

                supp_label[fore_mask[way, shot] == 1] = 1
                supp_label[back_mask[way, shot] == 1] = 0
                # Compute Loss
                loss.append(F.cross_entropy(supp_pred, supp_label[None, ...], ignore_index=255) / n_shots / n_ways)
                loss.append(get_tversky_loss(supp_pred.argmax(dim = 1, keepdim = True), supp_label[None, ...], 0.3, 0.7 ,1.0) / n_shots / n_ways)

        return torch.sum(torch.stack(loss))