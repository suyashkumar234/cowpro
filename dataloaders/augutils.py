'''
Utilities for augmentation. Partly credit to Dr. Jo Schlemper
'''
from os.path import join

import torch
import numpy as np
import torchvision.transforms as deftfx
import dataloaders.image_transforms as myit
import copy

sabs_aug = {
        # turn flipping off as medical data has fixed orientations
'flip'      : { 'v':False, 'h':False, 't': False, 'p':0.25 },
'affine'    : {
  'rotate':5,
  'shift':(5,5),
  'shear':5,
  'scale':(0.9, 1.2), 
},
'elastic'   : {'alpha':10,'sigma':5},
'patch': 256,
'reduce_2d': True,
'gamma_range': (0.5, 1.5)
}

sabs_augv3 = {
'flip'      : { 'v':False, 'h':False, 't': False, 'p':0.25 },
'affine'    : {
  'rotate':30,
  'shift':(30,30),
  'shear':30,
  'scale':(0.8, 1.3), 
},
'elastic'   : {'alpha':20,'sigma':5}, 
'patch': 256,
'reduce_2d': True,
'gamma_range': (0.2, 1.8)
}

augs = {
    'sabs_aug': sabs_aug,
    'aug_v3': sabs_augv3, # more aggresive
}

# Fixed version of dataloaders/augutils.py
# Replace the get_geometric_transformer function with this fixed version

def get_geometric_transformer(aug, order=3):
    """
    Fixed version that properly handles RandomAffine and ElasticTransform
    order: interpolation degree. Select order=0 for augmenting segmentation 
    """
    affine     = aug['aug'].get('affine', 0)
    alpha      = aug['aug'].get('elastic',{'alpha': 0})['alpha']
    sigma      = aug['aug'].get('elastic',{'sigma': 0})['sigma']
    # flip       = aug['aug'].get('flip', {'v': True, 'h': True, 't': True, 'p':0.125})

    tfx = []
    # if 'flip' in aug['aug']:
    #     tfx.append(myit.RandomFlip3D(**flip))

    if 'affine' in aug['aug']:
        # Create a wrapper class that handles the M parameter automatically
        class FixedRandomAffine:
            def __init__(self, rotate, shift, shear, scale, scale_iso, order):
                self.random_affine = myit.RandomAffine(
                    rotation_range=rotate,
                    translation_range=shift,
                    shear_range=shear,
                    zoom_range=scale,
                    zoom_keep_aspect=scale_iso,
                    order=order
                )
            
            def __call__(self, image):
                # Generate the transformation matrix automatically
                M = self.random_affine.build_M(image.shape[:2])
                # Apply the transform with the matrix
                return self.random_affine(image, M)
        
        tfx.append(FixedRandomAffine(
            rotate=affine.get('rotate'),
            shift=affine.get('shift'),
            shear=affine.get('shear'),
            scale=affine.get('scale'),
            scale_iso=affine.get('scale_iso', True),
            order=order
        ))

    if 'elastic' in aug['aug']:
        # Create a wrapper for ElasticTransform that only returns the image
        class FixedElasticTransform:
            def __init__(self, alpha, sigma):
                self.elastic = myit.ElasticTransform(alpha, sigma)
            
            def __call__(self, image):
                # ElasticTransform returns (image, dx_params, dy_params)
                # We only want the image for the augutils version
                result = self.elastic(image)
                if isinstance(result, tuple):
                    return result[0]  # Return only the transformed image
                else:
                    return result
        
        tfx.append(FixedElasticTransform(alpha, sigma))
    
    input_transform = deftfx.Compose(tfx)
    return input_transform

'''def get_geometric_transformer(aug, order=3):
    """order: interpolation degree. Select order=0 for augmenting segmentation """
    affine     = aug['aug'].get('affine', 0)
    alpha      = aug['aug'].get('elastic',{'alpha': 0})['alpha']
    sigma      = aug['aug'].get('elastic',{'sigma': 0})['sigma']
    # flip       = aug['aug'].get('flip', {'v': True, 'h': True, 't': True, 'p':0.125})

    tfx = []
    # if 'flip' in aug['aug']:
    #     tfx.append(myit.RandomFlip3D(**flip))

    if 'affine' in aug['aug']:
        tfx.append(myit.RandomAffine(affine.get('rotate'),
                                     affine.get('shift'),
                                     affine.get('shear'),
                                     affine.get('scale'),
                                     affine.get('scale_iso',True),
                                     order=order))

    if 'elastic' in aug['aug']:
        tfx.append(myit.ElasticTransform(alpha, sigma))
    input_transform = deftfx.Compose(tfx)
    return input_transform'''

def get_intensity_transformer(aug):
    """some basic intensity transforms"""

    def gamma_tansform(img):
        gamma_range = aug['aug']['gamma_range']
        if isinstance(gamma_range, tuple):
            gamma = np.random.rand() * (gamma_range[1] - gamma_range[0]) + gamma_range[0]
            cmin = img.min()
            irange = (img.max() - cmin + 1e-5)

            img = img - cmin + 1e-5
            img = irange * np.power(img * 1.0 / irange,  gamma)
            img = img + cmin

        elif gamma_range == False:
            pass
        else:
            raise ValueError("Cannot identify gamma transform range {}".format(gamma_range))
        return img

    return gamma_tansform

def transform_with_label(aug):
    """
    Doing image geometric transform
    Proposed image to have the following configurations
    [H x W x C + CL]
    Where CL is the number of channels for the label. It is NOT in one-hot form
    """

    geometric_tfx = get_geometric_transformer(aug)
    intensity_tfx = get_intensity_transformer(aug)

    def transform(comp, c_label, c_img, use_onehot, nclass, **kwargs):
        """
        Args
        comp:               a numpy array with shape [H x W x C + c_label]
        c_label:            number of channels for a compact label. Note that the current version only supports 1 slice (H x W x 1)
        nc_onehot:          -1 for not using one-hot representation of mask. otherwise, specify number of classes in the label

        """
        comp = copy.deepcopy(comp)
        if (use_onehot is True) and (c_label != 1):
            raise NotImplementedError("Only allow compact label, also the label can only be 2d")
        assert c_img + 1 == comp.shape[-1], "only allow single slice 2D label"

        # geometric transform
        _label = comp[..., c_img ]
        _h_label = np.float32(np.arange( nclass ) == (_label[..., None]) )
        comp = np.concatenate( [comp[...,  :c_img ], _h_label], -1 )
        comp = geometric_tfx(comp)
        # round one_hot labels to 0 or 1
        t_label_h = comp[..., c_img : ]
        t_label_h = np.rint(t_label_h)
        assert t_label_h.max() <= 1
        t_img = comp[..., 0 : c_img ]

        # intensity transform
        t_img = intensity_tfx(t_img)

        if use_onehot is True:
            t_label = t_label_h
        else:
            t_label = np.expand_dims(np.argmax(t_label_h, axis = -1), -1)
        return t_img, t_label

    return transform

