# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 12:05:59 2022

@author: Sebastian
"""
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.noise_transforms import GaussianBlurTransform
from batchgenerators.transforms.sample_normalization_transforms import RangeTransform
from batchgenerators.transforms.channel_selection_transforms import DataChannelSelectionTransform, \
    SegChannelSelectionTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform
from batchgenerators.transforms.utility_transforms import RemoveLabelTransform, RenameTransform, NumpyToTensor
from nnunet.training.data_augmentation.custom_transforms import MaskTransform
from nnunet.training.data_augmentation.downsampling import  DownsampleSegForDSTransform2
from nnunet.training.data_augmentation.DIKUTransforms import multiplicativeNoise, additiveNoise, \
    motionGhosting, gibbsRinging, biasField, contrastIntensity, contrastStretch

def get_DIKU_augmentation(dataloader_train, dataloader_val, patch_size, params=None,
                            seeds_train=None, seeds_val=None, deep_supervision_scales=None,
                            classes=None, pin_memory=True, single=False):
    """
    Same as get_DIKUDA_augmentation except ContrastStretch is removed, as it is applied in preprocessing
    And ContrastIntensity is removed as it is plain bad.
    """

    tr_transforms = []

    if params.get("selected_data_channels") is not None:
        tr_transforms.append(DataChannelSelectionTransform(params.get("selected_data_channels")))

    if params.get("selected_seg_channels") is not None:
        tr_transforms.append(SegChannelSelectionTransform(params.get("selected_seg_channels")))

    patch_size_spatial = patch_size

    if params.get("do_motionGhosting"):
        tr_transforms.append(motionGhosting(data_key = "data",
                                            p_per_sample = params.get("motionGhosting_p_per_sample"),
                                            alpha = params.get("motionGhosting_alpha"),
                                            numReps = params.get("motionGhosting_numReps"),
                                            dim = params.get("motionGhosting_dim")))
    if params.get("do_gibbsRinging"):
        tr_transforms.append(gibbsRinging(data_key = "data",
                                          p_per_sample = params.get("gibbsRinging_p_per_sample"),
                                          cutFreq = params.get("gibbsRinging_cutFreq"),
                                          dim = params.get("gibbsRinging_dim")))

    if params.get("do_biasField"):
        tr_transforms.append(biasField(data_key = "data",
                                       p_per_sample = params.get("biasField_p_per_sample")))

            
    tr_transforms.append(SpatialTransform(patch_size_spatial, patch_center_dist_from_border=params.get("random_crop_dist_to_border"),
        do_elastic_deform=params.get("do_elasticDeform"), alpha=params.get("elasticDeform_alpha"),
        sigma = params.get("elasticDeform_sigma"), 
        do_rotation=params.get("do_rotation"), 
        angle_x = params.get("rotation_x"), angle_y = params.get("rotation_y"), 
        angle_z = params.get("rotation_z"),
        do_scale=params.get("do_scale"), scale=params.get("scale_range"), 
        border_mode_data=params.get("border_mode_data"),
        border_cval_seg=-1, random_crop=params.get("random_crop"), 
        p_el_per_sample=params.get("elasticDeform_p_per_sample"), 
        p_rot_per_sample=params.get("rotation_p_per_sample"),        
        p_rot_per_axis = params.get("rotation_p_per_axis"),
        p_scale_per_sample = params.get("scale_p_per_sample")
    ))
    
    
    if params.get("do_multiplicativeNoise"):
        tr_transforms.append(multiplicativeNoise(data_key = "data", 
                                                 p_per_sample = params.get("multiplicativeNoise_p_per_sample"),
                                                 sigma = params.get("multiplicativeNoise_sigma")))

    if params.get("do_additiveNoise"):
        tr_transforms.append(additiveNoise(data_key = "data", 
                                           p_per_sample= params.get("additiveNoise_p_per_sample"),
                                           mean = params.get("additiveNoise_mean"),
                                           sigma = params.get("additiveNoise_sigma")))
    if params.get("do_blurring"):
        tr_transforms.append(GaussianBlurTransform(blur_sigma=params.get("blurring_sigma"), 
                                                   different_sigma_per_channel= params.get("blurring_per_channel"), 
                                                   p_per_sample= params.get("blurring_p_per_sample"),
                                                   different_sigma_per_axis = params.get("blurring_per_axis"),
                                                   p_isotropic = params.get("blurring_p_isotropic")))
        
    if params.get("do_mirror") or params.get("mirror"):
        tr_transforms.append(MirrorTransform(params.get("mirror_axes")))
        
    if params.get("do_normalization"):
        tr_transforms.append(RangeTransform(data_key = "data",
                                            rnge = (0, 1),
                                            per_channel=False))

    #We do the rest as these are formatting transformations rather than data augmentations
    #SpatialTransform to obtain correct PatchSize

    tr_transforms.append(RemoveLabelTransform(-1, 0))
    tr_transforms.append(RenameTransform('seg', 'target', True))

    if deep_supervision_scales is not None:
        tr_transforms.append(DownsampleSegForDSTransform2(deep_supervision_scales, 0, input_key='target',
                                                          output_key='target'))

    tr_transforms.append(NumpyToTensor(['data', 'target'], 'float'))
    tr_transforms = Compose(tr_transforms)
    
    if single:
        print("using SingleThreadedAugmenter. This will be slower, make sure it is on purpose.")
        batchgenerator_train = SingleThreadedAugmenter(dataloader_train, tr_transforms)
    else:
        batchgenerator_train = MultiThreadedAugmenter(dataloader_train, tr_transforms, params.get('num_threads'),
                                                   params.get("num_cached_per_thread"), seeds=seeds_train,
                                                   pin_memory=pin_memory)

    #Validation transforms - limited to compatibility transforms
    
    val_transforms = []
    
    val_transforms.append(RemoveLabelTransform(-1, 0))
    if params.get("selected_data_channels") is not None:
        val_transforms.append(DataChannelSelectionTransform(params.get("selected_data_channels")))
    if params.get("selected_seg_channels") is not None:
        val_transforms.append(SegChannelSelectionTransform(params.get("selected_seg_channels")))
        
    val_transforms.append(RenameTransform('seg', 'target', True))

    if deep_supervision_scales is not None:
        val_transforms.append(DownsampleSegForDSTransform2(deep_supervision_scales, 0, input_key='target',
                                                           output_key='target'))
        
    val_transforms.append(NumpyToTensor(['data', 'target'], 'float'))
    val_transforms = Compose(val_transforms)
    
    if single:
        batchgenerator_val = SingleThreadedAugmenter(dataloader_val, val_transforms)
    else:
        batchgenerator_val = MultiThreadedAugmenter(dataloader_val, val_transforms, max(params.get('num_threads') // 2, 1),
                                                    params.get("num_cached_per_thread"), seeds=seeds_val,
                                                    pin_memory=pin_memory)
    
    return batchgenerator_train, batchgenerator_val
