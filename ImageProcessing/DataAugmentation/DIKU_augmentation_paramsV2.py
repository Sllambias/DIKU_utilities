# -*- coding: utf-8 -*-
"""
2nd iteration of the parameter values.
V1 still exists as results were reported using those values.
@author: Sebastian
"""
### DIKU DATA AUGMENTATION PARAMETERS
import numpy as np
from copy import deepcopy

DIKU_3D_augmentation_paramsV2 = {
    
    "do_nnunet": False,
    "do_normalization": True,
    
    "selected_data_channels": None,
    "selected_seg_channels": [0],

    "do_additiveNoise": True,
    "additiveNoise_p_per_sample": 0.33,
    "additiveNoise_mean": 0,
    "additiveNoise_sigma": 1e-4 * np.random.uniform(),
    
    "do_biasField": True,
    "biasField_p_per_sample": 0.5,
  
    "do_elasticDeform": True,
    "elasticDeform_p_per_sample": 0.33,
    "elasticDeform_alpha": (200, 600),
    "elasticDeform_sigma": (20, 30),
    
    "do_gibbsRinging": True,
    "gibbsRinging_p_per_sample": 0.33,
    "gibbsRinging_cutFreq": np.random.randint(96, 129),
    "gibbsRinging_dim": np.random.randint(0,3),
    
    "do_motionGhosting": True,
    "motionGhosting_p_per_sample": 0.33,
    "motionGhosting_alpha": np.random.uniform(0.85, 0.95),
    "motionGhosting_numReps": np.random.randint(2, 11),
    "motionGhosting_dim": np.random.randint(0, 3),
    
    "do_multiplicativeNoise": True,
    "multiplicativeNoise_p_per_sample": 0.33,
    "multiplicativeNoise_mean": 0,
    "multiplicativeNoise_sigma": 1e-3 * np.random.uniform(),
    
    "do_rotation": True,
    "rotation_p_per_sample": 0.33,
    "rotation_p_per_axis": 0.66,
    "rotation_x": (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
    "rotation_y": (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
    "rotation_z": (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),

    "random_crop": False,
    "random_crop_dist_to_border": None,

    "do_blurring": False,
    "blurring_per_channel": False,
    "blurring_sigma": (0., 1.),
    "blurring_p_per_sample": 0.,
    "blurring_per_axis": False,
    "blurring_p_isotropic": 0.,
    
    #only relevant if do_nnunet == True
    "do_gamma": None,
    "gamma_retain_stats": True,
    "gamma_range": None,
    "p_gamma": 0.,
    
    #only relevant if do_nnunet == True
    "do_mirror": False,
    "mirror_axes": (0, 1, 2),

    "do_scale":False,
    "scale_range": (1, 1),
    "scale_p_per_sample": 0.33,

    "mask_was_used_for_normalization": None,
    "border_mode_data": "constant",

    #"num_threads": 12 if 'nnUNet_n_proc_DA' not in os.environ else int(os.environ['nnUNet_n_proc_DA']),
    "num_threads": 6,
    "num_cached_per_thread": 2,
}

DIKU_2D_augmentation_paramsV2 = deepcopy(DIKU_3D_augmentation_paramsV2)
DIKU_2D_augmentation_paramsV2['mirror_axes'] = (0, 1)
DIKU_2D_augmentation_paramsV2['rotation_y'] = (-0. / 360 * 2. * np.pi, 0. / 360 * 2. * np.pi)
DIKU_2D_augmentation_paramsV2['rotation_z'] = (-0. / 360 * 2. * np.pi, 0. / 360 * 2. * np.pi)
DIKU_2D_augmentation_paramsV2['motionGhosting_dim'] = np.random.randint(0,2)
DIKU_2D_augmentation_paramsV2['gibbsRinging_dim'] = np.random.randint(0,2)


DIKU_3D_augmentation_paramsBabyScaling = deepcopy(DIKU_3D_augmentation_paramsV2)
DIKU_3D_augmentation_paramsBabyScaling['do_scale'] = True
DIKU_3D_augmentation_paramsBabyScaling['scale_range'] = (1, 1.5)

DIKU_2D_augmentation_paramsBabyScaling = deepcopy(DIKU_2D_augmentation_paramsV2)
DIKU_2D_augmentation_paramsBabyScaling['do_scale'] = True
DIKU_2D_augmentation_paramsBabyScaling['scale_range'] = (1, 1.5)

DIKU_3D_augmentation_paramsV3 = deepcopy(DIKU_3D_augmentation_paramsV2)
DIKU_3D_augmentation_paramsV3["do_blurring"] = True
DIKU_3D_augmentation_paramsV3["blurring_per_channel"] = True
DIKU_3D_augmentation_paramsV3["blurring_sigma"] = (0., 1.)
DIKU_3D_augmentation_paramsV3["blurring_p_per_sample"] = 0.33
DIKU_3D_augmentation_paramsV3["blurring_per_axis"] = True
DIKU_3D_augmentation_paramsV3["blurring_p_isotropic"] = 0.33
DIKU_3D_augmentation_paramsV3["do_mirror"] = True

DIKU_2D_augmentation_paramsV3 = deepcopy(DIKU_3D_augmentation_paramsV3)
DIKU_2D_augmentation_paramsV3['mirror_axes'] = (0, 1)
DIKU_2D_augmentation_paramsV3['rotation_y'] = (-0. / 360 * 2. * np.pi, 0. / 360 * 2. * np.pi)
DIKU_2D_augmentation_paramsV3['rotation_z'] = (-0. / 360 * 2. * np.pi, 0. / 360 * 2. * np.pi)
DIKU_2D_augmentation_paramsV3['motionGhosting_dim'] = np.random.randint(0,2)
DIKU_2D_augmentation_paramsV3['gibbsRinging_dim'] = np.random.randint(0,2)
                              
                              
