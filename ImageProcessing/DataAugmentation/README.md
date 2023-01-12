All the augmentation implementations are based on the original Matlab implementations of Mostafa Mehdipour found in 
https://github.com/Mostafa-Ghazi/MRI-Augmentation

For the Elastic Deformation, Rotation, Blurring and Mirroring augmentations we use the standard nnUNet/batchgenerators implementations of these.

The Augmentations are implemented in [DIKUTransforms](DIKUTransforms.py) in accordance with the nnUNet pipeline conventions.

The current parameters for the augmentations are set in [DIKU_augmentation_paramsV2](DIKU_augmentation_paramsV2.py). This includes 
values such as their % of being applied and their intensity (e.g. degree of deformation/noise)

The pipeline calling the augmentations and values is found in [get_DIKU_augmentation](get_DIKU_augmentation.py). This is written to conform with 
nnUNet standards. It should be called in a custom nnUNetTrainerMyVersion.py script.
