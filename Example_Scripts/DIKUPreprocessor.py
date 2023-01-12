# -*- coding: utf-8 -*-
"""
@author: Sebastian

Example of a Preprocessing class based on the GenericPreprocessor used in nnUNet.

From the below it is clear that the DIKUPreprocessor inherits functions from
the GenericPreprocessor (which could be further explored).

The only difference is in resample_and_normalize() where the DIKU pipeline performs contrast clipping.
"""
import numpy as np
import os
import pickle
from nnunet.preprocessing.preprocessing import resample_patient, GenericPreprocessor
from batchgenerators.utilities.file_and_folder_operations import *
from skimage import exposure

class DIKUPreprocessor(GenericPreprocessor):
    def resample_and_normalize(self, data, target_spacing, properties, seg=None, force_separate_z=None):
        """
        data and seg must already have been transposed by transpose_forward. properties are the un-transposed values
        (spacing etc)
        :param data:
        :param target_spacing:
        :param properties:
        :param seg:
        :param force_separate_z:
        :return:
        """

        # target_spacing is already transposed, properties["original_spacing"] is not so we need to transpose it!
        # data, seg are already transposed. Double check this using the properties
        original_spacing_transposed = np.array(properties["original_spacing"])[self.transpose_forward]
        before = {
            'spacing': properties["original_spacing"],
            'spacing_transposed': original_spacing_transposed,
            'data.shape (data is transposed)': data.shape
        }

        # remove nans
        data[np.isnan(data)] = 0

        data, seg = resample_patient(data, seg, np.array(original_spacing_transposed), target_spacing, 3, 1,
                                     force_separate_z=force_separate_z, order_z_data=0, order_z_seg=0,
                                     separate_z_anisotropy_threshold=self.resample_separate_z_anisotropy_threshold)
        after = {
            'spacing': target_spacing,
            'data.shape (data is resampled)': data.shape
        }
        print("before:", before, "\nafter: ", after, "\n")

        if seg is not None:  
            seg[seg < -1] = 0

        properties["size_after_resampling"] = data[0].shape
        properties["spacing_after_resampling"] = target_spacing
        use_nonzero_mask = self.use_nonzero_mask

        assert len(self.normalization_scheme_per_modality) == len(data), "self.normalization_scheme_per_modality " \
                                                                         "must have as many entries as data has " \
                                                                         "modalities"
        assert len(self.use_nonzero_mask) == len(data), "self.use_nonzero_mask must have as many entries as data" \
                                                        " has modalities"

        #Perform contrast clipping, removing outliers in both directions and then [0, 1] normalizing. 
        for c in range(len(data)):
            scheme = self.normalization_scheme_per_modality[c]
            assert scheme == 'DIKU', "Scheme is not 'DIKU'. Wrong preprocessor or normalization scheme"
            lower_bound, upper_bound = np.percentile(data[c], (0.01, 99.99))
            data[c] = exposure.rescale_intensity(data[c], in_range=(lower_bound, upper_bound), 
                                                 out_range = (0, 1))
            if use_nonzero_mask[c]:
                data[c][seg[-1] < 0] = 0
        return data, seg, properties
