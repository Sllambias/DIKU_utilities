# -*- coding: utf-8 -*-
"""
@author: Sebastian

An example of an ExperimentPlanning script based on the original nnUNet scripts.
Adapted to point use the DIKUPreprocessor and save plans with a 'DIKU' tag, 
so they can be retrieved by a training script later.

It also specifies that the normalization scheme is the "DIKU" scheme which includes 
contrast clipping (as shown in the DIKUPreprocessor script)
"""

from nnunet.experiment_planning.experiment_planner_baseline_3DUNet_v21 import ExperimentPlanner3D_v21
from batchgenerators.utilities.file_and_folder_operations import join
from collections import OrderedDict

class ExperimentPlanner3D_DIKU(ExperimentPlanner3D_v21):
    def __init__(self, folder_with_cropped_data, preprocessed_output_folder):
        super(ExperimentPlanner3D_DIKU, self).__init__(folder_with_cropped_data, preprocessed_output_folder)
        self.preprocessor_name = "DIKUPreprocessor"
        self.data_identifier = "nnUNetData_DIKU"
        self.plans_fname = join(self.preprocessed_output_folder,
                                "nnUNetPlansDIKU_plans_3D.pkl")
        
    def determine_normalization_scheme(self):
        schemes = OrderedDict()
        modalities = self.dataset_properties['modalities']
        num_modalities = len(list(modalities.keys()))

        for i in range(num_modalities):
            schemes[i] = "DIKU"
        return schemes
