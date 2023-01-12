#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example script of clear data management.
Based on the nnunet pipeline and their dataset_conversion scripts.
"""
import gzip
import shutil
from nnunet.paths import nnUNet_raw_data
from nnunet.dataset_conversion.utils import generate_dataset_json
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p, subfiles

###
#First we decide on the task name and prefix. Task name is used to identify the dataset.
#Prefix is used to identify its samples
task_name = 'Task123_FakeTask'
prefix = 'FakeTask' 

###
#Now we specify the target paths, where the task will be saved.
#We also create the folders we will use now.
#To avoid hardcoded paths we use environment variables.

target_base = join(nnUNet_raw_data, task_name)

target_imagesTr = join(target_base, "imagesTr")
target_imagesTs = join(target_base, "imagesTs")
target_labelsTs = join(target_base, "labelsTs")
target_labelsTr = join(target_base, "labelsTr")

maybe_mkdir_p(target_imagesTr)
maybe_mkdir_p(target_labelsTs)
maybe_mkdir_p(target_imagesTs)
maybe_mkdir_p(target_labelsTr)

images_dir = join(base, 'Images')
labels_dir = join(base, 'Labels')

###
#Now we specify the path to the dataset. This is preferably one of the only hardcoded paths used in the project.
#We also split the samples in the dataset directory into train and test with a specified random state.
base = "/path/to/dataset" 
image_files = subfiles(images_dir, join=False, suffix=suffix, sort=True)

training_samples, test_samples = train_test_split(image_files, test_size=0.25, random_state=12345)

###
#Now store the training and testing samples in their appropriate folders, following naming conventions.
#This would also be the place to include any reorientation operations or similar.
for tr_sample in training_samples:
    serial_number = tr_sample[:-len(suffix)]
    image_file = open(join(images_dir, tr_sample), 'rb')
    label = open(join(labels_dir, tr_sample), 'rb')
    shutil.copyfileobj(image_file, gzip.open(f'{target_imagesTr}/{prefix}_{serial_number}_0000.nii.gz', 'wb'))
    shutil.copyfileobj(label, gzip.open(f'{target_labelsTr}/{prefix}_{serial_number}.nii.gz', 'wb'))
    
    
for ts_sample in test_samples:
    serial_number = ts_sample[:-len(suffix)]
    image_file = open(join(images_dir, ts_sample), 'rb')
    label = open(join(labels_dir, ts_sample), 'rb')
    shutil.copyfileobj(image_file, gzip.open(f'{target_imagesTs}/{prefix}_{serial_number}_0000.nii.gz', 'wb'))
    shutil.copyfileobj(label, gzip.open(f'{target_labelsTs}/{prefix}_{serial_number}.nii.gz', 'wb'))

###
#In the end we create the dataset json to store information about labels etc.
generate_dataset_json(join(target_base, 'dataset.json'), target_imagesTr, target_imagesTs, ('T1 weighted', ),
                          labels={0: 'background', 1: 'left hippocampus', 2: 'right hippocampus'}, dataset_name=task_name, license='hands off!')
