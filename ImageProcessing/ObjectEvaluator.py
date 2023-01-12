#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zcr545


Evaluation Class implemented to work in the nnUNet pipeline.

For functionality the standard nnUNet 'evaluate_folder()' and 'nnunet_evaluate_folder()' must be changed to accept 
arguments for the Evaluator Class, object sizes and object padding. 
The default functions are found in nnUNet/nnunet/evaluation/evaluator.py

An implementation adapting the functions for Object Evaluation can be seen below:

def evaluate_folder(folder_with_gts: str, folder_with_predictions: str, labels: tuple, evaluator: str = NiftiEvaluator, **metric_kwargs):
    files_gt = subfiles(folder_with_gts, suffix=".nii.gz", join=False)
    files_pred = subfiles(folder_with_predictions, suffix=".nii.gz", join=False)
    assert all([i in files_pred for i in files_gt]), f"files in {folder_with_gts} are missing in folder_with_predictions: {folder_with_predictions}"
    assert all([i in files_gt for i in files_pred]), f"files missing in folder_with_gts: {folder_with_gts}"
    test_ref_pairs = [(join(folder_with_predictions, i), join(folder_with_gts, i)) for i in files_pred]
    if evaluator == 'ObjectEvaluator':
        json_output_file = json_output_file=join(folder_with_predictions, "summary_object.json")
        labels = None
        evaluator = ObjectEvaluator
    else:
        json_output_file = json_output_file=join(folder_with_predictions, "summary.json")
    res = aggregate_scores(test_ref_pairs, json_output_file=json_output_file,
                           num_threads=8, labels=labels, evaluator=evaluator, **metric_kwargs)
    return res


def nnunet_evaluate_folder():
    import argparse
    parser = argparse.ArgumentParser("Evaluates the segmentations located in the folder pred. Output of this script is "
                                     "a json file. At the very bottom of the json file is going to be a 'mean' "
                                     "entry with averages metrics across all cases")
    parser.add_argument('-ref', required=True, type=str, help="Folder containing the reference segmentations in nifti "
                                                              "format.")
    parser.add_argument('-pred', required=True, type=str, help="Folder containing the predicted segmentations in nifti "
                                                               "format. File names must match between the folders!")
    parser.add_argument('-l', nargs='+', type=int, required=True, help="List of label IDs (integer values) that should "
                                                                       "be evaluated. Best practice is to use all int "
                                                                       "values present in the dataset, so for example "
                                                                       "for LiTS the labels are 0: background, 1: "
                                                                       "liver, 2: tumor. So this argument "
                                                                       "should be -l 1 2. You can if you want also "
                                                                       "evaluate the background label (0) but in "
                                                                       "this case that would not gie any useful "
                                                                       "information.")
    parser.add_argument("-obj_size", required=False, type=int, default=None)
    parser.add_argument("-obj_padding", required=False, type=int, default=None)
    parser.add_argument("-evaluator", default=NiftiEvaluator, help="Defaults to regular pixel-wise segmentation evaluation. "
                                                                     "Use 'ObjectEvaluator' for bounding box-based object "
                                                                     "detection evaluation.")
    args = parser.parse_args()
    return evaluate_folder(args.ref, args.pred, args.l, args.evaluator, obj_size = args.obj_size, obj_padding = args.obj_padding)
"""

import SimpleITK as sitk
import numpy as np
from skimage.measure import label as sk_label
from nnunet.evaluation.object_detection import remove_small_objects, Bbox, sensitivity, precision, jaccard, dice
import inspect
from collections import OrderedDict

class ObjectEvaluator:
    def __init__(self, *args, **kwargs):
        self.test = None
        self.test_bboxes = None
        self.reference = None
        self.ref_bboxes = None
        self.metrics = ['IOU', 'Sensitivity', 'Precision', 'Total Objects Reference',
                        'Total Objects Test', 'Total True Positives',  
                        'Total False Positives', 'Total False Negatives', 'Dice', 'Jaccard']
        self.OBJECT_METRICS = {
            "IOU": self.get_iou, "Sensitivity": self.get_sensitivity,
            "Precision": self.get_precision, "Total Objects Reference": self.get_reference_objects,
            "Total Objects Test" : self.get_test_objects,
            "Total True Positives": self.get_TP,
            "Total False Positives": self.get_FP, "Total False Negatives": self.get_FN,
            "Dice" : self.get_dice, "Jaccard": self.get_jaccard}
        self.test_nifti = None
        self.reference_nifti = None
        self.advanced_metrics = []
        self.labels = None
        self.test_label_counts = None
        self.reference_label_counts = None
        self.TP = 0
        self.FP = 0
        self.TN = 0
        self.FN = 0
        self.iou_results = []


    def reset(self):

        self.test_bboxes = None
        self.ref_bboxes = None
        self.labels = None
        self.test_label_counts = None
        self.reference_label_counts = None
        self.TP = 0
        self.FP = 0
        self.TN = 0
        self.FN = 0
        self.iou_results = []


    def set_test(self, test):
        """Set the test segmentation."""
        
        if test is not None:
            self.test_nifti = sitk.ReadImage(test)
            self.test = sk_label(sitk.GetArrayFromImage(self.test_nifti))
        else:
            self.test_nifti = None
            self.test = test
        self.reset()


    def set_reference(self, reference):
        """Set the reference segmentation."""

        if reference is not None:
            self.reference_nifti = sitk.ReadImage(reference)
            self.reference = sk_label(sitk.GetArrayFromImage(self.reference_nifti))
        else:
            self.reference_nifti = None
            self.reference = reference
        self.reset()
    
    def construct_labels(self):
        """
        Required for functionality
        """
        pass
    
    
    def construct_bounding_boxes(self, **kwargs):
        """
        Create function for constructing the bounding boxes used for
        the object evaluation functions.
        """
        #Filter out objects smaller than the required size
        
        self.reference, ref_labels = remove_small_objects(self.reference, **kwargs)

        #We don't want to filter predicted objects at this time.
        obj_size = kwargs["obj_size"]
        kwargs["obj_size"] = 0
        self.test, test_labels = remove_small_objects(self.test, **kwargs)
        kwargs["obj_size"] = obj_size
     
        #Get bounding boxes of remaining PRED CMBs
        self.test_labels = test_labels
        self.test_bboxes = [Bbox(self.test, label, kwargs["obj_padding"]) for label in test_labels if label != 0]
        
        #Get bounding boxes of remaning TRUE CMBs
        self.ref_labels = ref_labels
        self.ref_bboxes = [Bbox(self.reference, label, kwargs["obj_padding"]) for label in ref_labels if label != 0]
          
    def compute(self):
        #Compare each true CMB with each of the predicted CMBs
        
        for test_bbox in self.test_bboxes:
            for ref_bbox in self.ref_bboxes:
                iou = test_bbox.iou(ref_bbox)
                #if we have overlap store results
                if iou:
                    self.iou_results.append(iou)
                    break
                            

    def set_TP(self):
        self.TP = len(self.iou_results)
        
    def set_FP(self):
        self.FP = len(self.test_labels) - self.TP
        
    def set_FN(self):
        self.FN = len(self.ref_labels) - self.TP       

    def set_all(self):
        self.set_TP()
        self.set_FP()
        self.set_FN()

    def get_iou(self):
        if self.iou_results:
            return np.mean(self.iou_results)
        else:
            return 0.
        
    def get_TP(self):
        return self.TP
    
    def get_FP(self):
        return self.FP
    
    def get_FN(self):
        return self.FN
    
    def get_jaccard(self):
        return jaccard(self.TP, self.FP, self.TN, self.FN)
    
    def get_dice(self):
        return dice(self.TP, self.FP, self.TN, self.FN)
        
    def get_sensitivity(self):
        return sensitivity(self.TP, self.FP, self.TN, self.FN)
    
    def get_precision(self):
        return precision(self.TP, self.FP, self.TN, self.FN)
    
    def get_reference_objects(self):
        return len(self.ref_labels)
    
    def get_test_objects(self):
        return len(self.test_labels)

    def evaluate(self, test=None, reference=None, **metric_kwargs):

        if "voxel_spacing" not in metric_kwargs:
            metric_kwargs["voxel_spacing"] = np.array(self.test_nifti.GetSpacing())
 
        if "obj_size" not in metric_kwargs:
            metric_kwargs["obj_size"] = 5.
            
        if "obj_padding" not in metric_kwargs:
            metric_kwargs["obj_padding"] = 0.
            
        """Compute metrics for segmentations."""
        if test is not None:
            self.set_test(test)
    
        if reference is not None:
            self.set_reference(reference)
    
        if self.test is None or self.reference is None:
            raise ValueError("Need both test and reference segmentations.")
    
        if self.test_bboxes is None or self.ref_bboxes is None:
            self.construct_bounding_boxes(**metric_kwargs)
    
        #Here we do the actual calculations. Other functions will simply pull
        #results computed by this call.
        self.compute()
        self.set_all()
        
        self.metrics.sort()
    
        # get functions for evaluation
        # somewhat convoluted, but allows users to define additonal metrics
        # on the fly, e.g. inside an IPython console
        _funcs = {m: self.OBJECT_METRICS[m] for m in self.metrics + self.advanced_metrics}
        frames = inspect.getouterframes(inspect.currentframe())
        for metric in self.metrics:
            for f in frames:
                if metric in f[0].f_locals:
                    _funcs[metric] = f[0].f_locals[metric]
                    break
            else:
                if metric in _funcs:
                    continue
                else:
                    raise NotImplementedError(
                        "Metric {} not implemented.".format(metric))
    
        # get results
        self.result = OrderedDict()
    
        eval_metrics = self.metrics
    
        self.result = OrderedDict()
        self.result["objects"] = OrderedDict()
        for metric in eval_metrics:
            self.result["objects"][metric] = _funcs[metric]()
            
        self.result["objects"]["X spacing"] = float(metric_kwargs["voxel_spacing"][0])
        self.result["objects"]["Y spacing"] = float(metric_kwargs["voxel_spacing"][1])
        self.result["objects"]["Z spacing"] = float(metric_kwargs["voxel_spacing"][2])
        self.result["objects"]["min. CMB size"] = float(metric_kwargs["obj_size"])
        self.result["objects"]["padding"] = float(metric_kwargs["obj_padding"])

        return self.result
