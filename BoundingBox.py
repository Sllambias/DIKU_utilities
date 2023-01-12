#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 13:25:09 2022

@author: zcr545
"""
import numpy as np

class Bbox:
    def __init__(self, array = None, label = None, pad = 0):
        #Create bounding box for the object.

        x1, x2, y1, y2, z1, z2 = self.get_bounding_box_for_label(array, label)
        
        self.xmax = max(x1, x2) + pad
        self.xmin = min(x1, x2) - pad
        self.ymax = max(y1, y2) + pad
        self.ymin = min(y1, y2) - pad
        self.zmax = max(z1, z2) + pad
        self.zmin = min(z1, z2) - pad
        self.box = [self.xmin, self.xmax, self.ymin, self.ymax, self.zmin, self.zmax]
        self.width = abs(self.xmax - self.xmin)
        self.height = abs(self.ymax - self.ymin)
        self.depth = abs(self.zmax - self.zmin)
    
    def get_box(self):
        return self.box

    def get_bounding_box_for_label(self, segmentation, label=None):
        #Returns a tuple of values
        #corresponding to Left, Right, Bottom, Top, Near, Far
        if not label:
            label = 1
        a = np.where(segmentation == label)
        bbox = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1]), np.min(a[2]), np.max(a[2])
        return bbox        

    @property
    def area(self):
        """
        Calculates the surface area. useful for IOU!
        """
        return self.width * self.height * self.depth

    def intersect(self, bbox):
        
        #Checks that exclude intersection
        if self.xmin > bbox.xmax or bbox.xmin > self.xmax:
            return False
            
        if self.ymin > bbox.ymax or bbox.ymin > self.ymax:
            return False
        
        if self.zmin > bbox.zmax or bbox.zmin > self.zmax:
            return False
        
        
        x1 = min(self.xmax, bbox.xmax)
        x2 = max(self.xmin, bbox.xmin)
        
        y1 = min(self.ymax, bbox.ymax)
        y2 = max(self.ymin, bbox.ymin)
        
        z1 = min(self.zmax, bbox.zmax)
        z2 = max(self.zmin, bbox.zmin)
        
        intersection = max(x1 - x2, 0) * max(y1 - y2, 0) * max(z1 - z2, 0)
        return intersection

    def iou(self, bbox):
        intersection = self.intersect(bbox)
        if not intersection:
            return False
        
        iou = intersection / float(self.area + bbox.area - intersection)
        # return the intersection over union value
        return iou

def remove_small_objects(array=None, **kwargs):
    """
    This function will remove any connected components smaller
    than the minimum size in mm
    """
    labels, counts = np.unique(array, return_counts=True)
    labels = list(labels)
    minimum_size = kwargs["obj_size"]
    voxel_size = np.prod(kwargs["voxel_spacing"])
    
    for label in labels:
        if not voxel_size * counts[label] >= minimum_size:
            array[array == label] = 0
            labels.remove(label)
            
    return array, labels
    
def sensitivity(TP, FP, TN, FN):
    if (TP + FN) == 0:
        return float("NaN")
    return float(TP / (TP + FN))

def precision(TP, FP, TN, FN):
    if (TP + FP) == 0:
        return float("NaN")
    return float(TP / (TP + FP))

def dice(TP, FP, TN, FN):
    if (TP + FP) == 0:
        return float("NaN")
    return float(2. * TP / (2 * TP + FP + FN))

def jaccard(TP, FP, TN, FN):
    if (TP + FP) == 0:
        return float("NaN")
    return float(TP / (TP + FP + FN))
  
