# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 16:37:05 2022

@author: Sebastian
"""
from nnunet.utilities.to_torch import maybe_to_torch
from batchgenerators.transforms.abstract_transforms import AbstractTransform
from skimage import exposure
import numpy as np

class additiveNoise(AbstractTransform):
    """
    variables in DIKU_3D_augmentation_params:
        do_additiveNoise
        additiveNoise_p_per_sample
        additiveNoise_mean
        additiveNoise_sigma 
    """
    def __init__(self, data_key="data", p_per_sample=1,
                 mean = 0,
                 sigma = 1e-4 * np.random.uniform()):
        self.data_key = data_key
        self.p_per_sample = p_per_sample
        self.mean = mean
        self.sigma = sigma
    
    def __additiveNoise__(self, imageVolume, mean, sigma):
        # J = I+n
        gauss = np.random.normal(mean, sigma, imageVolume.shape)
        return imageVolume + gauss
    
    def __call__(self, **data_dict):
        assert (len(data_dict[self.data_key].shape) == 5 or len(data_dict[self.data_key].shape) == 4) and data_dict[self.data_key].shape[1] == 1, f"Incorrect data size or shape.\
            \nShould be (b, c, x, y, z) or (b, c, x, y) and is: {data_dict[self.data_key].shape}"
            
        for sample in range(len(data_dict[self.data_key])):
            if np.random.uniform() < self.p_per_sample:
                data_dict[self.data_key][sample][0] = self.__additiveNoise__(data_dict[self.data_key][sample][0], 
                self.mean, self.sigma)
        return data_dict


class biasField(AbstractTransform):
    """
    variables in DIKU_3D_augmentation_params:
        do_biasField
        biasField_p_per_sample
    """
    def __init__(self, data_key="data", p_per_sample=1):
        self.data_key = data_key
        self.p_per_sample = p_per_sample

    def __biasField__(self, imageVolume):
        if len(imageVolume.shape) == 3:
            x, y, z = imageVolume.shape
            X, Y, Z = np.meshgrid(
                np.linspace(0, x, x, endpoint=False),
                np.linspace(0, y, y, endpoint=False),
                np.linspace(0, z, z, endpoint=False), 
                indexing='ij')
            x0 = np.random.randint(0, x)
            y0 = np.random.randint(0, y)
            z0 = np.random.randint(0, z)
            G = 1 - (np.power((X - x0), 2) / (x ** 2) + 
                     np.power((Y - y0), 2) / (y ** 2) + 
                     np.power((Z - z0), 2) / (z ** 2))
        else:
            x, y = imageVolume.shape
            X, Y = np.meshgrid(
                np.linspace(0, x, x, endpoint=False),
                np.linspace(0, y, y, endpoint=False),
                indexing='ij')
            x0 = np.random.randint(0, x)
            y0 = np.random.randint(0, y)
            G = 1 - (np.power((X - x0), 2) / (x ** 2) + 
                     np.power((Y - y0), 2) / (y ** 2))
        return np.multiply(G, imageVolume)
    
    def __call__(self, **data_dict):
        assert (len(data_dict[self.data_key].shape) == 5 or len(data_dict[self.data_key].shape) == 4) and data_dict[self.data_key].shape[1] == 1, f"Incorrect data size or shape.\
            \nShould be (b, c, x, y, z) or (b, c, x, y) and is: {data_dict[self.data_key].shape}"
        
        for sample in range(len(data_dict[self.data_key])):
            if np.random.uniform() < self.p_per_sample:
                data_dict[self.data_key][sample][0] = self.__biasField__(data_dict[self.data_key][sample][0])
        return data_dict   


class contrastIntensity(AbstractTransform):
    """
    variables in DIKU_3D_augmentation_params:
        do_contrastIntensity
        contrastIntensity_p_per_sample
        contrastIntensity_factor 
    """
    def __init__(self, data_key="data", p_per_sample=1,
                 factor = np.random.uniform(0.75, 1.25)):
        self.data_key = data_key
        self.p_per_sample = p_per_sample
        self.factor = factor
    
    def __contrastIntensity__(self, imageVolume, factor):
        mean = np.mean(imageVolume)
        imageVolume = (imageVolume - mean) * factor + mean
        return imageVolume

    
    def __call__(self, **data_dict):
        assert len(data_dict[self.data_key].shape) == 5 and data_dict[self.data_key].shape[1] == 1, f"Incorrect data size or shape.\
            \nShould be (b, c, x, y, z) and is: {data_dict[self.data_key].shape}"
            
        for sample in range(len(data_dict[self.data_key])):
            if np.random.uniform() < self.p_per_sample:
                data_dict[self.data_key][sample][0] = self.__contrastIntensity__(data_dict[self.data_key][sample][0], 
                self.factor)
        return data_dict
    
    
class contrastStretch(AbstractTransform):
    """
    variables in DIKU_3D_augmentation_params:
        do_contrastStretch
        contrastStretch_p_per_sample
        contrastStretch_lower_tol
        contrastStretch_upper_tol
    """
    def __init__(self, data_key="data", p_per_sample=1,
                 lower_tol : float = 0.01, 
                 upper_tol : float = 99.99):
        self.data_key = data_key
        self.p_per_sample = p_per_sample
        self.lower_tol = lower_tol
        self.upper_tol = upper_tol
        self.low_out = 0
        self.upper_out = 1
        
        if not self.p_per_sample == 1 and not self.p_per_sample == 0:
            print("contrastStretch is not 0 or 1. This should always be the case as normalization is included!")
        
    def __contrastStretch__(self, imageVolume, lower:float = 0.01, upper:float = 99.99):
        lower_bound, upper_bound = np.percentile(imageVolume, (lower, upper))
        imageVolume = exposure.rescale_intensity(imageVolume, in_range=(lower_bound, upper_bound), out_range = (self.low_out, self.upper_out))
        return imageVolume
       
    def __call__(self, **data_dict):
        assert len(data_dict[self.data_key].shape) == 5 and data_dict[self.data_key].shape[1] == 1, f"Incorrect data size or shape.\
            \nShould be (b, c, x, y, z) and is: {data_dict[self.data_key].shape}"
            
        for sample in range(len(data_dict[self.data_key])):
            if np.random.uniform() < self.p_per_sample:
                data_dict[self.data_key][sample][0] = self.__contrastStretch__(data_dict[self.data_key][sample][0], 
                self.lower_tol, self.upper_tol)
        return data_dict


class gibbsRinging(AbstractTransform):
    """
    variables in DIKU_3D_augmentation_params:
        do_gibbsRinging
        gibbsRinging_p_per_sample
        gibbsRinging_cutFreq
        gibbsRinging_dim
    """
    def __init__(self, data_key="data", p_per_sample=1,
                 cutFreq = np.random.randint(96, 129), dim=np.random.randint(0,3)):
        self.data_key = data_key
        self.p_per_sample = p_per_sample
        self.cutFreq = cutFreq
        self.dim = dim

    def __gibbsRinging__(self, imageVolume, numSample, dim):
        if len(imageVolume.shape) == 3:
            assert dim in [0, 1, 2], "Incorrect or no dimension"
            
            h, w, d = imageVolume.shape
            if dim == 0:
                imageVolume = imageVolume.transpose(0, 2, 1)
                imageVolume = np.fft.fftshift(np.fft.fftn(imageVolume, s=[h, d, w]))
                imageVolume[:, :, 0 : int(np.ceil(w / 2) - np.ceil(numSample / 2))] = 0
                imageVolume[:, :, int(np.ceil(w / 2) + np.ceil(numSample / 2)) : w] = 0
                imageVolume = abs(np.fft.ifftn(np.fft.ifftshift(imageVolume), s=[h, d, w]))
                imageVolume = imageVolume.transpose(0, 2, 1)
            elif dim == 1:
                imageVolume = imageVolume.transpose(1, 2, 0)
                imageVolume = np.fft.fftshift(np.fft.fftn(imageVolume, s=[w, d, h]))
                imageVolume[:, :, 0 : int(np.ceil(h / 2) - np.ceil(numSample / 2))] = 0
                imageVolume[:, :, int(np.ceil(h / 2) + np.ceil(numSample / 2)) : h] = 0
                imageVolume = abs(np.fft.ifftn(np.fft.ifftshift(imageVolume), s=[w, d, h]))
                imageVolume = imageVolume.transpose(2, 0, 1)
            else:
                imageVolume = np.fft.fftshift(np.fft.fftn(imageVolume, s=[h, w, d]))
                imageVolume[:, :, 0 : int(np.ceil(d / 2) - np.ceil(numSample / 2))] = 0
                imageVolume[:, :, int(np.ceil(d / 2) + np.ceil(numSample / 2)) : d] = 0
                imageVolume = abs(np.fft.ifftn(np.fft.ifftshift(imageVolume), s=[h, w, d]))
        elif len(imageVolume.shape) == 2:
            assert dim in [0, 1], "incorrect or no dimension"
            h, w = imageVolume.shape
            if dim == 0:
                imageVolume = np.fft.fftshift(np.fft.fftn(imageVolume, s=[h, w]))
                imageVolume[:, 0 : int(np.ceil(w / 2) - np.ceil(numSample / 2))] = 0
                imageVolume[:, int(np.ceil(w / 2) + np.ceil(numSample / 2)) : w] = 0
                imageVolume = abs(np.fft.ifftn(np.fft.ifftshift(imageVolume), s = [h, w]))
            else:
                imageVolume = imageVolume.conj().T
                imageVolume = np.fft.fftshift(np.fft.fftn(imageVolume, s=[w, h]))
                imageVolume[:, 0 : int(np.ceil(h / 2) - np.ceil(numSample / 2))] = 0
                imageVolume[:, int(np.ceil(h / 2) + np.ceil(numSample / 2)) : h] = 0
                imageVolume = abs(np.fft.ifftn(np.fft.ifftshift(imageVolume), s = [w, h]))
                imageVolume = imageVolume.conj().T                      
        return imageVolume
    
    def __call__(self, **data_dict):
        assert (len(data_dict[self.data_key].shape) == 5 or len(data_dict[self.data_key].shape) == 4) and data_dict[self.data_key].shape[1] == 1, f"Incorrect data size or shape.\
            \nShould be (b, c, x, y, z) or (b, c, x, y) and is: {data_dict[self.data_key].shape}"
        for sample in range(len(data_dict[self.data_key])):
            if np.random.uniform() < self.p_per_sample:
                data_dict[self.data_key][sample][0] = self.__gibbsRinging__(data_dict[self.data_key][sample][0], 
                                                                            self.cutFreq, self.dim)
        return data_dict  


class motionGhosting(AbstractTransform):
    """
    variables in DIKU_3D_augmentation_params:
        do_motionGhosting
        motionGhosting_p_per_sample
        motionGhosting_alpha
        motionGhosting_numReps
        motionGhosting_dim
    """
    def __init__(self, data_key="data", p_per_sample=1,
                 alpha=np.random.uniform(0.85, 0.95), numReps=np.random.randint(2,5), dim=np.random.randint(0,3)):
        self.data_key = data_key
        self.p_per_sample = p_per_sample
        self.alpha = alpha
        self.numReps = numReps
        self.dim = dim
  
    def __motionGhosting__(self, imageVolume, alpha, numReps, dim):
        if len(imageVolume.shape) == 3:
            assert dim in [0, 1, 2], "Incorrect or no dimension"

            h, w, d = imageVolume.shape

            imageVolume = np.fft.fftn(imageVolume, s =[h, w, d])

            if dim == 0:
                imageVolume[0:-1:numReps, :, :] = alpha * imageVolume[0:-1:numReps, :, :]
            elif dim == 1:
                imageVolume[:, 0:-1:numReps, :] = alpha * imageVolume[:, 0:-1:numReps, :]
            else:
                imageVolume[:, :, 0:-1:numReps] = alpha * imageVolume[:, :, 0:-1:numReps]

            imageVolume = abs(np.fft.ifftn(imageVolume, s=[h, w, d]))
        if len(imageVolume.shape) == 2:
            assert dim in [0, 1], "Incorrect or no dimension"
            h, w = imageVolume.shape
            imageVolume = np.fft.fftn(imageVolume, s = [h, w])
            
            if dim == 0:
                imageVolume[0:-1:numReps, :] = alpha * imageVolume[0:-1:numReps, :]
            else:
                imageVolume[:, 0:-1:numReps] = alpha * imageVolume[:, 0:-1:numReps]
            imageVolume = abs(np.fft.ifftn(imageVolume, s=[h, w]))
        return imageVolume
    
    def __call__(self, **data_dict):
        assert (len(data_dict[self.data_key].shape) == 5 or len(data_dict[self.data_key].shape) == 4) and data_dict[self.data_key].shape[1] == 1, f"Incorrect data size or shape.\
            \nShould be (b, c, x, y, z) or (b, c, x, y) and is: {data_dict[self.data_key].shape}"
        for sample in range(len(data_dict[self.data_key])):
            if np.random.uniform() < self.p_per_sample:
                data_dict[self.data_key][sample][0] = self.__motionGhosting__(data_dict[self.data_key][sample][0], 
                                                                            self.alpha, self.numReps, self.dim)
        return data_dict  
    
    
class multiplicativeNoise(AbstractTransform):
    """
    variables in DIKU_3D_augmentation_params:
        do_multiplicativeNoise
        multiplicativeNoise_p_per_sample
        multiplicativeNoise_mean
        multiplicativeNoise_sigma 
    """
    def __init__(self, data_key="data", p_per_sample=1,
                 mean = 0,
                 sigma = 1e-4 * np.random.uniform()):
        self.data_key = data_key
        self.p_per_sample = p_per_sample
        self.mean = mean
        self.sigma = sigma
    
    def __multiplicativeNoise__(self, imageVolume, mean, sigma):
        # J = I + I*n
        gauss = np.random.normal(mean, sigma, imageVolume.shape)
        return imageVolume + imageVolume*gauss
    
    def __call__(self, **data_dict):
        assert (len(data_dict[self.data_key].shape) == 5 or len(data_dict[self.data_key].shape) == 4) and data_dict[self.data_key].shape[1] == 1, f"Incorrect data size or shape.\
            \nShould be (b, c, x, y, z) or (b, c, x, y) and is: {data_dict[self.data_key].shape}"
        for sample in range(len(data_dict[self.data_key])):
            if np.random.uniform() < self.p_per_sample:
                data_dict[self.data_key][sample][0] = self.__multiplicativeNoise__(data_dict[self.data_key][sample][0], 
                self.mean, self.sigma)
        return data_dict
