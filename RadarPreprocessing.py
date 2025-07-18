# -*- coding: utf-8 -*-
"""
Created on Sat Jul 12 14:42:39 2025

@author: Hp
"""
import numpy as np
from scipy.ndimage import zoom
from scipy.io import loadmat

class RadarPreprocessing:
    def __init__(self, cube_shape=(32, 32, 16)):
        self.cube_shape = cube_shape

    def load_data(self, radar_path):
        """
        Load the radar data from a .mat file.
        """
        try:
            mat = loadmat(radar_path)
            data_key = next(k for k in mat.keys() if not k.startswith('__'))  # Extract the key
            adc_data = mat[data_key]
            return adc_data
        except Exception as e:
            print(f" Failed to load {radar_path}: {e}")
            return None

    def handle_4d_data(self, adc_data):
        """
        If data is 4D (e.g., with a channel dimension), reduce it by averaging across the last axis (channel dimension).
        """
        if adc_data.ndim == 4:
            # Averaging over the last dimension (channels) or you can select a specific channel if needed
            adc_data = np.mean(adc_data, axis=-1)  # Averaging along the 12th dimension
        elif adc_data.ndim != 3:
            raise ValueError(f"Expected 3D or 4D input, but got shape {adc_data.shape}")
        return adc_data

    def apply_fft(self, adc_data):
        """
        Apply FFT along Range, Doppler, and Azimuth dimensions to extract frequency components.
        """
        if adc_data is not None and adc_data.ndim == 3:
            # Apply FFT on Range, Velocity (Doppler), and Azimuth axes
            fft_data = np.fft.fftn(adc_data, axes=(0, 1, 2))  # Apply FFT along all axes
            return np.abs(fft_data)  # Magnitude of FFT coefficients
        else:
            raise ValueError(f"Expected 3D input, but got shape {adc_data.shape if adc_data is not None else 'None'}")

    def resize_data(self, adc_data):
        """
        Resize the data to the fixed cube size specified during initialization.
        """
        if adc_data is not None:
            zoom_factors = tuple(n / o for n, o in zip(self.cube_shape, adc_data.shape))
            return zoom(adc_data, zoom_factors, order=1)
        else:
            raise ValueError("Data to resize cannot be None")

    def normalize_data(self, adc_data):
        """
        Normalize the radar data to the range [0, 1].
        """
        min_val, max_val = adc_data.min(), adc_data.max()
        if max_val - min_val > 1e-6:
            adc_data = (adc_data - min_val) / (max_val - min_val)
        else:
            adc_data = np.zeros_like(adc_data)  # Avoid division by zero
        return adc_data

    def preprocess(self, radar_path):
        """
        Perform full preprocessing: load, FFT, resize, and normalize.
        """
        adc_data = self.load_data(radar_path)
        if adc_data is not None:
            adc_data = self.handle_4d_data(adc_data)  # Handle 4D data
            adc_data = self.apply_fft(adc_data)  # Apply FFT
            adc_data = self.resize_data(adc_data)  # Resize
            adc_data = self.normalize_data(adc_data)  # Normalize
            return adc_data
        else:
            return None
