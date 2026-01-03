"""
Signal Feature Extraction Module

This module provides comprehensive signal processing capabilities for extracting
time-domain and frequency-domain features from 1D signals.

Features:
    Time Domain (13 features):
        - Statistical: Mean, Standard Deviation, Variance, RMS, Peak
        - Shape: Skewness, Kurtosis
        - Factors: Crest, Margin, Shape, Impulse, A-factor, B-factor

    Frequency Domain (3 features):
        - Dominant Frequency
        - Band Energy
        - Spectral Centroid

Author: EduCatCode
License: MIT
"""

import numpy as np
from scipy.stats import skew, kurtosis
from typing import List, Tuple, Optional
import warnings


class TimeFeatureExtractor:
    """Extract time-domain features from signals using sliding window analysis."""

    def __init__(self, frame_size: int, hop_length: int):
        """
        Initialize the time domain feature extractor.

        Args:
            frame_size: Number of samples per frame (window size)
            hop_length: Number of samples to advance between frames

        Raises:
            ValueError: If frame_size or hop_length <= 0
        """
        if frame_size <= 0 or hop_length <= 0:
            raise ValueError("frame_size and hop_length must be positive integers")

        self.frame_size = frame_size
        self.hop_length = hop_length
        self._epsilon = 1e-10  # Small value to prevent division by zero

    def get_mean_acceleration(self, signal: np.ndarray) -> np.ndarray:
        """
        Calculate mean acceleration for each frame.

        Formula: mean = (1/N) * Σ x[i]

        Args:
            signal: Input 1D signal array

        Returns:
            Array of mean values for each frame
        """
        means = []
        for i in range(0, len(signal), self.hop_length):
            frame = signal[i:i + self.frame_size]
            if len(frame) < self.frame_size:
                break
            means.append(np.mean(frame))
        return np.array(means)

    def get_std(self, signal: np.ndarray) -> np.ndarray:
        """
        Calculate standard deviation for each frame.

        Formula: std = sqrt((1/(N-1)) * Σ(x[i] - mean)^2)

        Args:
            signal: Input 1D signal array

        Returns:
            Array of standard deviation values for each frame
        """
        stds = []
        for i in range(0, len(signal), self.hop_length):
            frame = signal[i:i + self.frame_size]
            if len(frame) < self.frame_size:
                break
            stds.append(np.std(frame, ddof=1))
        return np.array(stds)

    def get_variance(self, signal: np.ndarray) -> np.ndarray:
        """
        Calculate variance for each frame.

        Formula: var = (1/(N-1)) * Σ(x[i] - mean)^2

        Note: Previous implementation was incorrect. Now uses standard statistical variance.

        Args:
            signal: Input 1D signal array

        Returns:
            Array of variance values for each frame
        """
        variances = []
        for i in range(0, len(signal), self.hop_length):
            frame = signal[i:i + self.frame_size]
            if len(frame) < self.frame_size:
                break
            variances.append(np.var(frame, ddof=1))
        return np.array(variances)

    def get_rms_acceleration(self, signal: np.ndarray) -> np.ndarray:
        """
        Calculate Root Mean Square (RMS) acceleration for each frame.

        Formula: RMS = sqrt((1/N) * Σ x[i]^2)

        Args:
            signal: Input 1D signal array

        Returns:
            Array of RMS values for each frame
        """
        rms_values = []
        for i in range(0, len(signal), self.hop_length):
            frame = signal[i:i + self.frame_size]
            if len(frame) < self.frame_size:
                break
            rms_values.append(np.sqrt(np.mean(frame**2)))
        return np.array(rms_values)

    def get_peak_acceleration(self, signal: np.ndarray) -> np.ndarray:
        """
        Calculate peak (maximum absolute value) for each frame.

        Args:
            signal: Input 1D signal array

        Returns:
            Array of peak values for each frame
        """
        peaks = []
        for i in range(0, len(signal), self.hop_length):
            frame = signal[i:i + self.frame_size]
            if len(frame) < self.frame_size:
                break
            peaks.append(np.max(np.abs(frame)))
        return np.array(peaks)

    def get_skewness(self, signal: np.ndarray) -> np.ndarray:
        """
        Calculate skewness for each frame.

        Skewness measures the asymmetry of the distribution.

        Args:
            signal: Input 1D signal array

        Returns:
            Array of skewness values for each frame
        """
        skewness_values = []
        for i in range(0, len(signal), self.hop_length):
            frame = signal[i:i + self.frame_size]
            if len(frame) < self.frame_size:
                break
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                skewness_values.append(skew(frame))
        return np.array(skewness_values)

    def get_kurtosis(self, signal: np.ndarray) -> np.ndarray:
        """
        Calculate kurtosis for each frame.

        Kurtosis measures the "tailedness" of the distribution.

        Args:
            signal: Input 1D signal array

        Returns:
            Array of kurtosis values for each frame
        """
        kurtosis_values = []
        for i in range(0, len(signal), self.hop_length):
            frame = signal[i:i + self.frame_size]
            if len(frame) < self.frame_size:
                break
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                kurtosis_values.append(kurtosis(frame))
        return np.array(kurtosis_values)

    def get_crest_factor(self, signal: np.ndarray) -> np.ndarray:
        """
        Calculate crest factor for each frame.

        Formula: Crest Factor = Peak / RMS

        Note: Previous implementation incorrectly divided by skewness.
        Crest factor indicates the ratio of peak to RMS, useful for detecting impulsive events.

        Args:
            signal: Input 1D signal array

        Returns:
            Array of crest factor values for each frame
        """
        crest_factors = []
        for i in range(0, len(signal), self.hop_length):
            frame = signal[i:i + self.frame_size]
            if len(frame) < self.frame_size:
                break
            peak = np.max(np.abs(frame))
            rms = np.sqrt(np.mean(frame**2))
            crest_factors.append(peak / (rms + self._epsilon))
        return np.array(crest_factors)

    def get_margin_factor(self, signal: np.ndarray) -> np.ndarray:
        """
        Calculate margin factor for each frame.

        Formula: Margin Factor = Peak / (mean(sqrt(|x|)))^2

        Args:
            signal: Input 1D signal array

        Returns:
            Array of margin factor values for each frame
        """
        margin_factors = []
        for i in range(0, len(signal), self.hop_length):
            frame = signal[i:i + self.frame_size]
            if len(frame) < self.frame_size:
                break
            peak = np.max(np.abs(frame))
            denominator = (np.mean(np.sqrt(np.abs(frame))))**2
            margin_factors.append(peak / (denominator + self._epsilon))
        return np.array(margin_factors)

    def get_shape_factor(self, signal: np.ndarray) -> np.ndarray:
        """
        Calculate shape factor for each frame.

        Formula: Shape Factor = RMS / mean(|x|)

        Args:
            signal: Input 1D signal array

        Returns:
            Array of shape factor values for each frame
        """
        shape_factors = []
        for i in range(0, len(signal), self.hop_length):
            frame = signal[i:i + self.frame_size]
            if len(frame) < self.frame_size:
                break
            rms = np.sqrt(np.mean(frame**2))
            mean_abs = np.mean(np.abs(frame))
            shape_factors.append(rms / (mean_abs + self._epsilon))
        return np.array(shape_factors)

    def get_impulse_factor(self, signal: np.ndarray) -> np.ndarray:
        """
        Calculate impulse factor for each frame.

        Formula: Impulse Factor = Peak / mean(|x|)

        Args:
            signal: Input 1D signal array

        Returns:
            Array of impulse factor values for each frame
        """
        impulse_factors = []
        for i in range(0, len(signal), self.hop_length):
            frame = signal[i:i + self.frame_size]
            if len(frame) < self.frame_size:
                break
            peak = np.max(np.abs(frame))
            mean_abs = np.mean(np.abs(frame))
            impulse_factors.append(peak / (mean_abs + self._epsilon))
        return np.array(impulse_factors)

    def get_a_factor(self, signal: np.ndarray) -> np.ndarray:
        """
        Calculate A-factor for each frame.

        Formula: A-factor = Peak / (std * variance)

        Custom diagnostic factor for signal analysis.

        Args:
            signal: Input 1D signal array

        Returns:
            Array of A-factor values for each frame
        """
        a_factors = []
        for i in range(0, len(signal), self.hop_length):
            frame = signal[i:i + self.frame_size]
            if len(frame) < self.frame_size:
                break
            peak = np.max(np.abs(frame))
            std_val = np.std(frame, ddof=1)
            var_val = np.var(frame, ddof=1)
            a_factors.append(peak / ((std_val + self._epsilon) * (var_val + self._epsilon)))
        return np.array(a_factors)

    def get_b_factor(self, signal: np.ndarray) -> np.ndarray:
        """
        Calculate B-factor for each frame.

        Formula: B-factor = (kurtosis * crest_factor) / std

        Complex custom diagnostic factor combining multiple statistical properties.

        Args:
            signal: Input 1D signal array

        Returns:
            Array of B-factor values for each frame
        """
        b_factors = []
        for i in range(0, len(signal), self.hop_length):
            frame = signal[i:i + self.frame_size]
            if len(frame) < self.frame_size:
                break
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                kurt = kurtosis(frame)
            peak = np.max(np.abs(frame))
            rms = np.sqrt(np.mean(frame**2))
            crest = peak / (rms + self._epsilon)
            std_val = np.std(frame, ddof=1)
            b_factors.append((kurt * crest) / (std_val + self._epsilon))
        return np.array(b_factors)

    def extract_all_features(self, signal: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """
        Extract all time-domain features from the signal.

        Args:
            signal: Input 1D signal array

        Returns:
            Tuple of (feature_matrix, feature_names)
            - feature_matrix: 2D array where each row is a feature
            - feature_names: List of feature names
        """
        features = []
        feature_names = [
            'Peak', 'RMS', 'Crest Factor', 'Std', 'Variance',
            'Skewness', 'Kurtosis', 'Shape Factor', 'Impulse Factor',
            'Margin Factor', 'Mean', 'A Factor', 'B Factor'
        ]

        features.append(self.get_peak_acceleration(signal))
        features.append(self.get_rms_acceleration(signal))
        features.append(self.get_crest_factor(signal))
        features.append(self.get_std(signal))
        features.append(self.get_variance(signal))
        features.append(self.get_skewness(signal))
        features.append(self.get_kurtosis(signal))
        features.append(self.get_shape_factor(signal))
        features.append(self.get_impulse_factor(signal))
        features.append(self.get_margin_factor(signal))
        features.append(self.get_mean_acceleration(signal))
        features.append(self.get_a_factor(signal))
        features.append(self.get_b_factor(signal))

        return np.array(features), feature_names


class FrequencyFeatureExtractor:
    """Extract frequency-domain features from signals using FFT analysis."""

    def __init__(self, frame_size: int, hop_length: int, sampling_rate: float):
        """
        Initialize the frequency domain feature extractor.

        Args:
            frame_size: Number of samples per frame (window size)
            hop_length: Number of samples to advance between frames
            sampling_rate: Sampling rate in Hz

        Raises:
            ValueError: If parameters are invalid
        """
        if frame_size <= 0 or hop_length <= 0:
            raise ValueError("frame_size and hop_length must be positive integers")
        if sampling_rate <= 0:
            raise ValueError("sampling_rate must be positive")

        self.frame_size = frame_size
        self.hop_length = hop_length
        self.sampling_rate = sampling_rate
        self._epsilon = 1e-10

    def get_dominant_frequency(self, signal: np.ndarray) -> np.ndarray:
        """
        Calculate the dominant frequency for each frame.

        The dominant frequency is the frequency with the maximum magnitude in the FFT.

        Args:
            signal: Input 1D signal array

        Returns:
            Array of dominant frequencies for each frame (in Hz)
        """
        dom_freqs = []
        for i in range(0, len(signal), self.hop_length):
            frame = signal[i:i + self.frame_size]
            if len(frame) < self.frame_size:
                break

            # Compute FFT
            fft_vals = np.fft.fft(frame)
            fft_magnitude = np.abs(fft_vals)[:len(frame) // 2]
            freqs = np.fft.fftfreq(len(frame), 1 / self.sampling_rate)[:len(frame) // 2]

            # Find dominant frequency
            dominant_freq = freqs[np.argmax(fft_magnitude)]
            dom_freqs.append(dominant_freq)

        return np.array(dom_freqs)

    def get_band_energy(self, signal: np.ndarray,
                       freq_band: Tuple[float, float] = (0, 5000)) -> np.ndarray:
        """
        Calculate energy in a specific frequency band for each frame.

        Args:
            signal: Input 1D signal array
            freq_band: Tuple of (low_freq, high_freq) in Hz

        Returns:
            Array of band energy values for each frame
        """
        band_energies = []
        low_freq, high_freq = freq_band

        for i in range(0, len(signal), self.hop_length):
            frame = signal[i:i + self.frame_size]
            if len(frame) < self.frame_size:
                break

            # Compute FFT
            fft_vals = np.fft.fft(frame)
            fft_magnitude = np.abs(fft_vals)[:len(frame) // 2]
            freqs = np.fft.fftfreq(len(frame), 1 / self.sampling_rate)[:len(frame) // 2]

            # Calculate energy in frequency band
            band_mask = (freqs >= low_freq) & (freqs <= high_freq)
            band_energy = np.sum(fft_magnitude[band_mask])
            band_energies.append(band_energy)

        return np.array(band_energies)

    def get_spectral_centroid(self, signal: np.ndarray) -> np.ndarray:
        """
        Calculate the spectral centroid for each frame.

        The spectral centroid is the "center of mass" of the spectrum,
        indicating where the "center" of the sound is located.

        Args:
            signal: Input 1D signal array

        Returns:
            Array of spectral centroid values for each frame (in Hz)
        """
        centroids = []
        for i in range(0, len(signal), self.hop_length):
            frame = signal[i:i + self.frame_size]
            if len(frame) < self.frame_size:
                break

            # Compute FFT
            fft_vals = np.fft.fft(frame)
            fft_magnitude = np.abs(fft_vals)[:len(frame) // 2]
            freqs = np.fft.fftfreq(len(frame), 1 / self.sampling_rate)[:len(frame) // 2]

            # Calculate spectral centroid
            centroid = np.sum(freqs * fft_magnitude) / (np.sum(fft_magnitude) + self._epsilon)
            centroids.append(centroid)

        return np.array(centroids)

    def extract_all_features(self, signal: np.ndarray,
                           freq_band: Tuple[float, float] = (0, 5000)) -> Tuple[np.ndarray, List[str]]:
        """
        Extract all frequency-domain features from the signal.

        Args:
            signal: Input 1D signal array
            freq_band: Frequency band for energy calculation (low_freq, high_freq)

        Returns:
            Tuple of (feature_matrix, feature_names)
            - feature_matrix: 2D array where each row is a feature
            - feature_names: List of feature names
        """
        features = []
        feature_names = ['Dominant Frequency', 'Band Energy', 'Spectral Centroid']

        features.append(self.get_dominant_frequency(signal))
        features.append(self.get_band_energy(signal, freq_band))
        features.append(self.get_spectral_centroid(signal))

        return np.array(features), feature_names
