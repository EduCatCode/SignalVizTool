"""
Advanced Signal Analysis Module

Provides advanced signal processing capabilities including:
- Fast Fourier Transform (FFT) analysis
- Wavelet Transform analysis
- Short-Time Fourier Transform (STFT)
- Power Spectral Density (PSD)
- Spectrogram generation

Author: EduCatCode - Engineering Department
Version: 2.1.0
License: MIT
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.fft import fft, fftfreq, rfft, rfftfreq
import pywt
from typing import Tuple, List, Optional, Dict
import logging


class FFTAnalyzer:
    """Advanced FFT analysis for frequency domain insights."""

    def __init__(self, sampling_rate: float, logger: Optional[logging.Logger] = None):
        """
        Initialize FFT analyzer.

        Args:
            sampling_rate: Sampling rate in Hz
            logger: Optional logger instance
        """
        self.sampling_rate = sampling_rate
        self.logger = logger or logging.getLogger(__name__)

    def compute_fft(self, signal_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Fast Fourier Transform.

        Args:
            signal_data: Input signal array

        Returns:
            Tuple of (frequencies, magnitudes)
        """
        n = len(signal_data)

        # Compute FFT
        fft_vals = rfft(signal_data)
        fft_magnitude = np.abs(fft_vals)
        freqs = rfftfreq(n, 1/self.sampling_rate)

        self.logger.info(f"FFT computed: {len(freqs)} frequency bins")
        return freqs, fft_magnitude

    def compute_power_spectrum(self, signal_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute power spectrum (squared magnitude).

        Args:
            signal_data: Input signal array

        Returns:
            Tuple of (frequencies, power)
        """
        freqs, magnitude = self.compute_fft(signal_data)
        power = magnitude ** 2
        return freqs, power

    def compute_psd(self, signal_data: np.ndarray,
                   nperseg: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Power Spectral Density using Welch's method.

        Args:
            signal_data: Input signal array
            nperseg: Length of each segment (default: 256)

        Returns:
            Tuple of (frequencies, psd)
        """
        if nperseg is None:
            nperseg = min(256, len(signal_data) // 4)

        freqs, psd = signal.welch(
            signal_data,
            fs=self.sampling_rate,
            nperseg=nperseg,
            scaling='density'
        )

        self.logger.info(f"PSD computed using Welch's method (nperseg={nperseg})")
        return freqs, psd

    def find_dominant_frequencies(self, signal_data: np.ndarray,
                                 n_peaks: int = 5) -> pd.DataFrame:
        """
        Find dominant frequencies in the signal.

        Args:
            signal_data: Input signal array
            n_peaks: Number of top peaks to return

        Returns:
            DataFrame with frequency and magnitude of peaks
        """
        freqs, magnitude = self.compute_fft(signal_data)

        # Find peaks
        peaks, properties = signal.find_peaks(magnitude, height=0)

        if len(peaks) == 0:
            self.logger.warning("No peaks found in FFT")
            return pd.DataFrame(columns=['Frequency (Hz)', 'Magnitude'])

        # Sort by magnitude
        peak_magnitudes = magnitude[peaks]
        sorted_indices = np.argsort(peak_magnitudes)[::-1][:n_peaks]

        dominant_freqs = freqs[peaks[sorted_indices]]
        dominant_mags = peak_magnitudes[sorted_indices]

        results = pd.DataFrame({
            'Frequency (Hz)': dominant_freqs,
            'Magnitude': dominant_mags
        })

        self.logger.info(f"Found {len(results)} dominant frequencies")
        return results

    def compute_frequency_bands_energy(self, signal_data: np.ndarray,
                                      bands: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
        """
        Compute energy in specified frequency bands.

        Args:
            signal_data: Input signal array
            bands: Dictionary of band_name: (low_freq, high_freq)

        Returns:
            DataFrame with energy in each band

        Example:
            bands = {
                'Low': (0, 100),
                'Mid': (100, 1000),
                'High': (1000, 5000)
            }
        """
        freqs, magnitude = self.compute_fft(signal_data)

        results = []
        for band_name, (low_freq, high_freq) in bands.items():
            mask = (freqs >= low_freq) & (freqs <= high_freq)
            energy = np.sum(magnitude[mask] ** 2)

            results.append({
                'Band': band_name,
                'Frequency Range': f"{low_freq}-{high_freq} Hz",
                'Energy': energy
            })

        return pd.DataFrame(results)


class WaveletAnalyzer:
    """Wavelet transform analysis for time-frequency localization."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize wavelet analyzer.

        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.available_wavelets = pywt.wavelist(kind='discrete')

    def compute_cwt(self, signal_data: np.ndarray,
                   scales: Optional[np.ndarray] = None,
                   wavelet: str = 'morl') -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Continuous Wavelet Transform.

        Args:
            signal_data: Input signal array
            scales: Array of scales (default: 1 to 128)
            wavelet: Wavelet name (default: 'morl' - Morlet)

        Returns:
            Tuple of (coefficients, frequencies)
        """
        if scales is None:
            scales = np.arange(1, 128)

        # Compute CWT
        coefficients, frequencies = pywt.cwt(signal_data, scales, wavelet)

        self.logger.info(f"CWT computed using {wavelet} wavelet, {len(scales)} scales")
        return coefficients, frequencies

    def compute_dwt(self, signal_data: np.ndarray,
                   wavelet: str = 'db4',
                   level: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Compute Discrete Wavelet Transform.

        Args:
            signal_data: Input signal array
            wavelet: Wavelet name (default: 'db4' - Daubechies 4)
            level: Decomposition level (default: auto)

        Returns:
            Dictionary with approximation and detail coefficients
        """
        if level is None:
            level = pywt.dwt_max_level(len(signal_data), wavelet)

        # Compute DWT
        coeffs = pywt.wavedec(signal_data, wavelet, level=level)

        results = {
            'approximation': coeffs[0],
        }

        for i, detail in enumerate(coeffs[1:], 1):
            results[f'detail_{i}'] = detail

        self.logger.info(f"DWT computed using {wavelet} wavelet, level {level}")
        return results

    def denoise_signal(self, signal_data: np.ndarray,
                      wavelet: str = 'db4',
                      level: Optional[int] = None,
                      threshold_mode: str = 'soft') -> np.ndarray:
        """
        Denoise signal using wavelet thresholding.

        Args:
            signal_data: Input noisy signal
            wavelet: Wavelet name
            level: Decomposition level
            threshold_mode: 'soft' or 'hard' thresholding

        Returns:
            Denoised signal
        """
        if level is None:
            level = pywt.dwt_max_level(len(signal_data), wavelet)

        # Decompose
        coeffs = pywt.wavedec(signal_data, wavelet, level=level)

        # Estimate noise sigma using MAD (Median Absolute Deviation)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745

        # Universal threshold
        threshold = sigma * np.sqrt(2 * np.log(len(signal_data)))

        # Threshold detail coefficients
        coeffs_thresh = [coeffs[0]]  # Keep approximation
        for detail in coeffs[1:]:
            if threshold_mode == 'soft':
                thresh_detail = pywt.threshold(detail, threshold, mode='soft')
            else:
                thresh_detail = pywt.threshold(detail, threshold, mode='hard')
            coeffs_thresh.append(thresh_detail)

        # Reconstruct
        denoised = pywt.waverec(coeffs_thresh, wavelet)

        # Handle length mismatch
        if len(denoised) > len(signal_data):
            denoised = denoised[:len(signal_data)]

        self.logger.info(f"Signal denoised using {wavelet} wavelet (threshold={threshold:.4f})")
        return denoised

    def extract_wavelet_features(self, signal_data: np.ndarray,
                                 wavelet: str = 'db4',
                                 level: int = 4) -> pd.DataFrame:
        """
        Extract statistical features from wavelet coefficients.

        Args:
            signal_data: Input signal array
            wavelet: Wavelet name
            level: Decomposition level

        Returns:
            DataFrame with wavelet-based features
        """
        coeffs_dict = self.compute_dwt(signal_data, wavelet, level)

        features = []
        for name, coeff in coeffs_dict.items():
            features.append({
                'Component': name,
                'Mean': np.mean(coeff),
                'Std': np.std(coeff),
                'Energy': np.sum(coeff ** 2),
                'Max': np.max(np.abs(coeff))
            })

        return pd.DataFrame(features)


class STFTAnalyzer:
    """Short-Time Fourier Transform for time-frequency analysis."""

    def __init__(self, sampling_rate: float, logger: Optional[logging.Logger] = None):
        """
        Initialize STFT analyzer.

        Args:
            sampling_rate: Sampling rate in Hz
            logger: Optional logger instance
        """
        self.sampling_rate = sampling_rate
        self.logger = logger or logging.getLogger(__name__)

    def compute_stft(self, signal_data: np.ndarray,
                    window: str = 'hann',
                    nperseg: Optional[int] = None,
                    noverlap: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute Short-Time Fourier Transform.

        Args:
            signal_data: Input signal array
            window: Window function ('hann', 'hamming', 'blackman')
            nperseg: Length of each segment (default: 256)
            noverlap: Number of overlapping points (default: nperseg//2)

        Returns:
            Tuple of (frequencies, times, STFT_magnitude)
        """
        if nperseg is None:
            nperseg = min(256, len(signal_data) // 4)
        if noverlap is None:
            noverlap = nperseg // 2

        # Compute STFT
        freqs, times, Zxx = signal.stft(
            signal_data,
            fs=self.sampling_rate,
            window=window,
            nperseg=nperseg,
            noverlap=noverlap
        )

        magnitude = np.abs(Zxx)

        self.logger.info(f"STFT computed: {len(freqs)} freqs x {len(times)} time bins")
        return freqs, times, magnitude

    def compute_spectrogram(self, signal_data: np.ndarray,
                           window: str = 'hann',
                           nperseg: Optional[int] = None,
                           noverlap: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute spectrogram (power spectral density over time).

        Args:
            signal_data: Input signal array
            window: Window function
            nperseg: Length of each segment
            noverlap: Number of overlapping points

        Returns:
            Tuple of (frequencies, times, spectrogram)
        """
        if nperseg is None:
            nperseg = min(256, len(signal_data) // 4)
        if noverlap is None:
            noverlap = nperseg // 2

        freqs, times, Sxx = signal.spectrogram(
            signal_data,
            fs=self.sampling_rate,
            window=window,
            nperseg=nperseg,
            noverlap=noverlap
        )

        self.logger.info(f"Spectrogram computed: {len(freqs)} freqs x {len(times)} time bins")
        return freqs, times, Sxx

    def extract_spectral_features_over_time(self, signal_data: np.ndarray) -> pd.DataFrame:
        """
        Extract spectral features over time windows.

        Args:
            signal_data: Input signal array

        Returns:
            DataFrame with time-varying spectral features
        """
        freqs, times, Sxx = self.compute_spectrogram(signal_data)

        features = []
        for i, t in enumerate(times):
            spectrum = Sxx[:, i]

            # Spectral centroid
            centroid = np.sum(freqs * spectrum) / (np.sum(spectrum) + 1e-10)

            # Spectral spread
            spread = np.sqrt(np.sum(((freqs - centroid) ** 2) * spectrum) / (np.sum(spectrum) + 1e-10))

            # Spectral entropy
            normalized_spectrum = spectrum / (np.sum(spectrum) + 1e-10)
            entropy = -np.sum(normalized_spectrum * np.log2(normalized_spectrum + 1e-10))

            # Spectral rolloff (95% energy)
            cumsum = np.cumsum(spectrum)
            rolloff_idx = np.where(cumsum >= 0.95 * cumsum[-1])[0]
            rolloff = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else freqs[-1]

            features.append({
                'Time (s)': t,
                'Spectral Centroid (Hz)': centroid,
                'Spectral Spread (Hz)': spread,
                'Spectral Entropy': entropy,
                'Spectral Rolloff (Hz)': rolloff
            })

        return pd.DataFrame(features)


class FilterDesigner:
    """Design and apply various digital filters."""

    def __init__(self, sampling_rate: float, logger: Optional[logging.Logger] = None):
        """
        Initialize filter designer.

        Args:
            sampling_rate: Sampling rate in Hz
            logger: Optional logger instance
        """
        self.sampling_rate = sampling_rate
        self.logger = logger or logging.getLogger(__name__)

    def design_butterworth_lowpass(self, cutoff_freq: float,
                                   order: int = 4) -> Tuple[np.ndarray, np.ndarray]:
        """
        Design Butterworth lowpass filter.

        Args:
            cutoff_freq: Cutoff frequency in Hz
            order: Filter order

        Returns:
            Tuple of (b, a) filter coefficients
        """
        nyquist = self.sampling_rate / 2
        normalized_cutoff = cutoff_freq / nyquist

        b, a = signal.butter(order, normalized_cutoff, btype='low', analog=False)

        self.logger.info(f"Butterworth lowpass filter designed (cutoff={cutoff_freq}Hz, order={order})")
        return b, a

    def design_butterworth_highpass(self, cutoff_freq: float,
                                    order: int = 4) -> Tuple[np.ndarray, np.ndarray]:
        """
        Design Butterworth highpass filter.

        Args:
            cutoff_freq: Cutoff frequency in Hz
            order: Filter order

        Returns:
            Tuple of (b, a) filter coefficients
        """
        nyquist = self.sampling_rate / 2
        normalized_cutoff = cutoff_freq / nyquist

        b, a = signal.butter(order, normalized_cutoff, btype='high', analog=False)

        self.logger.info(f"Butterworth highpass filter designed (cutoff={cutoff_freq}Hz, order={order})")
        return b, a

    def design_butterworth_bandpass(self, low_freq: float, high_freq: float,
                                    order: int = 4) -> Tuple[np.ndarray, np.ndarray]:
        """
        Design Butterworth bandpass filter.

        Args:
            low_freq: Low cutoff frequency in Hz
            high_freq: High cutoff frequency in Hz
            order: Filter order

        Returns:
            Tuple of (b, a) filter coefficients
        """
        nyquist = self.sampling_rate / 2
        low_normalized = low_freq / nyquist
        high_normalized = high_freq / nyquist

        b, a = signal.butter(order, [low_normalized, high_normalized], btype='band', analog=False)

        self.logger.info(f"Butterworth bandpass filter designed ({low_freq}-{high_freq}Hz, order={order})")
        return b, a

    def apply_filter(self, signal_data: np.ndarray,
                    b: np.ndarray, a: np.ndarray) -> np.ndarray:
        """
        Apply filter to signal.

        Args:
            signal_data: Input signal
            b, a: Filter coefficients

        Returns:
            Filtered signal
        """
        # Use filtfilt for zero-phase filtering
        filtered = signal.filtfilt(b, a, signal_data)

        self.logger.info(f"Filter applied to signal (length={len(signal_data)})")
        return filtered

    def design_and_apply_lowpass(self, signal_data: np.ndarray,
                                 cutoff_freq: float,
                                 order: int = 4) -> np.ndarray:
        """
        Design and apply lowpass filter.

        Args:
            signal_data: Input signal
            cutoff_freq: Cutoff frequency in Hz
            order: Filter order

        Returns:
            Filtered signal
        """
        b, a = self.design_butterworth_lowpass(cutoff_freq, order)
        return self.apply_filter(signal_data, b, a)

    def design_and_apply_highpass(self, signal_data: np.ndarray,
                                  cutoff_freq: float,
                                  order: int = 4) -> np.ndarray:
        """
        Design and apply highpass filter.

        Args:
            signal_data: Input signal
            cutoff_freq: Cutoff frequency in Hz
            order: Filter order

        Returns:
            Filtered signal
        """
        b, a = self.design_butterworth_highpass(cutoff_freq, order)
        return self.apply_filter(signal_data, b, a)

    def design_and_apply_bandpass(self, signal_data: np.ndarray,
                                  low_freq: float, high_freq: float,
                                  order: int = 4) -> np.ndarray:
        """
        Design and apply bandpass filter.

        Args:
            signal_data: Input signal
            low_freq: Low cutoff frequency in Hz
            high_freq: High cutoff frequency in Hz
            order: Filter order

        Returns:
            Filtered signal
        """
        b, a = self.design_butterworth_bandpass(low_freq, high_freq, order)
        return self.apply_filter(signal_data, b, a)
