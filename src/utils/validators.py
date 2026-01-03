"""
Data Validation Utilities

Provides validation functions for user input and data processing parameters.

Author: EduCatCode
License: MIT
"""

from typing import Tuple, Optional
import numpy as np


class InputValidator:
    """Validate user input and processing parameters."""

    @staticmethod
    def validate_frame_size(frame_size: int,
                           min_value: int = 1,
                           max_value: int = 200000) -> Tuple[bool, str]:
        """
        Validate frame size parameter.

        Args:
            frame_size: Frame size to validate
            min_value: Minimum allowed value
            max_value: Maximum allowed value

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(frame_size, int):
            return False, "Frame size must be an integer"

        if frame_size < min_value:
            return False, f"Frame size must be at least {min_value}"

        if frame_size > max_value:
            return False, f"Frame size must not exceed {max_value}"

        return True, ""

    @staticmethod
    def validate_hop_length(hop_length: int,
                           frame_size: Optional[int] = None,
                           min_value: int = 1,
                           max_value: int = 200000) -> Tuple[bool, str]:
        """
        Validate hop length parameter.

        Args:
            hop_length: Hop length to validate
            frame_size: Optional frame size for additional validation
            min_value: Minimum allowed value
            max_value: Maximum allowed value

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(hop_length, int):
            return False, "Hop length must be an integer"

        if hop_length < min_value:
            return False, f"Hop length must be at least {min_value}"

        if hop_length > max_value:
            return False, f"Hop length must not exceed {max_value}"

        if frame_size is not None and hop_length > frame_size:
            return False, f"Hop length ({hop_length}) should not exceed frame size ({frame_size})"

        return True, ""

    @staticmethod
    def validate_sampling_rate(sampling_rate: float,
                              min_value: float = 1.0,
                              max_value: float = 1000000.0) -> Tuple[bool, str]:
        """
        Validate sampling rate parameter.

        Args:
            sampling_rate: Sampling rate to validate
            min_value: Minimum allowed value
            max_value: Maximum allowed value

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(sampling_rate, (int, float)):
            return False, "Sampling rate must be a number"

        if sampling_rate <= 0:
            return False, "Sampling rate must be positive"

        if sampling_rate < min_value:
            return False, f"Sampling rate must be at least {min_value} Hz"

        if sampling_rate > max_value:
            return False, f"Sampling rate must not exceed {max_value} Hz"

        return True, ""

    @staticmethod
    def validate_frequency_band(freq_band: Tuple[float, float],
                               sampling_rate: Optional[float] = None) -> Tuple[bool, str]:
        """
        Validate frequency band parameter.

        Args:
            freq_band: Tuple of (low_freq, high_freq)
            sampling_rate: Optional sampling rate for Nyquist check

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(freq_band, (tuple, list)) or len(freq_band) != 2:
            return False, "Frequency band must be a tuple of (low_freq, high_freq)"

        low_freq, high_freq = freq_band

        if not isinstance(low_freq, (int, float)) or not isinstance(high_freq, (int, float)):
            return False, "Frequency values must be numbers"

        if low_freq < 0:
            return False, "Low frequency must be non-negative"

        if high_freq <= low_freq:
            return False, "High frequency must be greater than low frequency"

        if sampling_rate is not None:
            nyquist_freq = sampling_rate / 2
            if high_freq > nyquist_freq:
                return False, f"High frequency ({high_freq} Hz) exceeds Nyquist frequency ({nyquist_freq} Hz)"

        return True, ""

    @staticmethod
    def validate_file_paths(file_paths: list) -> Tuple[bool, str]:
        """
        Validate file path list.

        Args:
            file_paths: List of file paths

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not file_paths:
            return False, "No files selected"

        if not isinstance(file_paths, (list, tuple)):
            return False, "File paths must be a list"

        return True, ""

    @staticmethod
    def validate_column_selection(selected_columns: list,
                                 available_columns: list) -> Tuple[bool, str]:
        """
        Validate column selection.

        Args:
            selected_columns: List of selected column names
            available_columns: List of available column names

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not selected_columns:
            return False, "No columns selected"

        missing_columns = [col for col in selected_columns if col not in available_columns]

        if missing_columns:
            return False, f"Selected columns not found: {missing_columns}"

        return True, ""

    @staticmethod
    def validate_output_path(output_path: str) -> Tuple[bool, str]:
        """
        Validate output file path.

        Args:
            output_path: Output file path

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not output_path:
            return False, "No output path specified"

        if not isinstance(output_path, str):
            return False, "Output path must be a string"

        # Check for invalid characters (basic check)
        invalid_chars = '<>:"|?*'
        if any(char in output_path for char in invalid_chars):
            return False, f"Output path contains invalid characters: {invalid_chars}"

        return True, ""

    @staticmethod
    def validate_signal_array(signal: np.ndarray,
                             min_length: int = 10) -> Tuple[bool, str]:
        """
        Validate signal array for processing.

        Args:
            signal: Signal array to validate
            min_length: Minimum required length

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(signal, np.ndarray):
            return False, "Signal must be a NumPy array"

        if signal.ndim != 1:
            return False, f"Signal must be 1-dimensional (got {signal.ndim}D)"

        if len(signal) == 0:
            return False, "Signal array is empty"

        if len(signal) < min_length:
            return False, f"Signal too short (length={len(signal)}, minimum={min_length})"

        if not np.all(np.isfinite(signal)):
            nan_count = np.sum(np.isnan(signal))
            inf_count = np.sum(np.isinf(signal))
            return False, f"Signal contains non-finite values (NaN={nan_count}, Inf={inf_count})"

        if np.all(signal == 0):
            return False, "Signal is all zeros"

        return True, ""
