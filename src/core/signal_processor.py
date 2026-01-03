"""
Signal Processing Coordinator Module

This module coordinates signal processing operations, combining data loading,
feature extraction, and result management.

Author: EduCatCode
License: MIT
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Callable, Tuple
import logging

from .feature_extractor import TimeFeatureExtractor, FrequencyFeatureExtractor
from .data_loader import DataLoader


class SignalProcessor:
    """Coordinate signal processing operations."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the signal processor.

        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.data_loader = DataLoader(logger=self.logger)

    def process_time_features(self,
                              file_path: Path,
                              column_name: str,
                              frame_size: int,
                              hop_length: int) -> Tuple[pd.DataFrame, str]:
        """
        Extract time-domain features from a signal in a CSV file.

        Args:
            file_path: Path to CSV file
            column_name: Name of signal column to process
            frame_size: Window size for feature extraction
            hop_length: Hop length between windows

        Returns:
            Tuple of (features_dataframe, file_basename)

        Raises:
            ValueError: If column not found or data invalid
        """
        # Load data
        df = self.data_loader.load_csv(file_path)
        if df is None:
            raise ValueError(f"Failed to load file: {file_path}")

        # Validate column exists
        if column_name not in df.columns:
            raise ValueError(f"Column '{column_name}' not found in {file_path}")

        # Extract signal
        signal = df[column_name].values

        # Validate signal
        is_valid, error_msg = self.data_loader.validate_signal_data(
            signal, min_length=frame_size
        )
        if not is_valid:
            raise ValueError(f"Invalid signal data: {error_msg}")

        # Extract features
        extractor = TimeFeatureExtractor(frame_size, hop_length)
        feature_matrix, feature_names = extractor.extract_all_features(signal)

        # Create DataFrame
        features_df = pd.DataFrame(feature_matrix.T, columns=[
            f'{column_name} {name}' for name in feature_names
        ])

        file_basename = file_path.stem
        self.logger.info(f"Extracted {len(feature_names)} time features from {file_basename}")

        return features_df, file_basename

    def process_frequency_features(self,
                                   file_path: Path,
                                   column_name: str,
                                   frame_size: int,
                                   hop_length: int,
                                   sampling_rate: float,
                                   freq_band: Tuple[float, float] = (0, 5000)) -> Tuple[pd.DataFrame, str]:
        """
        Extract frequency-domain features from a signal in a CSV file.

        Args:
            file_path: Path to CSV file
            column_name: Name of signal column to process
            frame_size: Window size for feature extraction
            hop_length: Hop length between windows
            sampling_rate: Sampling rate in Hz
            freq_band: Frequency band for energy calculation (low, high)

        Returns:
            Tuple of (features_dataframe, file_basename)

        Raises:
            ValueError: If column not found or data invalid
        """
        # Load data
        df = self.data_loader.load_csv(file_path)
        if df is None:
            raise ValueError(f"Failed to load file: {file_path}")

        # Validate column exists
        if column_name not in df.columns:
            raise ValueError(f"Column '{column_name}' not found in {file_path}")

        # Extract signal
        signal = df[column_name].values

        # Validate signal
        is_valid, error_msg = self.data_loader.validate_signal_data(
            signal, min_length=frame_size
        )
        if not is_valid:
            raise ValueError(f"Invalid signal data: {error_msg}")

        # Extract features
        extractor = FrequencyFeatureExtractor(frame_size, hop_length, sampling_rate)
        feature_matrix, feature_names = extractor.extract_all_features(signal, freq_band)

        # Create DataFrame
        features_df = pd.DataFrame(feature_matrix.T, columns=[
            f'{column_name} {name}' for name in feature_names
        ])

        file_basename = file_path.stem
        self.logger.info(f"Extracted {len(feature_names)} frequency features from {file_basename}")

        return features_df, file_basename

    def batch_process_time_features(self,
                                    file_paths: List[Path],
                                    selected_columns: List[str],
                                    frame_size: int,
                                    hop_length: int,
                                    progress_callback: Optional[Callable[[int, int], None]] = None) -> Dict[str, pd.DataFrame]:
        """
        Process time-domain features for multiple files.

        Args:
            file_paths: List of CSV file paths
            selected_columns: Columns to process in each file
            frame_size: Window size
            hop_length: Hop length
            progress_callback: Optional callback function(current, total)

        Returns:
            Dictionary mapping file_basename to features DataFrame
        """
        results = {}
        total_files = len(file_paths)

        for i, file_path in enumerate(file_paths):
            try:
                self.logger.info(f"Processing {i+1}/{total_files}: {file_path.name}")

                # Load file
                df = self.data_loader.load_csv(file_path)

                # Check which columns exist
                available_columns = [col for col in selected_columns if col in df.columns]
                if not available_columns:
                    self.logger.warning(f"No selected columns found in {file_path.name}")
                    continue

                # Process each column
                for column_name in available_columns:
                    features_df, file_basename = self.process_time_features(
                        file_path, column_name, frame_size, hop_length
                    )

                    # Store results
                    result_key = f"{file_basename}_{column_name}"
                    results[result_key] = features_df

                # Update progress
                if progress_callback:
                    progress_callback(i + 1, total_files)

            except Exception as e:
                self.logger.error(f"Error processing {file_path.name}: {str(e)}")
                continue

        self.logger.info(f"Batch processing complete: {len(results)} results")
        return results

    def batch_process_frequency_features(self,
                                        file_paths: List[Path],
                                        selected_columns: List[str],
                                        frame_size: int,
                                        hop_length: int,
                                        sampling_rate: float,
                                        freq_band: Tuple[float, float] = (0, 5000),
                                        progress_callback: Optional[Callable[[int, int], None]] = None) -> Dict[str, pd.DataFrame]:
        """
        Process frequency-domain features for multiple files.

        Args:
            file_paths: List of CSV file paths
            selected_columns: Columns to process in each file
            frame_size: Window size
            hop_length: Hop length
            sampling_rate: Sampling rate in Hz
            freq_band: Frequency band (low, high)
            progress_callback: Optional callback function(current, total)

        Returns:
            Dictionary mapping file_basename to features DataFrame
        """
        results = {}
        total_files = len(file_paths)

        for i, file_path in enumerate(file_paths):
            try:
                self.logger.info(f"Processing {i+1}/{total_files}: {file_path.name}")

                # Load file
                df = self.data_loader.load_csv(file_path)

                # Check which columns exist
                available_columns = [col for col in selected_columns if col in df.columns]
                if not available_columns:
                    self.logger.warning(f"No selected columns found in {file_path.name}")
                    continue

                # Process each column
                for column_name in available_columns:
                    features_df, file_basename = self.process_frequency_features(
                        file_path, column_name, frame_size, hop_length,
                        sampling_rate, freq_band
                    )

                    # Store results
                    result_key = f"{file_basename}_{column_name}"
                    results[result_key] = features_df

                # Update progress
                if progress_callback:
                    progress_callback(i + 1, total_files)

            except Exception as e:
                self.logger.error(f"Error processing {file_path.name}: {str(e)}")
                continue

        self.logger.info(f"Batch processing complete: {len(results)} results")
        return results

    def merge_csv_files(self,
                       file_paths: List[Path],
                       output_path: Path,
                       progress_callback: Optional[Callable[[int, int], None]] = None) -> bool:
        """
        Merge multiple CSV files by concatenating rows.

        Args:
            file_paths: List of CSV file paths to merge
            output_path: Output file path
            progress_callback: Optional callback function(current, total)

        Returns:
            True if successful, False otherwise
        """
        try:
            dataframes = []
            total_files = len(file_paths)

            for i, file_path in enumerate(file_paths):
                try:
                    df = self.data_loader.load_csv(file_path)
                    dataframes.append(df)

                    if progress_callback:
                        progress_callback(i + 1, total_files)

                except Exception as e:
                    self.logger.error(f"Error loading {file_path.name}: {str(e)}")
                    continue

            if not dataframes:
                self.logger.error("No dataframes loaded for merging")
                return False

            # Merge dataframes
            merged_df = self.data_loader.merge_dataframes(dataframes, axis=0)

            # Save result
            merged_df.to_csv(output_path, index=False)
            self.logger.info(f"Merged {len(dataframes)} files to {output_path}")
            return True

        except Exception as e:
            self.logger.error(f"Error merging files: {str(e)}")
            return False

    def align_and_merge_by_timestamp(self,
                                    file_path1: Path,
                                    file_path2: Path,
                                    output_path: Path,
                                    timestamp_col: str = 'timestamp',
                                    direction: str = 'nearest') -> bool:
        """
        Align and merge two CSV files by timestamp.

        Args:
            file_path1: First CSV file path
            file_path2: Second CSV file path
            output_path: Output file path
            timestamp_col: Name of timestamp column
            direction: Merge direction ('nearest', 'forward', 'backward')

        Returns:
            True if successful, False otherwise
        """
        try:
            # Load files
            df1 = self.data_loader.load_csv(file_path1)
            df2 = self.data_loader.load_csv(file_path2)

            if df1 is None or df2 is None:
                self.logger.error("Failed to load one or both files")
                return False

            # Align and merge
            merged_df = self.data_loader.align_by_timestamp(
                df1, df2, timestamp_col, direction
            )

            # Save result
            merged_df.to_csv(output_path, index=False)
            self.logger.info(f"Aligned and merged files to {output_path}")
            return True

        except Exception as e:
            self.logger.error(f"Error aligning files: {str(e)}")
            return False
