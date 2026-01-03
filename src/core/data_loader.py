"""
Data Loading and Validation Module

This module handles CSV file loading, validation, and preprocessing for signal analysis.

Features:
    - Robust CSV loading with error handling
    - Data validation (type checking, range validation)
    - Missing value handling
    - Multi-file batch loading
    - Timestamp parsing and alignment

Author: EduCatCode
License: MIT
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Union
import logging


class DataLoader:
    """Load and validate CSV data files for signal processing."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the data loader.

        Args:
            logger: Optional logger instance for logging operations
        """
        self.logger = logger or logging.getLogger(__name__)

    def load_csv(self, file_path: Union[str, Path],
                encoding: str = 'utf-8') -> Optional[pd.DataFrame]:
        """
        Load a single CSV file with comprehensive error handling.

        Args:
            file_path: Path to the CSV file
            encoding: File encoding (default: 'utf-8')

        Returns:
            DataFrame if successful, None if failed

        Raises:
            FileNotFoundError: If file doesn't exist
            pd.errors.EmptyDataError: If file is empty
            pd.errors.ParserError: If CSV parsing fails
        """
        file_path = Path(file_path)

        # Validate file exists
        if not file_path.exists():
            self.logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")

        # Validate file size
        if file_path.stat().st_size == 0:
            self.logger.error(f"File is empty: {file_path}")
            raise pd.errors.EmptyDataError(f"File is empty: {file_path}")

        try:
            # Try loading with specified encoding
            df = pd.read_csv(file_path, encoding=encoding)
            self.logger.info(f"Successfully loaded {file_path} ({len(df)} rows, {len(df.columns)} columns)")
            return df

        except UnicodeDecodeError:
            # Try alternative encodings
            self.logger.warning(f"Failed to decode with {encoding}, trying alternative encodings...")
            for alt_encoding in ['gbk', 'utf-8-sig', 'latin1']:
                try:
                    df = pd.read_csv(file_path, encoding=alt_encoding)
                    self.logger.info(f"Successfully loaded with {alt_encoding} encoding")
                    return df
                except UnicodeDecodeError:
                    continue

            self.logger.error(f"Failed to decode file with any encoding: {file_path}")
            raise

        except pd.errors.ParserError as e:
            self.logger.error(f"CSV parsing error in {file_path}: {str(e)}")
            raise

        except Exception as e:
            self.logger.error(f"Unexpected error loading {file_path}: {str(e)}")
            raise

    def load_multiple_csv(self, file_paths: List[Union[str, Path]],
                         encoding: str = 'utf-8') -> List[Tuple[Path, pd.DataFrame]]:
        """
        Load multiple CSV files.

        Args:
            file_paths: List of file paths
            encoding: File encoding

        Returns:
            List of (file_path, DataFrame) tuples for successfully loaded files
        """
        loaded_data = []

        for file_path in file_paths:
            try:
                df = self.load_csv(file_path, encoding)
                loaded_data.append((Path(file_path), df))
            except Exception as e:
                self.logger.warning(f"Skipping {file_path} due to error: {str(e)}")
                continue

        self.logger.info(f"Successfully loaded {len(loaded_data)}/{len(file_paths)} files")
        return loaded_data

    def validate_columns(self, df: pd.DataFrame,
                        required_columns: Optional[List[str]] = None) -> Tuple[bool, List[str]]:
        """
        Validate that DataFrame contains required columns.

        Args:
            df: DataFrame to validate
            required_columns: List of required column names

        Returns:
            Tuple of (is_valid, missing_columns)
        """
        if required_columns is None:
            return True, []

        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            self.logger.warning(f"Missing columns: {missing_columns}")
            return False, missing_columns

        return True, []

    def validate_numeric_columns(self, df: pd.DataFrame,
                                columns: Optional[List[str]] = None) -> Tuple[bool, List[str]]:
        """
        Validate that specified columns contain numeric data.

        Args:
            df: DataFrame to validate
            columns: List of column names to check (None = check all)

        Returns:
            Tuple of (is_valid, non_numeric_columns)
        """
        columns = columns or df.columns.tolist()
        non_numeric = []

        for col in columns:
            if col not in df.columns:
                continue

            if not pd.api.types.is_numeric_dtype(df[col]):
                # Try converting to numeric
                try:
                    pd.to_numeric(df[col])
                except (ValueError, TypeError):
                    non_numeric.append(col)

        if non_numeric:
            self.logger.warning(f"Non-numeric columns: {non_numeric}")
            return False, non_numeric

        return True, []

    def handle_missing_values(self, df: pd.DataFrame,
                             strategy: str = 'drop',
                             fill_value: Optional[float] = None) -> pd.DataFrame:
        """
        Handle missing values in DataFrame.

        Args:
            df: Input DataFrame
            strategy: Strategy for handling missing values
                     ('drop', 'forward_fill', 'backward_fill', 'interpolate', 'constant')
            fill_value: Value to use for 'constant' strategy

        Returns:
            DataFrame with missing values handled
        """
        missing_count = df.isnull().sum().sum()

        if missing_count == 0:
            self.logger.info("No missing values found")
            return df

        self.logger.info(f"Found {missing_count} missing values")

        if strategy == 'drop':
            df_clean = df.dropna()
            self.logger.info(f"Dropped {len(df) - len(df_clean)} rows with missing values")
            return df_clean

        elif strategy == 'forward_fill':
            return df.fillna(method='ffill')

        elif strategy == 'backward_fill':
            return df.fillna(method='bfill')

        elif strategy == 'interpolate':
            return df.interpolate(method='linear')

        elif strategy == 'constant':
            if fill_value is None:
                fill_value = 0
            return df.fillna(fill_value)

        else:
            self.logger.warning(f"Unknown strategy '{strategy}', using 'drop'")
            return df.dropna()

    def get_common_columns(self, dataframes: List[pd.DataFrame]) -> List[str]:
        """
        Find common columns across multiple DataFrames.

        Args:
            dataframes: List of DataFrames

        Returns:
            List of column names common to all DataFrames
        """
        if not dataframes:
            return []

        common_cols = set(dataframes[0].columns)
        for df in dataframes[1:]:
            common_cols &= set(df.columns)

        common_cols = sorted(list(common_cols))
        self.logger.info(f"Found {len(common_cols)} common columns across {len(dataframes)} files")
        return common_cols

    def merge_dataframes(self, dataframes: List[pd.DataFrame],
                        axis: int = 0) -> pd.DataFrame:
        """
        Merge multiple DataFrames.

        Args:
            dataframes: List of DataFrames to merge
            axis: Concatenation axis (0=rows, 1=columns)

        Returns:
            Merged DataFrame
        """
        if not dataframes:
            raise ValueError("No dataframes to merge")

        merged = pd.concat(dataframes, axis=axis, ignore_index=True)
        self.logger.info(f"Merged {len(dataframes)} dataframes into shape {merged.shape}")
        return merged

    def align_by_timestamp(self, df1: pd.DataFrame, df2: pd.DataFrame,
                          timestamp_col: str = 'timestamp',
                          direction: str = 'nearest') -> pd.DataFrame:
        """
        Align and merge two DataFrames by timestamp.

        Args:
            df1: First DataFrame
            df2: Second DataFrame
            timestamp_col: Name of timestamp column
            direction: Merge direction ('nearest', 'forward', 'backward')

        Returns:
            Merged DataFrame aligned by timestamp

        Raises:
            ValueError: If timestamp column not found
        """
        # Validate timestamp column exists
        if timestamp_col not in df1.columns:
            raise ValueError(f"Timestamp column '{timestamp_col}' not found in first DataFrame")
        if timestamp_col not in df2.columns:
            raise ValueError(f"Timestamp column '{timestamp_col}' not found in second DataFrame")

        # Parse timestamps
        try:
            df1_copy = df1.copy()
            df2_copy = df2.copy()

            df1_copy[timestamp_col] = pd.to_datetime(df1_copy[timestamp_col])
            df2_copy[timestamp_col] = pd.to_datetime(df2_copy[timestamp_col])

            # Set timestamp as index
            df1_copy = df1_copy.set_index(timestamp_col)
            df2_copy = df2_copy.set_index(timestamp_col)

            # Find overlapping time range
            start_time = max(df1_copy.index.min(), df2_copy.index.min())
            end_time = min(df1_copy.index.max(), df2_copy.index.max())

            self.logger.info(f"Aligning timestamps from {start_time} to {end_time}")

            # Filter to overlapping range
            df1_filtered = df1_copy[(df1_copy.index >= start_time) & (df1_copy.index <= end_time)]
            df2_filtered = df2_copy[(df2_copy.index >= start_time) & (df2_copy.index <= end_time)]

            # Merge using merge_asof
            merged = pd.merge_asof(
                df1_filtered.sort_index(),
                df2_filtered.sort_index(),
                left_index=True,
                right_index=True,
                direction=direction
            )

            self.logger.info(f"Aligned merge complete: {len(merged)} rows")
            return merged.reset_index()

        except Exception as e:
            self.logger.error(f"Error aligning timestamps: {str(e)}")
            raise

    def validate_signal_data(self, signal: np.ndarray,
                           min_length: int = 10,
                           check_finite: bool = True) -> Tuple[bool, str]:
        """
        Validate signal data for processing.

        Args:
            signal: Signal array to validate
            min_length: Minimum required length
            check_finite: Whether to check for infinite/NaN values

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check if array is empty
        if len(signal) == 0:
            return False, "Signal is empty"

        # Check minimum length
        if len(signal) < min_length:
            return False, f"Signal too short (length={len(signal)}, minimum={min_length})"

        # Check for finite values
        if check_finite:
            if not np.all(np.isfinite(signal)):
                nan_count = np.sum(np.isnan(signal))
                inf_count = np.sum(np.isinf(signal))
                return False, f"Signal contains non-finite values (NaN={nan_count}, Inf={inf_count})"

        # Check if signal is all zeros
        if np.all(signal == 0):
            return False, "Signal is all zeros"

        return True, ""
