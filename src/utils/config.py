"""
Configuration Management Module

Handles application configuration loading, validation, and access.

Author: EduCatCode
License: MIT
"""

import json
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
import logging


class Config:
    """Application configuration manager with singleton pattern."""

    _instance = None
    _config_data = {}

    def __new__(cls):
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize configuration with defaults."""
        if not self._config_data:
            self._config_data = self._get_default_config()
            self.logger = logging.getLogger(__name__)

    @staticmethod
    def _get_default_config() -> Dict[str, Any]:
        """
        Get default configuration values.

        Returns:
            Dictionary of default configuration
        """
        return {
            # Application settings
            'app': {
                'name': 'SignalVizTool',
                'version': '2.0.0',
                'window_title': 'SignalVizTool - Professional Signal Analysis',
                'icon_file': 'EduCatCode.ico'
            },

            # Processing settings
            'processing': {
                'default_frame_size': 1024,
                'default_hop_length': 512,
                'default_sampling_rate': 10000,
                'max_frame_size': 200000,
                'max_hop_length': 200000,
                'max_file_size_mb': 100,
                'chunk_size': 10000,  # For large file processing
            },

            # Frequency analysis settings
            'frequency': {
                'default_freq_band_low': 0,
                'default_freq_band_high': 5000,
                'min_frequency': 0,
                'max_frequency': 50000,
            },

            # UI settings
            'ui': {
                'theme': 'default',
                'font_family': 'SimHei',  # Chinese font support
                'font_size': 10,
                'progress_bar_width': 200,
                'button_padding': 10,
                'window_state': 'zoomed',  # or 'normal'
            },

            # Plotting settings
            'plotting': {
                'figure_width': 12,
                'figure_height': 4,
                'grid_rows': 4,
                'grid_cols': 4,
                'dpi': 100,
                'save_format': 'png',
                'color_map': 'tab20',
            },

            # File handling
            'files': {
                'default_encoding': 'utf-8',
                'alternative_encodings': ['gbk', 'utf-8-sig', 'latin1'],
                'csv_delimiter': ',',
                'output_directory': 'results',
            },

            # Logging settings
            'logging': {
                'level': 'INFO',  # DEBUG, INFO, WARNING, ERROR, CRITICAL
                'file_logging': True,
                'log_directory': 'logs',
                'log_file_name': 'signalviztool.log',
                'max_log_size_mb': 10,
                'backup_count': 5,
                'console_logging': True,
            },

            # Performance settings
            'performance': {
                'enable_multithreading': True,
                'max_workers': 4,
                'memory_limit_mb': 1024,
            },

            # Data validation
            'validation': {
                'min_signal_length': 10,
                'check_finite_values': True,
                'handle_missing_values': 'drop',  # drop, interpolate, forward_fill, etc.
            }
        }

    def load_from_file(self, config_file: Path) -> bool:
        """
        Load configuration from a file (JSON or YAML).

        Args:
            config_file: Path to configuration file

        Returns:
            True if successful, False otherwise
        """
        if not config_file.exists():
            self.logger.warning(f"Config file not found: {config_file}")
            return False

        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                if config_file.suffix in ['.yaml', '.yml']:
                    user_config = yaml.safe_load(f)
                elif config_file.suffix == '.json':
                    user_config = json.load(f)
                else:
                    self.logger.error(f"Unsupported config file format: {config_file.suffix}")
                    return False

            # Merge with defaults
            self._merge_config(user_config)
            self.logger.info(f"Loaded configuration from {config_file}")
            return True

        except Exception as e:
            self.logger.error(f"Error loading config file: {str(e)}")
            return False

    def _merge_config(self, user_config: Dict[str, Any]):
        """
        Merge user configuration with defaults.

        Args:
            user_config: User-provided configuration dictionary
        """
        for section, values in user_config.items():
            if section in self._config_data:
                if isinstance(values, dict):
                    self._config_data[section].update(values)
                else:
                    self._config_data[section] = values
            else:
                self._config_data[section] = values

    def save_to_file(self, config_file: Path) -> bool:
        """
        Save current configuration to file.

        Args:
            config_file: Path to save configuration

        Returns:
            True if successful, False otherwise
        """
        try:
            config_file.parent.mkdir(parents=True, exist_ok=True)

            with open(config_file, 'w', encoding='utf-8') as f:
                if config_file.suffix in ['.yaml', '.yml']:
                    yaml.dump(self._config_data, f, default_flow_style=False)
                elif config_file.suffix == '.json':
                    json.dump(self._config_data, f, indent=2)
                else:
                    self.logger.error(f"Unsupported config file format: {config_file.suffix}")
                    return False

            self.logger.info(f"Saved configuration to {config_file}")
            return True

        except Exception as e:
            self.logger.error(f"Error saving config file: {str(e)}")
            return False

    def get(self, section: str, key: Optional[str] = None, default: Any = None) -> Any:
        """
        Get configuration value.

        Args:
            section: Configuration section
            key: Optional key within section
            default: Default value if not found

        Returns:
            Configuration value or default

        Examples:
            >>> config.get('app', 'name')
            'SignalVizTool'
            >>> config.get('processing')
            {'default_frame_size': 1024, ...}
        """
        if section not in self._config_data:
            return default

        if key is None:
            return self._config_data[section]

        return self._config_data[section].get(key, default)

    def set(self, section: str, key: str, value: Any):
        """
        Set configuration value.

        Args:
            section: Configuration section
            key: Key within section
            value: Value to set
        """
        if section not in self._config_data:
            self._config_data[section] = {}

        self._config_data[section][key] = value
        self.logger.debug(f"Set config: {section}.{key} = {value}")

    def get_all(self) -> Dict[str, Any]:
        """
        Get all configuration data.

        Returns:
            Complete configuration dictionary
        """
        return self._config_data.copy()

    def reset_to_defaults(self):
        """Reset configuration to default values."""
        self._config_data = self._get_default_config()
        self.logger.info("Configuration reset to defaults")


# Global configuration instance
config = Config()
