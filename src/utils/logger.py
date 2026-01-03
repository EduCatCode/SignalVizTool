"""
Logging Configuration Module

Provides centralized logging configuration for the application.

Author: EduCatCode
License: MIT
"""

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Optional


class LoggerSetup:
    """Setup and configure application logging."""

    _loggers = {}

    @staticmethod
    def setup_logger(name: str = 'SignalVizTool',
                    level: str = 'INFO',
                    log_dir: Optional[Path] = None,
                    log_file: Optional[str] = None,
                    console_logging: bool = True,
                    file_logging: bool = True,
                    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
                    backup_count: int = 5) -> logging.Logger:
        """
        Setup and configure a logger instance.

        Args:
            name: Logger name
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_dir: Directory for log files
            log_file: Log file name
            console_logging: Enable console output
            file_logging: Enable file output
            max_bytes: Maximum log file size before rotation
            backup_count: Number of backup files to keep

        Returns:
            Configured logger instance
        """
        # Return existing logger if already configured
        if name in LoggerSetup._loggers:
            return LoggerSetup._loggers[name]

        # Create logger
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level.upper()))
        logger.propagate = False

        # Clear existing handlers
        logger.handlers.clear()

        # Create formatter
        formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Console handler
        if console_logging:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, level.upper()))
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        # File handler with rotation
        if file_logging:
            if log_dir is None:
                log_dir = Path('logs')
            if log_file is None:
                log_file = f'{name.lower()}.log'

            # Create log directory
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)

            log_path = log_dir / log_file

            file_handler = RotatingFileHandler(
                log_path,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding='utf-8'
            )
            file_handler.setLevel(getattr(logging, level.upper()))
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        # Store logger reference
        LoggerSetup._loggers[name] = logger

        logger.info(f"Logger '{name}' initialized with level {level}")
        return logger

    @staticmethod
    def get_logger(name: str = 'SignalVizTool') -> logging.Logger:
        """
        Get an existing logger or create a new one with defaults.

        Args:
            name: Logger name

        Returns:
            Logger instance
        """
        if name in LoggerSetup._loggers:
            return LoggerSetup._loggers[name]

        return LoggerSetup.setup_logger(name)

    @staticmethod
    def set_level(name: str, level: str):
        """
        Set logging level for an existing logger.

        Args:
            name: Logger name
            level: New logging level
        """
        if name in LoggerSetup._loggers:
            logger = LoggerSetup._loggers[name]
            logger.setLevel(getattr(logging, level.upper()))
            for handler in logger.handlers:
                handler.setLevel(getattr(logging, level.upper()))
            logger.info(f"Logging level changed to {level}")

    @staticmethod
    def shutdown():
        """Shutdown all loggers and close handlers."""
        for logger in LoggerSetup._loggers.values():
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)
        LoggerSetup._loggers.clear()
        logging.shutdown()


def get_logger(name: str = 'SignalVizTool') -> logging.Logger:
    """
    Convenience function to get a logger instance.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    return LoggerSetup.get_logger(name)
