"""
Visualization and Plotting Module

Provides plotting functions for signal data and features.

Author: EduCatCode
License: MIT
"""

import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
import logging


class SignalPlotter:
    """Handle signal visualization and plotting."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the signal plotter.

        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)

        # Configure matplotlib for Chinese fonts
        matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        matplotlib.rcParams['axes.unicode_minus'] = False

    def plot_time_domain_features(self,
                                  features_df: pd.DataFrame,
                                  title: str = "Time Domain Features",
                                  figsize: Tuple[int, int] = (10, 8),
                                  save_path: Optional[Path] = None) -> plt.Figure:
        """
        Plot time-domain features in a grid layout.

        Args:
            features_df: DataFrame containing features
            title: Plot title
            figsize: Figure size (width, height)
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure object
        """
        n_features = len(features_df.columns)
        n_rows = 4
        n_cols = 4

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        fig.suptitle(title, fontsize=16)

        # Flatten axes for easier iteration
        axes_flat = axes.flatten()

        # Plot each feature
        for i, feature_name in enumerate(features_df.columns):
            if i >= len(axes_flat):
                break

            ax = axes_flat[i]
            data = features_df[feature_name]

            # Create bar plot
            ax.bar(range(len(data)), data,
                  color=plt.cm.tab20(i / max(n_features, 1)),
                  alpha=0.8)

            ax.set_title(feature_name, fontsize=8)
            ax.set_xlabel('Frame Index', fontsize=6)
            ax.set_ylabel('Value', fontsize=6)
            ax.tick_params(axis='both', labelsize=5)
            ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.3)

        # Hide unused subplots
        for i in range(n_features, len(axes_flat)):
            axes_flat[i].axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # Save if path provided
        if save_path:
            try:
                fig.savefig(save_path, dpi=100, bbox_inches='tight')
                self.logger.info(f"Saved plot to {save_path}")
            except Exception as e:
                self.logger.error(f"Error saving plot: {str(e)}")

        return fig

    def plot_frequency_domain_features(self,
                                      features_df: pd.DataFrame,
                                      title: str = "Frequency Domain Features",
                                      figsize: Tuple[int, int] = (10, 8),
                                      save_path: Optional[Path] = None) -> plt.Figure:
        """
        Plot frequency-domain features in a grid layout.

        Args:
            features_df: DataFrame containing features
            title: Plot title
            figsize: Figure size (width, height)
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure object
        """
        return self.plot_time_domain_features(features_df, title, figsize, save_path)

    def plot_signal_comparison(self,
                              dataframes: List[Tuple[str, pd.DataFrame]],
                              selected_columns: List[str],
                              figsize: Tuple[int, int] = (12, 4),
                              save_dir: Optional[Path] = None) -> List[plt.Figure]:
        """
        Plot multiple signals for comparison.

        Args:
            dataframes: List of (label, dataframe) tuples
            selected_columns: Columns to plot
            figsize: Figure size for each plot
            save_dir: Optional directory to save figures

        Returns:
            List of matplotlib figure objects
        """
        figures = []

        for label, df in dataframes:
            # Create figure
            fig, ax = plt.subplots(figsize=figsize)

            # Plot selected columns
            for col in selected_columns:
                if col in df.columns:
                    ax.plot(df[col], label=col, linewidth=1.5, alpha=0.8)

            ax.set_title(label, fontsize=14)
            ax.set_xlabel('Sample Index', fontsize=10)
            ax.set_ylabel('Amplitude', fontsize=10)
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.3)

            plt.tight_layout()

            # Save if directory provided
            if save_dir:
                try:
                    save_dir = Path(save_dir)
                    save_dir.mkdir(parents=True, exist_ok=True)
                    save_path = save_dir / f"{label}.png"
                    fig.savefig(save_path, dpi=100, bbox_inches='tight')
                    self.logger.info(f"Saved plot to {save_path}")
                except Exception as e:
                    self.logger.error(f"Error saving plot for {label}: {str(e)}")

            figures.append(fig)

        return figures

    def plot_single_signal(self,
                          signal: np.ndarray,
                          title: str = "Signal",
                          xlabel: str = "Sample Index",
                          ylabel: str = "Amplitude",
                          figsize: Tuple[int, int] = (12, 4),
                          save_path: Optional[Path] = None) -> plt.Figure:
        """
        Plot a single signal.

        Args:
            signal: 1D signal array
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            figsize: Figure size
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=figsize)

        ax.plot(signal, linewidth=1.5, color='#2E86AB', alpha=0.8)
        ax.set_title(title, fontsize=14)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.3)

        plt.tight_layout()

        if save_path:
            try:
                fig.savefig(save_path, dpi=100, bbox_inches='tight')
                self.logger.info(f"Saved plot to {save_path}")
            except Exception as e:
                self.logger.error(f"Error saving plot: {str(e)}")

        return fig

    def close_figure(self, fig: plt.Figure):
        """
        Properly close a matplotlib figure to free memory.

        Args:
            fig: Figure to close
        """
        try:
            plt.close(fig)
        except Exception as e:
            self.logger.warning(f"Error closing figure: {str(e)}")

    def close_all_figures(self):
        """Close all matplotlib figures."""
        plt.close('all')
        self.logger.debug("Closed all matplotlib figures")
