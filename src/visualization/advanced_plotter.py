"""
Advanced Visualization Module

Provides modern, interactive, and publication-quality visualizations including:
- Interactive plots with Plotly
- Spectrograms and time-frequency plots
- 3D visualizations
- Publication-ready scientific plots
- Dashboard-style multi-panel layouts

Author: EduCatCode - Design & Engineering Department
Version: 2.1.0
License: MIT
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import logging

# Try to import plotly for interactive plots
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Seaborn for beautiful statistical plots
try:
    import seaborn as sns
    sns.set_theme(style="whitegrid")
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False


class AdvancedPlotter:
    """Advanced plotting with modern aesthetics and interactivity."""

    def __init__(self, style: str = 'modern', logger: Optional[logging.Logger] = None):
        """
        Initialize advanced plotter.

        Args:
            style: Plot style ('modern', 'scientific', 'dark', 'minimal')
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.style = style
        self._setup_style()

    def _setup_style(self):
        """Setup matplotlib style based on selection."""
        if self.style == 'modern':
            plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
            matplotlib.rcParams['font.sans-serif'] = ['Arial', 'SimHei', 'DejaVu Sans']
            matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(
                color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
            )
        elif self.style == 'scientific':
            matplotlib.rcParams.update({
                'font.size': 10,
                'axes.labelsize': 11,
                'axes.titlesize': 12,
                'xtick.labelsize': 9,
                'ytick.labelsize': 9,
                'legend.fontsize': 9,
                'figure.titlesize': 13,
                'font.family': 'sans-serif',
            })
        elif self.style == 'dark':
            plt.style.use('dark_background')

    def plot_spectrogram(self, freqs: np.ndarray, times: np.ndarray,
                        Sxx: np.ndarray, title: str = "Spectrogram",
                        save_path: Optional[Path] = None,
                        interactive: bool = False) -> Optional[Figure]:
        """
        Plot spectrogram with modern aesthetics.

        Args:
            freqs: Frequency array
            times: Time array
            Sxx: Spectrogram matrix
            title: Plot title
            save_path: Optional save path
            interactive: Use interactive Plotly plot

        Returns:
            Figure object or None if interactive
        """
        if interactive and PLOTLY_AVAILABLE:
            return self._plot_spectrogram_interactive(freqs, times, Sxx, title, save_path)
        else:
            return self._plot_spectrogram_static(freqs, times, Sxx, title, save_path)

    def _plot_spectrogram_static(self, freqs, times, Sxx, title, save_path):
        """Plot static spectrogram."""
        fig, ax = plt.subplots(figsize=(12, 6))

        # Use log scale for better visualization
        Sxx_db = 10 * np.log10(Sxx + 1e-10)

        # Plot
        im = ax.pcolormesh(times, freqs, Sxx_db, shading='gouraud', cmap='viridis')

        ax.set_ylabel('Frequency (Hz)', fontsize=11)
        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')

        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Power (dB)', fontsize=10)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            self.logger.info(f"Spectrogram saved to {save_path}")

        return fig

    def _plot_spectrogram_interactive(self, freqs, times, Sxx, title, save_path):
        """Plot interactive spectrogram using Plotly."""
        Sxx_db = 10 * np.log10(Sxx + 1e-10)

        fig = go.Figure(data=go.Heatmap(
            z=Sxx_db,
            x=times,
            y=freqs,
            colorscale='Viridis',
            colorbar=dict(title='Power (dB)')
        ))

        fig.update_layout(
            title=title,
            xaxis_title='Time (s)',
            yaxis_title='Frequency (Hz)',
            width=1000,
            height=600
        )

        if save_path:
            # Save as HTML for interactivity
            html_path = save_path.with_suffix('.html')
            fig.write_html(str(html_path))
            self.logger.info(f"Interactive spectrogram saved to {html_path}")

        fig.show()
        return None

    def plot_wavelet_transform(self, coefficients: np.ndarray, scales: np.ndarray,
                              title: str = "Wavelet Transform",
                              save_path: Optional[Path] = None) -> Figure:
        """
        Plot wavelet transform coefficients.

        Args:
            coefficients: CWT coefficients matrix
            scales: Scale values
            title: Plot title
            save_path: Optional save path

        Returns:
            Figure object
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot
        im = ax.imshow(
            np.abs(coefficients),
            aspect='auto',
            cmap='viridis',
            extent=[0, coefficients.shape[1], scales[-1], scales[0]],
            interpolation='bilinear'
        )

        ax.set_ylabel('Scale', fontsize=11)
        ax.set_xlabel('Time', fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')

        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Magnitude', fontsize=10)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    def plot_fft_analysis(self, freqs: np.ndarray, magnitude: np.ndarray,
                         title: str = "FFT Analysis",
                         dominant_freqs: Optional[pd.DataFrame] = None,
                         save_path: Optional[Path] = None,
                         interactive: bool = False) -> Optional[Figure]:
        """
        Plot FFT analysis with highlighted dominant frequencies.

        Args:
            freqs: Frequency array
            magnitude: FFT magnitude
            title: Plot title
            dominant_freqs: DataFrame with dominant frequencies
            save_path: Optional save path
            interactive: Use interactive plot

        Returns:
            Figure object or None
        """
        if interactive and PLOTLY_AVAILABLE:
            fig = go.Figure()

            # Main FFT line
            fig.add_trace(go.Scatter(
                x=freqs,
                y=magnitude,
                mode='lines',
                name='FFT Magnitude',
                line=dict(color='#2E86AB', width=2)
            ))

            # Mark dominant frequencies
            if dominant_freqs is not None:
                fig.add_trace(go.Scatter(
                    x=dominant_freqs['Frequency (Hz)'],
                    y=dominant_freqs['Magnitude'],
                    mode='markers',
                    name='Dominant Frequencies',
                    marker=dict(size=12, color='red', symbol='star')
                ))

            fig.update_layout(
                title=title,
                xaxis_title='Frequency (Hz)',
                yaxis_title='Magnitude',
                hovermode='x unified',
                width=1000,
                height=500
            )

            if save_path:
                fig.write_html(str(save_path.with_suffix('.html')))

            fig.show()
            return None

        else:
            fig, ax = plt.subplots(figsize=(12, 5))

            # Plot FFT
            ax.plot(freqs, magnitude, linewidth=2, color='#2E86AB', alpha=0.8)
            ax.fill_between(freqs, magnitude, alpha=0.3, color='#2E86AB')

            # Mark dominant frequencies
            if dominant_freqs is not None:
                ax.plot(
                    dominant_freqs['Frequency (Hz)'],
                    dominant_freqs['Magnitude'],
                    'r*', markersize=15, label='Dominant Frequencies'
                )

                # Add labels
                for _, row in dominant_freqs.iterrows():
                    ax.annotate(
                        f"{row['Frequency (Hz)']:.1f} Hz",
                        xy=(row['Frequency (Hz)'], row['Magnitude']),
                        xytext=(10, 10),
                        textcoords='offset points',
                        fontsize=8,
                        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
                    )

            ax.set_xlabel('Frequency (Hz)', fontsize=11)
            ax.set_ylabel('Magnitude', fontsize=11)
            ax.set_title(title, fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3)
            if dominant_freqs is not None:
                ax.legend()

            plt.tight_layout()

            if save_path:
                fig.savefig(save_path, dpi=150, bbox_inches='tight')

            return fig

    def plot_signal_comparison_interactive(self, signals: Dict[str, np.ndarray],
                                          title: str = "Signal Comparison",
                                          save_path: Optional[Path] = None):
        """
        Create interactive comparison plot with Plotly.

        Args:
            signals: Dictionary of {label: signal_array}
            title: Plot title
            save_path: Optional save path
        """
        if not PLOTLY_AVAILABLE:
            self.logger.warning("Plotly not available. Use static plots instead.")
            return

        fig = go.Figure()

        for label, signal in signals.items():
            fig.add_trace(go.Scatter(
                y=signal,
                mode='lines',
                name=label,
                line=dict(width=2)
            ))

        fig.update_layout(
            title=title,
            xaxis_title='Sample Index',
            yaxis_title='Amplitude',
            hovermode='x unified',
            width=1200,
            height=600,
            legend=dict(x=0.01, y=0.99)
        )

        if save_path:
            fig.write_html(str(save_path))
            self.logger.info(f"Interactive plot saved to {save_path}")

        fig.show()

    def create_analysis_dashboard(self, signal_data: np.ndarray,
                                 freqs: np.ndarray, fft_magnitude: np.ndarray,
                                 times_stft: np.ndarray, freqs_stft: np.ndarray,
                                 spectrogram: np.ndarray,
                                 title: str = "Signal Analysis Dashboard",
                                 save_path: Optional[Path] = None) -> Figure:
        """
        Create comprehensive analysis dashboard.

        Args:
            signal_data: Time-domain signal
            freqs: FFT frequencies
            fft_magnitude: FFT magnitude
            times_stft: STFT time array
            freqs_stft: STFT frequency array
            spectrogram: Spectrogram matrix
            title: Dashboard title
            save_path: Optional save path

        Returns:
            Figure object
        """
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

        # Title
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)

        # 1. Time domain signal
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(signal_data, linewidth=1, color='#2E86AB', alpha=0.8)
        ax1.set_title('Time Domain Signal', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Sample Index')
        ax1.set_ylabel('Amplitude')
        ax1.grid(True, alpha=0.3)

        # 2. FFT
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(freqs, fft_magnitude, linewidth=2, color='#A23B72')
        ax2.fill_between(freqs, fft_magnitude, alpha=0.3, color='#A23B72')
        ax2.set_title('Frequency Spectrum (FFT)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Magnitude')
        ax2.grid(True, alpha=0.3)

        # 3. Power Spectrum
        ax3 = fig.add_subplot(gs[1, 1])
        power = fft_magnitude ** 2
        ax3.semilogy(freqs, power, linewidth=2, color='#F18F01')
        ax3.set_title('Power Spectrum', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Frequency (Hz)')
        ax3.set_ylabel('Power')
        ax3.grid(True, alpha=0.3)

        # 4. Spectrogram
        ax4 = fig.add_subplot(gs[2, :])
        Sxx_db = 10 * np.log10(spectrogram + 1e-10)
        im = ax4.pcolormesh(times_stft, freqs_stft, Sxx_db, shading='gouraud', cmap='viridis')
        ax4.set_title('Spectrogram', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Frequency (Hz)')
        cbar = plt.colorbar(im, ax=ax4)
        cbar.set_label('Power (dB)')

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            self.logger.info(f"Dashboard saved to {save_path}")

        return fig

    def plot_ai_results(self, predictions: np.ndarray, true_labels: Optional[np.ndarray] = None,
                       title: str = "AI Analysis Results",
                       save_path: Optional[Path] = None) -> Figure:
        """
        Visualize AI analysis results.

        Args:
            predictions: Predicted values or labels
            true_labels: Optional true labels for comparison
            title: Plot title
            save_path: Optional save path

        Returns:
            Figure object
        """
        if true_labels is not None:
            # Confusion matrix or comparison
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

            # Predictions over time
            ax1.plot(predictions, 'o-', label='Predictions', markersize=4)
            ax1.plot(true_labels, 's-', label='True Labels', markersize=4, alpha=0.6)
            ax1.set_title('Predictions vs True Labels', fontweight='bold')
            ax1.set_xlabel('Sample Index')
            ax1.set_ylabel('Value')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Distribution comparison
            ax2.hist(predictions, bins=30, alpha=0.6, label='Predictions', color='blue')
            ax2.hist(true_labels, bins=30, alpha=0.6, label='True Labels', color='orange')
            ax2.set_title('Distribution Comparison', fontweight='bold')
            ax2.set_xlabel('Value')
            ax2.set_ylabel('Frequency')
            ax2.legend()

        else:
            # Just predictions
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(predictions, 'o-', markersize=4, color='#2E86AB')
            ax.set_title(title, fontweight='bold')
            ax.set_xlabel('Sample Index')
            ax.set_ylabel('Predicted Value')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig
