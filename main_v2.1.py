"""
SignalVizTool v2.1 - Professional Signal Analysis Tool

Integration of all advanced features:
- Advanced Signal Analysis (FFT, Wavelet, STFT, Filters)
- AI Machine Learning (Anomaly Detection, Classification, Prediction)
- Modern Visualization (Interactive Plots, Dashboards)

Author: EduCatCode - Integration Team
Version: 2.1.0
License: MIT
"""

import sys
from pathlib import Path

# Add src directory to path
src_dir = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_dir))

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import pandas as pd
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import threading

# Import all advanced modules
from src.core.advanced_analysis import (
    FFTAnalyzer, WaveletAnalyzer, STFTAnalyzer, FilterDesigner
)
from src.ai.ml_analyzer import (
    AnomalyDetector, SignalClassifier, PatternRecognition
)
from src.visualization.advanced_plotter import AdvancedPlotter
from src.utils.logger import LoggerSetup
from src.utils.config import config


class SignalVizToolV21:
    """Main application class integrating all v2.1 features."""

    def __init__(self, root):
        self.root = root
        self.root.title("SignalVizTool v2.1 - Professional Signal Analysis")
        self.root.geometry("1400x900")

        # Initialize logger
        self.logger = LoggerSetup.setup_logger(
            name='SignalVizTool_v2.1',
            level='INFO',
            log_dir=Path('logs'),
            console_logging=True
        )

        self.logger.info("="*70)
        self.logger.info("SignalVizTool v2.1 - Professional Edition Starting...")
        self.logger.info("="*70)

        # Current data
        self.current_file = None
        self.current_data = None
        self.current_signal = None
        self.sampling_rate = 10000  # Default

        # Setup UI
        self._setup_ui()

        self.logger.info("Application initialized successfully")

    def _setup_ui(self):
        """Setup the user interface."""
        # Configure style
        style = ttk.Style()
        style.theme_use('clam')

        # Main container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create notebook (tabbed interface)
        self.notebook = ttk.Notebook(main_container)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Tab 1: Data Loading & Basic Analysis
        self.tab_basic = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_basic, text=" 📁 Data & Basic ")

        # Tab 2: FFT Analysis
        self.tab_fft = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_fft, text=" 📊 FFT Analysis ")

        # Tab 3: Wavelet Analysis
        self.tab_wavelet = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_wavelet, text=" 🌊 Wavelet ")

        # Tab 4: Time-Frequency (STFT)
        self.tab_stft = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_stft, text=" ⏱️ STFT ")

        # Tab 5: Filters
        self.tab_filter = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_filter, text=" 🔧 Filters ")

        # Tab 6: AI Analysis
        self.tab_ai = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_ai, text=" 🤖 AI Analysis ")

        # Tab 7: Utilities (CSV Merge, Multi-file Plot)
        self.tab_utils = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_utils, text=" 🛠️ Utilities ")

        # Setup each tab
        self._setup_basic_tab()
        self._setup_fft_tab()
        self._setup_wavelet_tab()
        self._setup_stft_tab()
        self._setup_filter_tab()
        self._setup_ai_tab()
        self._setup_utils_tab()

        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("就緒 | 請載入數據開始分析")
        status_bar = ttk.Label(self.root, textvariable=self.status_var,
                              relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def _setup_basic_tab(self):
        """Setup basic data loading tab."""
        # Left panel - controls
        left_panel = ttk.LabelFrame(self.tab_basic, text="數據載入", padding=10)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        ttk.Button(left_panel, text="📁 選擇 CSV 檔案",
                  command=self.load_file, width=20).pack(pady=5)

        ttk.Label(left_panel, text="採樣率 (Hz):").pack(pady=(10, 0))
        self.sampling_rate_var = tk.StringVar(value="10000")
        ttk.Entry(left_panel, textvariable=self.sampling_rate_var,
                 width=20).pack(pady=5)

        ttk.Label(left_panel, text="選擇信號欄位:").pack(pady=(10, 0))
        self.column_var = tk.StringVar()
        self.column_combo = ttk.Combobox(left_panel, textvariable=self.column_var,
                                        state='readonly', width=18)
        self.column_combo.pack(pady=5)

        ttk.Button(left_panel, text="📈 繪製信號",
                  command=self.plot_signal, width=20).pack(pady=5)

        ttk.Separator(left_panel, orient='horizontal').pack(fill=tk.X, pady=10)

        ttk.Label(left_panel, text="快速測試:").pack(pady=5)
        ttk.Button(left_panel, text="🧪 載入 Demo 數據",
                  command=self.load_demo_data, width=20).pack(pady=5)

        # Right panel - display
        right_panel = ttk.Frame(self.tab_basic)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Info display
        info_frame = ttk.LabelFrame(right_panel, text="數據資訊", padding=10)
        info_frame.pack(fill=tk.X, padx=5, pady=5)

        self.info_text = scrolledtext.ScrolledText(info_frame, height=6, width=80)
        self.info_text.pack(fill=tk.X)

        # Plot area
        plot_frame = ttk.LabelFrame(right_panel, text="信號視覺化", padding=10)
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.basic_canvas_frame = ttk.Frame(plot_frame)
        self.basic_canvas_frame.pack(fill=tk.BOTH, expand=True)

    def _setup_fft_tab(self):
        """Setup FFT analysis tab."""
        # Controls
        control_frame = ttk.LabelFrame(self.tab_fft, text="FFT 分析控制", padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        ttk.Button(control_frame, text="計算 FFT",
                  command=self.compute_fft, width=20).pack(pady=5)

        ttk.Button(control_frame, text="功率譜密度 (PSD)",
                  command=self.compute_psd, width=20).pack(pady=5)

        ttk.Button(control_frame, text="尋找主頻率",
                  command=self.find_dominant_freq, width=20).pack(pady=5)

        ttk.Separator(control_frame, orient='horizontal').pack(fill=tk.X, pady=10)

        ttk.Label(control_frame, text="主頻率數量:").pack()
        self.n_peaks_var = tk.StringVar(value="5")
        ttk.Entry(control_frame, textvariable=self.n_peaks_var, width=20).pack(pady=5)

        # Results
        result_frame = ttk.Frame(self.tab_fft)
        result_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.fft_canvas_frame = ttk.Frame(result_frame)
        self.fft_canvas_frame.pack(fill=tk.BOTH, expand=True)

    def _setup_wavelet_tab(self):
        """Setup wavelet analysis tab."""
        control_frame = ttk.LabelFrame(self.tab_wavelet, text="小波分析控制", padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        ttk.Label(control_frame, text="小波類型:").pack()
        self.wavelet_var = tk.StringVar(value="db4")
        wavelet_combo = ttk.Combobox(control_frame, textvariable=self.wavelet_var,
                                     values=['db4', 'db8', 'sym4', 'coif1', 'morl'],
                                     state='readonly', width=18)
        wavelet_combo.pack(pady=5)

        ttk.Button(control_frame, text="連續小波轉換 (CWT)",
                  command=self.compute_cwt, width=20).pack(pady=5)

        ttk.Button(control_frame, text="離散小波轉換 (DWT)",
                  command=self.compute_dwt, width=20).pack(pady=5)

        ttk.Button(control_frame, text="小波去噪",
                  command=self.denoise_wavelet, width=20).pack(pady=5)

        result_frame = ttk.Frame(self.tab_wavelet)
        result_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.wavelet_canvas_frame = ttk.Frame(result_frame)
        self.wavelet_canvas_frame.pack(fill=tk.BOTH, expand=True)

    def _setup_stft_tab(self):
        """Setup STFT analysis tab."""
        control_frame = ttk.LabelFrame(self.tab_stft, text="時頻分析控制", padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        ttk.Button(control_frame, text="計算 STFT",
                  command=self.compute_stft, width=20).pack(pady=5)

        ttk.Button(control_frame, text="生成頻譜圖",
                  command=self.compute_spectrogram, width=20).pack(pady=5)

        ttk.Button(control_frame, text="綜合分析儀表板",
                  command=self.create_dashboard, width=20).pack(pady=5)

        result_frame = ttk.Frame(self.tab_stft)
        result_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.stft_canvas_frame = ttk.Frame(result_frame)
        self.stft_canvas_frame.pack(fill=tk.BOTH, expand=True)

    def _setup_filter_tab(self):
        """Setup filter design tab."""
        control_frame = ttk.LabelFrame(self.tab_filter, text="濾波器控制", padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        ttk.Label(control_frame, text="截止頻率 (Hz):").pack()
        self.cutoff_var = tk.StringVar(value="500")
        ttk.Entry(control_frame, textvariable=self.cutoff_var, width=20).pack(pady=5)

        ttk.Label(control_frame, text="濾波器階數:").pack()
        self.filter_order_var = tk.StringVar(value="4")
        ttk.Entry(control_frame, textvariable=self.filter_order_var, width=20).pack(pady=5)

        ttk.Button(control_frame, text="低通濾波",
                  command=self.apply_lowpass, width=20).pack(pady=5)

        ttk.Button(control_frame, text="高通濾波",
                  command=self.apply_highpass, width=20).pack(pady=5)

        result_frame = ttk.Frame(self.tab_filter)
        result_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.filter_canvas_frame = ttk.Frame(result_frame)
        self.filter_canvas_frame.pack(fill=tk.BOTH, expand=True)

    def _setup_ai_tab(self):
        """Setup AI analysis tab."""
        control_frame = ttk.LabelFrame(self.tab_ai, text="AI 分析控制", padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        ttk.Button(control_frame, text="🔍 異常檢測",
                  command=self.detect_anomalies, width=20).pack(pady=5)

        ttk.Button(control_frame, text="🎯 模式識別",
                  command=self.recognize_patterns, width=20).pack(pady=5)

        ttk.Separator(control_frame, orient='horizontal').pack(fill=tk.X, pady=10)

        ttk.Label(control_frame, text="異常檢測方法:").pack()
        self.anomaly_method_var = tk.StringVar(value="isolation_forest")
        method_combo = ttk.Combobox(control_frame, textvariable=self.anomaly_method_var,
                                    values=['isolation_forest', 'one_class_svm'],
                                    state='readonly', width=18)
        method_combo.pack(pady=5)

        result_frame = ttk.Frame(self.tab_ai)
        result_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.ai_canvas_frame = ttk.Frame(result_frame)
        self.ai_canvas_frame.pack(fill=tk.BOTH, expand=True)

    # Data loading methods
    def load_file(self):
        """Load CSV file."""
        filename = filedialog.askopenfilename(
            title="選擇 CSV 檔案",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )

        if not filename:
            return

        try:
            self.current_file = Path(filename)
            self.current_data = pd.read_csv(filename)

            # Update column selector
            columns = list(self.current_data.columns)
            self.column_combo['values'] = columns
            if columns:
                self.column_combo.current(0)

            # Update sampling rate if possible
            try:
                self.sampling_rate = int(self.sampling_rate_var.get())
            except:
                self.sampling_rate = 10000

            # Display info
            info = f"檔案: {self.current_file.name}\n"
            info += f"行數: {len(self.current_data)}\n"
            info += f"欄位: {', '.join(columns)}\n"
            info += f"採樣率: {self.sampling_rate} Hz\n"

            self.info_text.delete(1.0, tk.END)
            self.info_text.insert(1.0, info)

            self.status_var.set(f"已載入: {self.current_file.name}")
            self.logger.info(f"Loaded file: {filename}")

        except Exception as e:
            messagebox.showerror("錯誤", f"載入檔案失敗:\n{str(e)}")
            self.logger.error(f"Error loading file: {e}")

    def load_demo_data(self):
        """Load demo dataset."""
        demo_files = list(Path('demo_data').glob('*.csv')) if Path('demo_data').exists() else []

        if not demo_files:
            messagebox.showinfo("提示",
                              "找不到 Demo 數據！\n\n"
                              "請先執行 generate_demo_data.bat 生成測試數據。")
            return

        # Show selection dialog
        file_window = tk.Toplevel(self.root)
        file_window.title("選擇 Demo 數據集")
        file_window.geometry("500x400")

        ttk.Label(file_window, text="請選擇一個測試數據集:",
                 font=('Arial', 12, 'bold')).pack(pady=10)

        listbox = tk.Listbox(file_window, width=60, height=15)
        listbox.pack(padx=20, pady=10)

        for f in demo_files:
            listbox.insert(tk.END, f.name)

        def select_file():
            selection = listbox.curselection()
            if selection:
                selected_file = demo_files[selection[0]]
                file_window.destroy()
                # Load the selected file
                self.current_file = selected_file
                self.current_data = pd.read_csv(selected_file)

                columns = list(self.current_data.columns)
                self.column_combo['values'] = columns
                if columns:
                    self.column_combo.current(0)

                info = f"Demo 檔案: {self.current_file.name}\n"
                info += f"行數: {len(self.current_data)}\n"
                info += f"欄位: {', '.join(columns)}\n"

                self.info_text.delete(1.0, tk.END)
                self.info_text.insert(1.0, info)

                self.status_var.set(f"已載入 Demo: {self.current_file.name}")
                self.logger.info(f"Loaded demo file: {selected_file}")

        ttk.Button(file_window, text="選擇", command=select_file).pack(pady=10)

    def plot_signal(self):
        """Plot selected signal."""
        if self.current_data is None:
            messagebox.showwarning("警告", "請先載入數據！")
            return

        column = self.column_var.get()
        if not column:
            messagebox.showwarning("警告", "請選擇信號欄位！")
            return

        try:
            self.current_signal = self.current_data[column].values

            # Clear previous plot
            for widget in self.basic_canvas_frame.winfo_children():
                widget.destroy()

            # Plot
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(self.current_signal, linewidth=0.5, color='#2E86AB')
            ax.set_title(f'Signal: {column}', fontsize=14, fontweight='bold')
            ax.set_xlabel('Sample Index')
            ax.set_ylabel('Amplitude')
            ax.grid(True, alpha=0.3)

            canvas = FigureCanvasTkAgg(fig, self.basic_canvas_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            self.status_var.set(f"已繪製: {column} ({len(self.current_signal)} 樣本)")

        except Exception as e:
            messagebox.showerror("錯誤", f"繪製失敗:\n{str(e)}")

    # FFT methods
    def compute_fft(self):
        """Compute and plot FFT."""
        if self.current_signal is None:
            messagebox.showwarning("警告", "請先選擇並繪製信號！")
            return

        try:
            analyzer = FFTAnalyzer(self.sampling_rate, logger=self.logger)
            freqs, magnitude = analyzer.compute_fft(self.current_signal)

            # Clear previous
            for widget in self.fft_canvas_frame.winfo_children():
                widget.destroy()

            # Plot
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(freqs, magnitude, linewidth=2, color='#2E86AB')
            ax.fill_between(freqs, magnitude, alpha=0.3, color='#2E86AB')
            ax.set_title('FFT Analysis', fontsize=14, fontweight='bold')
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('Magnitude')
            ax.grid(True, alpha=0.3)

            canvas = FigureCanvasTkAgg(fig, self.fft_canvas_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            self.status_var.set("FFT 分析完成")

        except Exception as e:
            messagebox.showerror("錯誤", f"FFT 計算失敗:\n{str(e)}")

    def compute_psd(self):
        """Compute and plot PSD."""
        if self.current_signal is None:
            messagebox.showwarning("警告", "請先選擇並繪製信號！")
            return

        try:
            analyzer = FFTAnalyzer(self.sampling_rate, logger=self.logger)
            freqs, psd = analyzer.compute_psd(self.current_signal)

            for widget in self.fft_canvas_frame.winfo_children():
                widget.destroy()

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.semilogy(freqs, psd, linewidth=2, color='#A23B72')
            ax.set_title('Power Spectral Density', fontsize=14, fontweight='bold')
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('PSD')
            ax.grid(True, alpha=0.3)

            canvas = FigureCanvasTkAgg(fig, self.fft_canvas_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            self.status_var.set("PSD 分析完成")

        except Exception as e:
            messagebox.showerror("錯誤", f"PSD 計算失敗:\n{str(e)}")

    def find_dominant_freq(self):
        """Find and display dominant frequencies."""
        if self.current_signal is None:
            messagebox.showwarning("警告", "請先選擇並繪製信號！")
            return

        try:
            n_peaks = int(self.n_peaks_var.get())
            analyzer = FFTAnalyzer(self.sampling_rate, logger=self.logger)
            dominant_freqs = analyzer.find_dominant_frequencies(self.current_signal, n_peaks=n_peaks)

            # Show results
            result_window = tk.Toplevel(self.root)
            result_window.title("主頻率分析結果")
            result_window.geometry("400x300")

            text = scrolledtext.ScrolledText(result_window, width=50, height=15)
            text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

            text.insert(tk.END, f"找到 {len(dominant_freqs)} 個主頻率:\n\n")
            text.insert(tk.END, dominant_freqs.to_string(index=False))

            self.status_var.set(f"找到 {len(dominant_freqs)} 個主頻率")

        except Exception as e:
            messagebox.showerror("錯誤", f"主頻率分析失敗:\n{str(e)}")

    # Wavelet methods
    def compute_cwt(self):
        """Compute CWT."""
        if self.current_signal is None:
            messagebox.showwarning("警告", "請先選擇並繪製信號！")
            return

        try:
            wavelet = self.wavelet_var.get()
            analyzer = WaveletAnalyzer(logger=self.logger)

            # Use subset if signal too long
            signal_subset = self.current_signal[:5000] if len(self.current_signal) > 5000 else self.current_signal

            scales = np.arange(1, 128)
            coeffs, freqs = analyzer.compute_cwt(signal_subset, scales=scales, wavelet=wavelet)

            for widget in self.wavelet_canvas_frame.winfo_children():
                widget.destroy()

            plotter = AdvancedPlotter(logger=self.logger)
            fig = plotter.plot_wavelet_transform(coeffs, scales,
                                                title=f"Continuous Wavelet Transform ({wavelet})")

            canvas = FigureCanvasTkAgg(fig, self.wavelet_canvas_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            self.status_var.set("CWT 分析完成")

        except Exception as e:
            messagebox.showerror("錯誤", f"CWT 計算失敗:\n{str(e)}")

    def compute_dwt(self):
        """Compute DWT."""
        if self.current_signal is None:
            messagebox.showwarning("警告", "請先選擇並繪製信號！")
            return

        try:
            wavelet = self.wavelet_var.get()
            analyzer = WaveletAnalyzer(logger=self.logger)
            dwt_result = analyzer.compute_dwt(self.current_signal, wavelet=wavelet)

            # Show results
            result_window = tk.Toplevel(self.root)
            result_window.title("DWT 分析結果")
            result_window.geometry("500x400")

            text = scrolledtext.ScrolledText(result_window, width=60, height=20)
            text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

            text.insert(tk.END, f"離散小波轉換結果 (小波: {wavelet}):\n\n")
            for key, coeff in dwt_result.items():
                text.insert(tk.END, f"{key}: {len(coeff)} 係數\n")
                text.insert(tk.END, f"  均值: {np.mean(coeff):.4f}\n")
                text.insert(tk.END, f"  標準差: {np.std(coeff):.4f}\n\n")

            self.status_var.set("DWT 分析完成")

        except Exception as e:
            messagebox.showerror("錯誤", f"DWT 計算失敗:\n{str(e)}")

    def denoise_wavelet(self):
        """Denoise signal using wavelets."""
        if self.current_signal is None:
            messagebox.showwarning("警告", "請先選擇並繪製信號！")
            return

        try:
            wavelet = self.wavelet_var.get()
            analyzer = WaveletAnalyzer(logger=self.logger)
            denoised = analyzer.denoise_signal(self.current_signal, wavelet=wavelet)

            for widget in self.wavelet_canvas_frame.winfo_children():
                widget.destroy()

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

            ax1.plot(self.current_signal[:1000], alpha=0.7, color='gray', label='Original')
            ax1.set_title('Original Signal (first 1000 samples)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            ax2.plot(denoised[:1000], color='#2E86AB', label='Denoised')
            ax2.set_title('Denoised Signal')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()

            canvas = FigureCanvasTkAgg(fig, self.wavelet_canvas_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            self.status_var.set("小波去噪完成")

        except Exception as e:
            messagebox.showerror("錯誤", f"去噪失敗:\n{str(e)}")

    # STFT methods
    def compute_stft(self):
        """Compute STFT."""
        if self.current_signal is None:
            messagebox.showwarning("警告", "請先選擇並繪製信號！")
            return

        try:
            analyzer = STFTAnalyzer(self.sampling_rate, logger=self.logger)
            freqs, times, Zxx = analyzer.compute_stft(self.current_signal[:10000])

            for widget in self.stft_canvas_frame.winfo_children():
                widget.destroy()

            plotter = AdvancedPlotter(logger=self.logger)
            fig = plotter.plot_spectrogram(freqs, times, np.abs(Zxx)**2,
                                          title="STFT Spectrogram")

            canvas = FigureCanvasTkAgg(fig, self.stft_canvas_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            self.status_var.set("STFT 分析完成")

        except Exception as e:
            messagebox.showerror("錯誤", f"STFT 計算失敗:\n{str(e)}")

    def compute_spectrogram(self):
        """Compute spectrogram."""
        if self.current_signal is None:
            messagebox.showwarning("警告", "請先選擇並繪製信號！")
            return

        try:
            analyzer = STFTAnalyzer(self.sampling_rate, logger=self.logger)
            freqs, times, Sxx = analyzer.compute_spectrogram(self.current_signal[:10000])

            for widget in self.stft_canvas_frame.winfo_children():
                widget.destroy()

            plotter = AdvancedPlotter(logger=self.logger)
            fig = plotter.plot_spectrogram(freqs, times, Sxx,
                                          title="Spectrogram")

            canvas = FigureCanvasTkAgg(fig, self.stft_canvas_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            self.status_var.set("頻譜圖生成完成")

        except Exception as e:
            messagebox.showerror("錯誤", f"頻譜圖生成失敗:\n{str(e)}")

    def create_dashboard(self):
        """Create comprehensive analysis dashboard."""
        if self.current_signal is None:
            messagebox.showwarning("警告", "請先選擇並繪製信號！")
            return

        messagebox.showinfo("提示", "正在生成綜合分析儀表板，請稍候...")

        def generate_dashboard():
            try:
                signal_subset = self.current_signal[:5000]

                # FFT
                fft_analyzer = FFTAnalyzer(self.sampling_rate, logger=self.logger)
                freqs, fft_mag = fft_analyzer.compute_fft(signal_subset)

                # STFT
                stft_analyzer = STFTAnalyzer(self.sampling_rate, logger=self.logger)
                freqs_stft, times_stft, Sxx = stft_analyzer.compute_spectrogram(signal_subset)

                # Create dashboard
                plotter = AdvancedPlotter(logger=self.logger)
                fig = plotter.create_analysis_dashboard(
                    signal_data=signal_subset,
                    freqs=freqs,
                    fft_magnitude=fft_mag,
                    times_stft=times_stft,
                    freqs_stft=freqs_stft,
                    spectrogram=Sxx,
                    title="Signal Analysis Dashboard"
                )

                # Display
                self.root.after(0, lambda: self._display_dashboard(fig))

            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("錯誤", f"儀表板生成失敗:\n{str(e)}"))

        threading.Thread(target=generate_dashboard, daemon=True).start()

    def _display_dashboard(self, fig):
        """Display dashboard in STFT tab."""
        for widget in self.stft_canvas_frame.winfo_children():
            widget.destroy()

        canvas = FigureCanvasTkAgg(fig, self.stft_canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.status_var.set("綜合分析儀表板已生成")

    # Filter methods
    def apply_lowpass(self):
        """Apply lowpass filter."""
        if self.current_signal is None:
            messagebox.showwarning("警告", "請先選擇並繪製信號！")
            return

        try:
            cutoff = float(self.cutoff_var.get())
            order = int(self.filter_order_var.get())

            designer = FilterDesigner(self.sampling_rate, logger=self.logger)
            filtered = designer.design_and_apply_lowpass(self.current_signal, cutoff, order)

            for widget in self.filter_canvas_frame.winfo_children():
                widget.destroy()

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

            ax1.plot(self.current_signal[:1000], alpha=0.7, label='Original')
            ax1.set_title('Original Signal')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            ax2.plot(filtered[:1000], color='#2E86AB', label=f'Lowpass ({cutoff} Hz)')
            ax2.set_title('Filtered Signal')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()

            canvas = FigureCanvasTkAgg(fig, self.filter_canvas_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            self.status_var.set(f"低通濾波完成 (截止: {cutoff} Hz)")

        except Exception as e:
            messagebox.showerror("錯誤", f"濾波失敗:\n{str(e)}")

    def apply_highpass(self):
        """Apply highpass filter."""
        if self.current_signal is None:
            messagebox.showwarning("警告", "請先選擇並繪製信號！")
            return

        try:
            cutoff = float(self.cutoff_var.get())
            order = int(self.filter_order_var.get())

            designer = FilterDesigner(self.sampling_rate, logger=self.logger)
            filtered = designer.design_and_apply_highpass(self.current_signal, cutoff, order)

            for widget in self.filter_canvas_frame.winfo_children():
                widget.destroy()

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

            ax1.plot(self.current_signal[:1000], alpha=0.7, label='Original')
            ax1.set_title('Original Signal')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            ax2.plot(filtered[:1000], color='#A23B72', label=f'Highpass ({cutoff} Hz)')
            ax2.set_title('Filtered Signal')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()

            canvas = FigureCanvasTkAgg(fig, self.filter_canvas_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            self.status_var.set(f"高通濾波完成 (截止: {cutoff} Hz)")

        except Exception as e:
            messagebox.showerror("錯誤", f"濾波失敗:\n{str(e)}")

    # AI methods
    def detect_anomalies(self):
        """Detect anomalies using AI."""
        if self.current_signal is None:
            messagebox.showwarning("警告", "請先選擇並繪製信號！")
            return

        try:
            method = self.anomaly_method_var.get()
            detector = AnomalyDetector(method=method, logger=self.logger)
            anomaly_df = detector.detect_signal_anomalies(self.current_signal, window_size=100)

            # Count anomalies
            n_anomalies = anomaly_df['Is_Anomaly'].sum()

            # Show results
            result_window = tk.Toplevel(self.root)
            result_window.title("異常檢測結果")
            result_window.geometry("600x400")

            text = scrolledtext.ScrolledText(result_window, width=70, height=20)
            text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

            text.insert(tk.END, f"使用方法: {method}\n")
            text.insert(tk.END, f"檢測到 {n_anomalies} 個異常點 (共 {len(anomaly_df)} 個視窗)\n\n")
            text.insert(tk.END, "異常點位置:\n")
            text.insert(tk.END, anomaly_df[anomaly_df['Is_Anomaly']].to_string(index=False))

            self.status_var.set(f"異常檢測完成: 找到 {n_anomalies} 個異常")

        except Exception as e:
            messagebox.showerror("錯誤", f"異常檢測失敗:\n{str(e)}")

    def recognize_patterns(self):
        """Recognize patterns using clustering."""
        if self.current_signal is None:
            messagebox.showwarning("警告", "請先選擇並繪製信號！")
            return

        try:
            recognizer = PatternRecognition(n_clusters=3, logger=self.logger)
            pattern_df = recognizer.analyze_signal_patterns(self.current_signal, window_size=100)

            # Show results
            result_window = tk.Toplevel(self.root)
            result_window.title("模式識別結果")
            result_window.geometry("600x400")

            text = scrolledtext.ScrolledText(result_window, width=70, height=20)
            text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

            text.insert(tk.END, "識別到 3 種模式:\n\n")

            for pattern_id in range(3):
                count = (pattern_df['Pattern_ID'] == pattern_id).sum()
                percentage = (count / len(pattern_df)) * 100
                text.insert(tk.END, f"模式 {pattern_id}: {count} 個視窗 ({percentage:.1f}%)\n")

            text.insert(tk.END, "\n詳細結果:\n")
            text.insert(tk.END, pattern_df.head(20).to_string(index=False))

            self.status_var.set("模式識別完成")

        except Exception as e:
            messagebox.showerror("錯誤", f"模式識別失敗:\n{str(e)}")

    def _setup_utils_tab(self):
        """Setup utilities tab for CSV merge and multi-file plotting."""
        # Main container
        container = ttk.Frame(self.tab_utils)
        container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Title
        title_label = ttk.Label(container, text="實用工具 - CSV 操作、批次繪圖與工序分析",
                               font=('Arial', 14, 'bold'))
        title_label.pack(pady=10)

        # Description
        desc_text = """
        本工具提供原始版本的實用功能與製造業專業分析：
        • 合併多個 CSV 檔案（簡單串接）
        • 依時間戳對齊合併兩個 CSV 檔案
        • 選擇多個檔案並批次繪圖
        • 工序分析（加工訊號自動分段與分析）
        """
        desc_label = ttk.Label(container, text=desc_text, justify=tk.LEFT)
        desc_label.pack(pady=5)

        ttk.Separator(container, orient='horizontal').pack(fill=tk.X, pady=10)

        # Create three sections
        # Section 1: Simple CSV Merge
        merge_frame = ttk.LabelFrame(container, text="1. 合併多個 CSV 檔案", padding=15)
        merge_frame.pack(fill=tk.X, padx=20, pady=10)

        merge_desc = ttk.Label(merge_frame,
                              text="將多個 CSV 檔案簡單串接合併成一個檔案（垂直堆疊）",
                              foreground='gray')
        merge_desc.pack(pady=5)

        ttk.Button(merge_frame, text="🔗 選擇檔案並合併",
                  command=self.merge_csv_files,
                  width=30).pack(pady=10)

        # Section 2: Align and Merge by Timestamp
        align_frame = ttk.LabelFrame(container, text="2. 依時間戳對齊合併", padding=15)
        align_frame.pack(fill=tk.X, padx=20, pady=10)

        align_desc = ttk.Label(align_frame,
                              text="將兩個 CSV 檔案依照時間戳對齊後合併（適用於不同感測器數據）",
                              foreground='gray')
        align_desc.pack(pady=5)

        ttk.Button(align_frame, text="⏱️ 選擇兩個檔案對齊合併",
                  command=self.align_merge_csv,
                  width=30).pack(pady=10)

        # Section 3: Multi-file Plotting
        plot_frame = ttk.LabelFrame(container, text="3. 批次繪圖", padding=15)
        plot_frame.pack(fill=tk.X, padx=20, pady=10)

        plot_desc = ttk.Label(plot_frame,
                             text="選擇多個 CSV 檔案，為每個檔案生成圖表並保存",
                             foreground='gray')
        plot_desc.pack(pady=5)

        ttk.Button(plot_frame, text="📊 選擇多個檔案並繪圖",
                  command=self.multi_file_plot,
                  width=30).pack(pady=10)

        # Section 4: Process Analysis
        process_frame = ttk.LabelFrame(container, text="4. 工序分析", padding=15)
        process_frame.pack(fill=tk.X, padx=20, pady=10)

        process_desc = ttk.Label(process_frame,
                                text="自動識別加工訊號中的待機與工序，並進行詳細分析",
                                foreground='gray')
        process_desc.pack(pady=5)

        # Parameters input
        param_frame = ttk.Frame(process_frame)
        param_frame.pack(fill=tk.X, pady=5)

        # Sampling rate
        ttk.Label(param_frame, text="取樣率 (Hz):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=3)
        self.process_sampling_rate = tk.StringVar(value="10000")
        ttk.Entry(param_frame, textvariable=self.process_sampling_rate, width=15).grid(row=0, column=1, padx=5, pady=3)

        # Idle threshold
        ttk.Label(param_frame, text="待機閾值:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=3)
        self.process_idle_threshold = tk.StringVar(value="0.5")
        ttk.Entry(param_frame, textvariable=self.process_idle_threshold, width=15).grid(row=1, column=1, padx=5, pady=3)

        # Idle duration
        ttk.Label(param_frame, text="待機秒數 (s):").grid(row=2, column=0, sticky=tk.W, padx=5, pady=3)
        self.process_idle_duration = tk.StringVar(value="2.0")
        ttk.Entry(param_frame, textvariable=self.process_idle_duration, width=15).grid(row=2, column=1, padx=5, pady=3)

        ttk.Button(process_frame, text="🔬 開始工序分析",
                  command=self.process_analysis,
                  width=30).pack(pady=10)

        # Progress bar
        self.utils_progress_var = tk.DoubleVar()
        self.utils_progress = ttk.Progressbar(container,
                                             orient="horizontal",
                                             length=400,
                                             mode="determinate",
                                             variable=self.utils_progress_var)
        self.utils_progress.pack(pady=20)

        # Info label
        info_text = """
        💡 提示：
        • 合併功能會保留所有欄位
        • 時間戳對齊適用於需要同步不同數據源的場景
        • 批次繪圖會自動保存圖表為 PNG 檔案
        • 工序分析需先在 Tab 1 載入加工訊號數據
        """
        info_label = ttk.Label(container, text=info_text,
                             justify=tk.LEFT, foreground='blue')
        info_label.pack(pady=10)

    def merge_csv_files(self):
        """Simple concatenation of multiple CSV files."""
        try:
            # Select multiple files
            file_paths = filedialog.askopenfilenames(
                title="選擇要合併的 CSV 檔案",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )

            if not file_paths or len(file_paths) < 2:
                messagebox.showinfo("提示", "請至少選擇兩個檔案")
                return

            # Ask for save location
            save_path = filedialog.asksaveasfilename(
                title="儲存合併後的檔案",
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv")]
            )

            if not save_path:
                return

            # Perform merge in thread
            def merge_thread():
                try:
                    self.status_var.set(f"正在合併 {len(file_paths)} 個檔案...")
                    self.utils_progress_var.set(0)

                    combined_df = pd.DataFrame()
                    n_total = len(file_paths)

                    for i, file_path in enumerate(file_paths):
                        df = pd.read_csv(file_path)
                        combined_df = pd.concat([combined_df, df], ignore_index=True)

                        # Update progress
                        progress = ((i + 1) / n_total) * 100
                        self.utils_progress_var.set(progress)
                        self.logger.info(f"已處理: {Path(file_path).name}")

                    # Save merged file
                    combined_df.to_csv(save_path, index=False)

                    self.utils_progress_var.set(100)
                    self.status_var.set("合併完成")

                    messagebox.showinfo("成功",
                                      f"已成功合併 {len(file_paths)} 個檔案\n"
                                      f"總計 {len(combined_df)} 筆記錄\n"
                                      f"儲存於: {save_path}")

                    self.logger.info(f"CSV merge completed: {save_path}")

                except Exception as e:
                    self.logger.error(f"CSV merge failed: {str(e)}")
                    messagebox.showerror("錯誤", f"合併失敗:\n{str(e)}")
                finally:
                    self.utils_progress_var.set(0)

            threading.Thread(target=merge_thread, daemon=True).start()

        except Exception as e:
            messagebox.showerror("錯誤", f"操作失敗:\n{str(e)}")

    def align_merge_csv(self):
        """Align and merge two CSV files by timestamp."""
        try:
            # Select first file (e.g., vibration data)
            file1 = filedialog.askopenfilename(
                title="選擇第一個 CSV 檔案（例如：振動數據）",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )

            if not file1:
                return

            # Select second file (e.g., current data)
            file2 = filedialog.askopenfilename(
                title="選擇第二個 CSV 檔案（例如：電流數據）",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )

            if not file2:
                return

            # Ask for timestamp column name
            timestamp_col = tk.simpledialog.askstring(
                "時間戳欄位",
                "請輸入時間戳欄位名稱:",
                initialvalue="timestamp"
            )

            if not timestamp_col:
                messagebox.showinfo("取消", "操作已取消")
                return

            # Ask for save location
            save_path = filedialog.asksaveasfilename(
                title="儲存對齊合併後的檔案",
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv")]
            )

            if not save_path:
                return

            # Perform alignment in thread
            def align_thread():
                try:
                    self.status_var.set("正在對齊並合併檔案...")
                    self.utils_progress_var.set(25)

                    # Load files
                    df1 = pd.read_csv(file1)
                    df2 = pd.read_csv(file2)

                    self.utils_progress_var.set(50)

                    # Convert timestamp to datetime and set as index
                    df1[timestamp_col] = pd.to_datetime(df1[timestamp_col])
                    df2[timestamp_col] = pd.to_datetime(df2[timestamp_col])

                    df1 = df1.set_index(timestamp_col)
                    df2 = df2.set_index(timestamp_col)

                    # Find overlapping time range
                    start_time = max(df1.index.min(), df2.index.min())
                    end_time = min(df1.index.max(), df2.index.max())

                    # Filter to overlapping range
                    df1_filtered = df1[(df1.index >= start_time) & (df1.index <= end_time)]
                    df2_filtered = df2[(df2.index >= start_time) & (df2.index <= end_time)]

                    self.utils_progress_var.set(75)

                    # Merge using nearest timestamp
                    merged_df = pd.merge_asof(
                        df1_filtered.sort_index(),
                        df2_filtered.sort_index(),
                        left_index=True,
                        right_index=True,
                        direction='nearest',
                        suffixes=('_file1', '_file2')
                    )

                    # Reset index to save timestamp as column
                    merged_df = merged_df.reset_index()

                    # Save
                    merged_df.to_csv(save_path, index=False)

                    self.utils_progress_var.set(100)
                    self.status_var.set("對齊合併完成")

                    messagebox.showinfo("成功",
                                      f"已成功對齊並合併兩個檔案\n"
                                      f"時間範圍: {start_time} ~ {end_time}\n"
                                      f"總計 {len(merged_df)} 筆記錄\n"
                                      f"儲存於: {save_path}")

                    self.logger.info(f"Aligned merge completed: {save_path}")

                except Exception as e:
                    self.logger.error(f"Aligned merge failed: {str(e)}")
                    messagebox.showerror("錯誤", f"對齊合併失敗:\n{str(e)}")
                finally:
                    self.utils_progress_var.set(0)

            threading.Thread(target=align_thread, daemon=True).start()

        except Exception as e:
            messagebox.showerror("錯誤", f"操作失敗:\n{str(e)}")

    def multi_file_plot(self):
        """Plot multiple CSV files and save charts."""
        try:
            # Select multiple files
            file_paths = filedialog.askopenfilenames(
                title="選擇要繪圖的 CSV 檔案",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )

            if not file_paths:
                return

            # Find common columns across all files
            common_columns = set(pd.read_csv(file_paths[0]).columns)
            for file_path in file_paths[1:]:
                df = pd.read_csv(file_path)
                common_columns.intersection_update(df.columns)

            # Remove timestamp-like columns
            common_columns = [col for col in common_columns
                            if not any(keyword in col.lower()
                                     for keyword in ['time', 'date', 'index'])]

            if not common_columns:
                messagebox.showwarning("警告", "所選檔案沒有共同的數值欄位")
                return

            # Let user select columns to plot
            selected_cols = self._select_columns_dialog(list(common_columns))

            if not selected_cols:
                return

            # Ask for output directory
            output_dir = filedialog.askdirectory(title="選擇圖表輸出目錄")

            if not output_dir:
                return

            output_dir = Path(output_dir)

            # Perform plotting in thread
            def plot_thread():
                try:
                    self.status_var.set(f"正在為 {len(file_paths)} 個檔案生成圖表...")
                    self.utils_progress_var.set(0)

                    n_total = len(file_paths)
                    saved_files = []

                    for i, file_path in enumerate(file_paths):
                        # Load data
                        df = pd.read_csv(file_path)
                        file_name = Path(file_path).stem

                        # Create plot
                        fig, ax = plt.subplots(figsize=(12, 6))

                        for col in selected_cols:
                            if col in df.columns:
                                ax.plot(df[col], label=col, linewidth=1.5)

                        ax.set_title(f'{file_name}', fontsize=14, fontweight='bold')
                        ax.set_xlabel('樣本數', fontsize=12)
                        ax.set_ylabel('振幅', fontsize=12)
                        ax.legend(loc='best')
                        ax.grid(True, alpha=0.3)

                        # Save figure
                        output_path = output_dir / f"{file_name}.png"
                        fig.savefig(output_path, dpi=150, bbox_inches='tight')
                        plt.close(fig)

                        saved_files.append(output_path)

                        # Update progress
                        progress = ((i + 1) / n_total) * 100
                        self.utils_progress_var.set(progress)
                        self.logger.info(f"已繪製: {file_name}.png")

                    self.utils_progress_var.set(100)
                    self.status_var.set("批次繪圖完成")

                    messagebox.showinfo("成功",
                                      f"已成功生成 {len(saved_files)} 個圖表\n"
                                      f"儲存於: {output_dir}\n\n"
                                      f"繪製的欄位: {', '.join(selected_cols)}")

                    self.logger.info(f"Multi-file plotting completed: {len(saved_files)} charts")

                except Exception as e:
                    self.logger.error(f"Multi-file plotting failed: {str(e)}")
                    messagebox.showerror("錯誤", f"批次繪圖失敗:\n{str(e)}")
                finally:
                    self.utils_progress_var.set(0)

            threading.Thread(target=plot_thread, daemon=True).start()

        except Exception as e:
            messagebox.showerror("錯誤", f"操作失敗:\n{str(e)}")

    def _select_columns_dialog(self, columns):
        """Dialog to select columns for plotting."""
        dialog = tk.Toplevel(self.root)
        dialog.title("選擇要繪製的欄位")
        dialog.geometry("400x500")
        dialog.transient(self.root)
        dialog.grab_set()

        # Instructions
        ttk.Label(dialog, text="請選擇要繪製的欄位：",
                 font=('Arial', 11, 'bold')).pack(pady=10)

        # Scrollable frame for checkboxes
        canvas = tk.Canvas(dialog)
        scrollbar = ttk.Scrollbar(dialog, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Create checkboxes
        selected_vars = {}
        for col in columns:
            var = tk.BooleanVar(value=True)  # Default: all selected
            cb = ttk.Checkbutton(scrollable_frame, text=col, variable=var)
            cb.pack(anchor='w', padx=20, pady=2)
            selected_vars[col] = var

        canvas.pack(side="left", fill="both", expand=True, padx=10)
        scrollbar.pack(side="right", fill="y")

        # Buttons
        button_frame = ttk.Frame(dialog)
        button_frame.pack(fill=tk.X, padx=10, pady=10)

        selected_columns = []

        def on_confirm():
            nonlocal selected_columns
            selected_columns = [col for col, var in selected_vars.items() if var.get()]
            if not selected_columns:
                messagebox.showwarning("警告", "請至少選擇一個欄位")
                return
            dialog.destroy()

        def on_cancel():
            dialog.destroy()

        ttk.Button(button_frame, text="確認", command=on_confirm).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="取消", command=on_cancel).pack(side=tk.LEFT, padx=5)

        # Wait for dialog to close
        self.root.wait_window(dialog)

        return selected_columns

    def process_analysis(self):
        """
        Manufacturing process analysis: automatically identify idle periods
        and process steps in machining signals.
        """
        try:
            # Check if data is loaded
            if self.current_signal is None:
                messagebox.showwarning("警告", "請先在 Tab 1 載入訊號數據")
                return

            # Get parameters
            try:
                sampling_rate = float(self.process_sampling_rate.get())
                idle_threshold = float(self.process_idle_threshold.get())
                idle_duration = float(self.process_idle_duration.get())
            except ValueError:
                messagebox.showerror("錯誤", "參數格式錯誤，請輸入有效數值")
                return

            if sampling_rate <= 0 or idle_duration <= 0:
                messagebox.showerror("錯誤", "取樣率和待機秒數必須大於 0")
                return

            # Perform analysis in thread
            def analysis_thread():
                try:
                    self.status_var.set("正在進行工序分析...")
                    self.utils_progress_var.set(10)

                    signal = self.current_signal.copy()
                    n_samples = len(signal)

                    # Step 1: Identify idle periods
                    idle_samples = int(idle_duration * sampling_rate)
                    is_idle = np.abs(signal) < idle_threshold

                    # Find continuous idle periods
                    idle_periods = []
                    in_idle = False
                    idle_start = 0

                    for i in range(n_samples):
                        if is_idle[i] and not in_idle:
                            idle_start = i
                            in_idle = True
                        elif not is_idle[i] and in_idle:
                            if i - idle_start >= idle_samples:
                                idle_periods.append((idle_start, i))
                            in_idle = False

                    # Check last period
                    if in_idle and n_samples - idle_start >= idle_samples:
                        idle_periods.append((idle_start, n_samples))

                    self.utils_progress_var.set(30)
                    self.logger.info(f"Found {len(idle_periods)} idle periods")

                    # Step 2: Extract machining segments
                    machining_segments = []

                    if len(idle_periods) == 0:
                        # No idle period, entire signal is one machining segment
                        machining_segments.append((0, n_samples))
                    else:
                        # Before first idle
                        if idle_periods[0][0] > 0:
                            machining_segments.append((0, idle_periods[0][0]))

                        # Between idle periods
                        for i in range(len(idle_periods) - 1):
                            start = idle_periods[i][1]
                            end = idle_periods[i + 1][0]
                            if end > start:
                                machining_segments.append((start, end))

                        # After last idle
                        if idle_periods[-1][1] < n_samples:
                            machining_segments.append((idle_periods[-1][1], n_samples))

                    self.utils_progress_var.set(50)
                    self.logger.info(f"Found {len(machining_segments)} machining segments")

                    # Step 3: For each machining segment, find process steps
                    all_process_steps = []

                    for seg_idx, (seg_start, seg_end) in enumerate(machining_segments):
                        segment_signal = signal[seg_start:seg_end]

                        # Calculate 99.9 percentile as step threshold
                        step_threshold = np.percentile(np.abs(segment_signal), 99.9)

                        # Find points exceeding threshold (process step starts)
                        step_starts = []
                        above_threshold = np.abs(segment_signal) > step_threshold

                        for i in range(1, len(above_threshold)):
                            if above_threshold[i] and not above_threshold[i-1]:
                                step_starts.append(seg_start + i)

                        # If no steps found, treat entire segment as one step
                        if len(step_starts) == 0:
                            step_starts = [seg_start]

                        # Calculate step durations
                        step_info = []
                        for i in range(len(step_starts)):
                            step_start_idx = step_starts[i]
                            if i < len(step_starts) - 1:
                                step_end_idx = step_starts[i + 1]
                            else:
                                step_end_idx = seg_end

                            step_duration = (step_end_idx - step_start_idx) / sampling_rate
                            step_info.append({
                                'segment': seg_idx,
                                'step': i,
                                'start_idx': step_start_idx,
                                'end_idx': step_end_idx,
                                'duration': step_duration
                            })

                        all_process_steps.extend(step_info)

                    self.utils_progress_var.set(70)

                    # Step 4: Create visualization
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

                    # Plot 1: Full signal with idle and machining periods
                    time_axis = np.arange(n_samples) / sampling_rate
                    ax1.plot(time_axis, signal, 'b-', linewidth=0.8, alpha=0.7, label='訊號')

                    # Mark idle periods
                    for idle_start, idle_end in idle_periods:
                        ax1.axvspan(idle_start/sampling_rate, idle_end/sampling_rate,
                                   alpha=0.3, color='gray', label='待機' if idle_start == idle_periods[0][0] else '')

                    # Mark machining segments
                    for seg_idx, (seg_start, seg_end) in enumerate(machining_segments):
                        ax1.axvspan(seg_start/sampling_rate, seg_end/sampling_rate,
                                   alpha=0.2, color='green',
                                   label='加工' if seg_idx == 0 else '')

                    ax1.axhline(y=idle_threshold, color='r', linestyle='--',
                               linewidth=1.5, label=f'待機閾值 ({idle_threshold})')
                    ax1.axhline(y=-idle_threshold, color='r', linestyle='--', linewidth=1.5)

                    ax1.set_xlabel('時間 (秒)', fontsize=12)
                    ax1.set_ylabel('振幅', fontsize=12)
                    ax1.set_title('工序分析 - 待機與加工識別', fontsize=14, fontweight='bold')
                    ax1.legend(loc='upper right')
                    ax1.grid(True, alpha=0.3)

                    # Plot 2: Detailed process steps
                    ax2.plot(time_axis, signal, 'b-', linewidth=0.8, alpha=0.7)

                    # Draw red vertical lines at process step starts
                    step_lines_drawn = False
                    for step in all_process_steps:
                        ax2.axvline(x=step['start_idx']/sampling_rate,
                                   color='red', linestyle='-', linewidth=2,
                                   label='工序開始' if not step_lines_drawn else '')
                        step_lines_drawn = True

                    ax2.set_xlabel('時間 (秒)', fontsize=12)
                    ax2.set_ylabel('振幅', fontsize=12)
                    ax2.set_title('工序分析 - 工序分割詳細視圖', fontsize=14, fontweight='bold')
                    ax2.legend(loc='upper right')
                    ax2.grid(True, alpha=0.3)

                    plt.tight_layout()

                    # Save figure
                    output_path = Path('process_analysis_result.png')
                    fig.savefig(output_path, dpi=150, bbox_inches='tight')

                    self.utils_progress_var.set(90)

                    # Generate analysis report
                    report = self._generate_process_report(
                        machining_segments, all_process_steps,
                        sampling_rate, n_samples
                    )

                    self.utils_progress_var.set(100)
                    self.status_var.set("工序分析完成")

                    # Show report window
                    self._show_process_report(report, fig, output_path)

                    self.logger.info("Process analysis completed successfully")

                except Exception as e:
                    self.logger.error(f"Process analysis failed: {str(e)}")
                    messagebox.showerror("錯誤", f"工序分析失敗:\n{str(e)}")
                finally:
                    self.utils_progress_var.set(0)

            threading.Thread(target=analysis_thread, daemon=True).start()

        except Exception as e:
            messagebox.showerror("錯誤", f"啟動分析失敗:\n{str(e)}")

    def _generate_process_report(self, machining_segments, process_steps,
                                 sampling_rate, total_samples):
        """Generate detailed process analysis report."""
        report = []
        report.append("=" * 70)
        report.append("工序分析報告")
        report.append("=" * 70)
        report.append("")

        # Overall statistics
        total_duration = total_samples / sampling_rate
        report.append(f"📊 總體統計")
        report.append(f"   總訊號長度: {total_duration:.2f} 秒")
        report.append(f"   總樣本數: {total_samples:,}")
        report.append(f"   取樣率: {sampling_rate:,} Hz")
        report.append("")

        # Machining segments
        report.append(f"🔧 加工段分析")
        report.append(f"   加工段數量: {len(machining_segments)}")
        report.append("")

        total_machining_time = 0
        for idx, (start, end) in enumerate(machining_segments):
            duration = (end - start) / sampling_rate
            total_machining_time += duration

            # Count steps in this segment
            steps_in_segment = [s for s in process_steps if s['segment'] == idx]

            report.append(f"   加工段 {idx + 1}:")
            report.append(f"      時間範圍: {start/sampling_rate:.2f}s ~ {end/sampling_rate:.2f}s")
            report.append(f"      持續時間: {duration:.2f} 秒")
            report.append(f"      包含工序數: {len(steps_in_segment)}")
            report.append("")

        report.append(f"   總加工時間: {total_machining_time:.2f} 秒")
        report.append(f"   加工時間佔比: {(total_machining_time/total_duration)*100:.1f}%")
        report.append("")

        # Process steps
        report.append(f"⚙️ 工序詳細分析")
        report.append(f"   總工序數量: {len(process_steps)}")
        report.append("")

        for step in process_steps:
            report.append(f"   工序 {step['segment']+1}-{step['step']+1}:")
            report.append(f"      開始時間: {step['start_idx']/sampling_rate:.2f} 秒")
            report.append(f"      結束時間: {step['end_idx']/sampling_rate:.2f} 秒")
            report.append(f"      持續時間: {step['duration']:.2f} 秒")
            report.append("")

        # Summary
        if len(process_steps) > 0:
            avg_step_duration = np.mean([s['duration'] for s in process_steps])
            min_step_duration = np.min([s['duration'] for s in process_steps])
            max_step_duration = np.max([s['duration'] for s in process_steps])

            report.append(f"📈 工序統計")
            report.append(f"   平均工序時長: {avg_step_duration:.2f} 秒")
            report.append(f"   最短工序時長: {min_step_duration:.2f} 秒")
            report.append(f"   最長工序時長: {max_step_duration:.2f} 秒")
            report.append("")

        report.append("=" * 70)

        return "\n".join(report)

    def _show_process_report(self, report_text, fig, image_path):
        """Display process analysis report in a new window."""
        report_window = tk.Toplevel(self.root)
        report_window.title("工序分析報告")
        report_window.geometry("900x700")

        # Create notebook for tabs
        notebook = ttk.Notebook(report_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Tab 1: Report text
        report_tab = ttk.Frame(notebook)
        notebook.add(report_tab, text="📊 分析報告")

        report_text_widget = scrolledtext.ScrolledText(report_tab,
                                                       width=100, height=35,
                                                       font=('Courier', 10))
        report_text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        report_text_widget.insert(tk.END, report_text)
        report_text_widget.config(state=tk.DISABLED)

        # Tab 2: Visualization
        viz_tab = ttk.Frame(notebook)
        notebook.add(viz_tab, text="📈 視覺化圖表")

        canvas = FigureCanvasTkAgg(fig, master=viz_tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Buttons
        button_frame = ttk.Frame(report_window)
        button_frame.pack(fill=tk.X, padx=10, pady=10)

        def save_report():
            file_path = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
            )
            if file_path:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(report_text)
                messagebox.showinfo("成功", f"報告已儲存至:\n{file_path}")

        ttk.Button(button_frame, text="💾 儲存報告",
                  command=save_report).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="關閉",
                  command=report_window.destroy).pack(side=tk.LEFT, padx=5)

        # Info label
        info_label = ttk.Label(button_frame,
                              text=f"圖表已儲存: {image_path}",
                              foreground='blue')
        info_label.pack(side=tk.LEFT, padx=20)


def main():
    """Main entry point."""
    root = tk.Tk()
    app = SignalVizToolV21(root)
    root.mainloop()


if __name__ == '__main__':
    main()
