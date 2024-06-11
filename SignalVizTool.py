import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import Canvas, Frame
from tkinter import filedialog, simpledialog, ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import os
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 頻域特徵函數
def get_dominant_frequency(signal, frame_size, hop_length, sampling_rate):
    dom_freqs = []
    for i in range(0, len(signal), hop_length):
        L = len(signal[i:i + frame_size])
        y = abs(np.fft.fft(signal[i:i + frame_size]))[:int(L / 2)]
        f = np.fft.fftfreq(L, 1 / sampling_rate)[:int(L / 2)]
        dominant_freq = f[np.argmax(y)]
        dom_freqs.append(dominant_freq)
    return np.array(dom_freqs)

def get_band_energy(signal, frame_size, hop_length, sampling_rate, freq_band):
    band_energies = []
    for i in range(0, len(signal), hop_length):
        L = len(signal[i:i + frame_size])
        y = abs(np.fft.fft(signal[i:i + frame_size]))[:int(L / 2)]
        f = np.fft.fftfreq(L, 1 / sampling_rate)[:int(L / 2)]
        band_energy = np.sum(y[(f >= freq_band[0]) & (f <= freq_band[1])])
        band_energies.append(band_energy)
    return np.array(band_energies)

def get_spectral_centroid(signal, frame_size, hop_length, sampling_rate):
    centroids = []
    for i in range(0, len(signal), hop_length):
        L = len(signal[i:i + frame_size])
        y = abs(np.fft.fft(signal[i:i + frame_size]))[:int(L / 2)]
        f = np.fft.fftfreq(L, 1 / sampling_rate)[:int(L / 2)]
        centroid = np.sum(f * y) / np.sum(y)
        centroids.append(centroid)
    return np.array(centroids)

list_features_function_freq = [get_dominant_frequency, get_band_energy, get_spectral_centroid]

def get_all_features_freq(signal, frame_size, hop_length, sampling_rate):
    stationary_features = []
    for func in list_features_function_freq:
        if func == get_band_energy:
            f = func(signal, frame_size, hop_length, sampling_rate, freq_band=(0, 5000))  # 假設我們關心0到5000Hz的能量
        else:
            f = func(signal, frame_size, hop_length, sampling_rate)
        stationary_features.append(f)
    return stationary_features

# 更新進度條
def update_progress_bar(n_total, n_current):
    progress = n_current / n_total * 100
    progress_var.set(progress)
    root.update_idletasks()

# 清除繪圖區域
def clear_frame(frame):
    for widget in frame.winfo_children():
        widget.destroy()

# 讓使用者選擇要處理的特徵
def select_features(file_path):
    df = pd.read_csv(file_path)
    selected_features = []

    feature_window = tk.Toplevel(root)
    feature_window.title("Select Features to Process")

    vars = {}
    for col in df.columns:
        var = tk.BooleanVar()
        chk = tk.Checkbutton(feature_window, text=col, variable=var)
        chk.pack(side=tk.TOP, anchor="w")
        vars[col] = var

    def confirm_selection():
        nonlocal selected_features
        selected_features = [col for col, var in vars.items() if var.get()]
        feature_window.destroy()

    confirm_button = tk.Button(feature_window, text="Confirm", command=confirm_selection)
    confirm_button.pack(side=tk.BOTTOM)

    feature_window.wait_window()
    return selected_features



# 繪製並儲存特徵圖
def plot_timedomain_features(features_df, feature_name, file_name):
    fig, axes = plt.subplots(4, 4, figsize=(10, 8))  # 調整圖表大小
    fig.suptitle(file_name, fontsize=16)  # 設置大圖標題為檔名
    for i, feature in enumerate(features_df.columns):
        row, col = divmod(i, 4)
        ax = axes[row, col]
        ax.bar(range(len(features_df)), features_df[feature], color=plt.cm.tab20(i / len(features_df.columns)))
        ax.set_title(feature, fontsize=8)  # 調整字體大小
        ax.set_xlabel('Index', fontsize=6)
        ax.set_ylabel('Value', fontsize=6)
        ax.tick_params(axis='both', labelsize=5)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    for j in range(i + 1, 16):
        row, col = divmod(j, 4)
        axes[row, col].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 調整佈局以適應大圖標題
    plt.savefig(f'{file_name}_{feature_name}.png')
    plt.close()

    return fig







# 繪製並儲存特徵圖
def plot_frequencydomain_features(features_df, feature_name, file_name):
    fig, axes = plt.subplots(4, 4, figsize=(10, 8))  # 調整圖表大小
    fig.suptitle(file_name, fontsize=16)  # 設置大圖標題為檔名
    for i, feature in enumerate(features_df.columns):
        row, col = divmod(i, 4)
        ax = axes[row, col]
        ax.bar(range(len(features_df)), features_df[feature], color=plt.cm.tab20(i / len(features_df.columns)))
        ax.set_title(feature, fontsize=8)  # 調整字體大小
        ax.set_xlabel('Index', fontsize=6)
        ax.set_ylabel('Value', fontsize=6)
        ax.tick_params(axis='both', labelsize=5)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    for j in range(i + 1, 16):
        row, col = divmod(j, 4)
        axes[row, col].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 調整佈局以適應大圖標題
    plt.savefig(f'{file_name}_{feature_name}.png')
    plt.close()

    return fig




def process_and_save_time_features():
    file_paths = filedialog.askopenfilenames(filetypes=[("CSV files", "*.csv")])
    if not file_paths:
        return
    
    frame_size = simpledialog.askinteger("Input", "Enter the frame size:", minvalue=1, maxvalue=200000)
    hop_length = simpledialog.askinteger("Input", "Enter the hop length:", minvalue=1, maxvalue=200000)
    n_total = len(file_paths)

    first_file_features = select_features(file_paths[0])
    if not first_file_features:
        return

    clear_frame(scrollable_frame)  # 清除之前的圖表

    for i, file_path in enumerate(file_paths):
        update_progress_bar(n_total, i + 1)
        df = pd.read_csv(file_path)
        missing_features = [f for f in first_file_features if f not in df.columns]
        if missing_features:
            tk.messagebox.showwarning("Warning", f"Features {missing_features} not found in {file_path}.")
            continue

        for feature_name in first_file_features:
            signal = df[feature_name]
            features = get_all_features_time(signal, frame_size, hop_length)
            features_df = pd.DataFrame(features).T

            features_df.columns = [f'{feature_name} Peak', f'{feature_name} Rms', f'{feature_name} Crest factor', f'{feature_name} Std',
                                   f'{feature_name} Variance', f'{feature_name} Skewness', f'{feature_name} Kurtosis',
                                   f'{feature_name} Shape factor', f'{feature_name} Impulse factor', f'{feature_name} Margin factor',
                                   f'{feature_name} Mean', f'{feature_name} A factor', f'{feature_name} B factor']

            file_name = os.path.basename(file_path).split('.')[0]
            fig = plot_timedomain_features(features_df, feature_name, file_name)
            features_df.to_csv(f'{file_name}_{feature_name}.csv', index=False)

            # 在UI上顯示圖片
            canvas = FigureCanvasTkAgg(fig, master=scrollable_frame)
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=False)
            canvas.draw()

        update_progress_bar(n_total, 0)

    tk.messagebox.showinfo("Success", "All files processed and features saved.")










# 頻域特徵處理並儲存
def process_and_save_frequency_features():
    file_paths = filedialog.askopenfilenames(filetypes=[("CSV files", "*.csv")])
    if not file_paths:
        return
    
    sampling_rate = simpledialog.askinteger("Input", "Enter the sampling rate:", minvalue=1, maxvalue=200000)
    if sampling_rate is None:
        return
    
    frame_size = simpledialog.askinteger("Input", "Enter the frame size:", minvalue=1, maxvalue=200000)
    if frame_size is None:
        return
    
    hop_length = simpledialog.askinteger("Input", "Enter the hop length:", minvalue=1, maxvalue=200000)
    if hop_length is None:
        return
    
    n_total = len(file_paths)

    first_file_features = select_features(file_paths[0])
    if not first_file_features:
        return

    clear_frame(scrollable_frame)  # 清除之前的圖表

    for i, file_path in enumerate(file_paths):
        update_progress_bar(n_total, i + 1)
        df = pd.read_csv(file_path)
        missing_features = [f for f in first_file_features if f not in df.columns]
        if missing_features:
            tk.messagebox.showwarning("Warning", f"Features {missing_features} not found in {file_path}.")
            continue

        for feature_name in first_file_features:
            signal = df[feature_name]
            features = get_all_features_freq(signal, frame_size, hop_length, sampling_rate)
            features_df = pd.DataFrame(features).T

            features_df.columns = [f'{feature_name} Dominant Frequency', f'{feature_name} Band Energy', f'{feature_name} Spectral Centroid']

            file_name = os.path.basename(file_path).split('.')[0]
            fig = plot_frequencydomain_features(features_df, feature_name, file_name)
            features_df.to_csv(f'{file_name}_{feature_name}.csv', index=False)

            # 在UI上顯示圖片
            canvas = FigureCanvasTkAgg(fig, master=scrollable_frame)
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=False)
            canvas.draw()

        update_progress_bar(n_total, 0)

    tk.messagebox.showinfo("Success", "All files processed and features saved.")

# 繪製多個文件的圖表並儲存
def select_multiple_files_and_plot():
    file_paths = filedialog.askopenfilenames(filetypes=[("CSV files", "*.csv")])
    if not file_paths:
        return  # 如果沒有選擇任何文件則退出

    common_columns = set(pd.read_csv(file_paths[0]).columns)
    for file_path in file_paths[1:]:
        df = pd.read_csv(file_path)
        common_columns.intersection_update(df.columns)

    if not common_columns:
        tk.messagebox.showwarning("Warning", "No common columns found in selected files.")
        return

    selected_cols = select_features_to_plot(list(common_columns))
    if not selected_cols:
        return

    n_total = len(file_paths)
    clear_frame(scrollable_frame)

    for i, file_path in enumerate(file_paths):
        df = pd.read_csv(file_path)[selected_cols]
        fig, ax = plt.subplots(figsize=(12, 4))
        label = file_path.split('/')[-1]
        for col in selected_cols:
            ax.plot(df[col], label=col)
        ax.set_title(f'{label[:-4]}')
        ax.legend()

        canvas = FigureCanvasTkAgg(fig, master=scrollable_frame)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        canvas.draw()

        update_progress_bar(n_total, i + 1)
    
    canvas_frame.update_idletasks()
    canvas_frame.configure(scrollregion=canvas_frame.bbox("all"))

# 合併 CSV 檔案
def read_and_merge_files(file_paths, save_path):
    n_total = len(file_paths)
    combined_df = pd.DataFrame()
    
    for i, file_path in enumerate(file_paths):
        df = pd.read_csv(file_path)
        combined_df = pd.concat([combined_df, df])
        update_progress_bar(n_total, i + 1)  # 更新進度條

    if save_path:
        combined_df.to_csv(save_path, index=False)
        tk.messagebox.showinfo("Success", f"Files merged and saved as {save_path}")

    update_progress_bar(n_total, n_total)  # 重製進度條

def merge_multiple_csv_files():
    file_paths = filedialog.askopenfilenames(filetypes=[("CSV files", "*.csv")])
    
    if file_paths:
        save_path = filedialog.asksaveasfilename(defaultextension=".csv",
                                                 filetypes=[("CSV files", "*.csv")])
        threading.Thread(target=read_and_merge_files, args=(file_paths, save_path)).start()

# 將兩個 CSV 文件依照時間戳記對齊並合併
def align_and_merge_csv():
    file_path_vib = filedialog.askopenfilename(title="Select vibration data CSV", filetypes=[("CSV files", "*.csv")])
    file_path_cur = filedialog.askopenfilename(title="Select current data CSV", filetypes=[("CSV files", "*.csv")])
    
    if not file_path_vib or not file_path_cur:
        messagebox.showinfo("Info", "Operation cancelled or file not selected.")
        return

    vib = pd.read_csv(file_path_vib)
    cur_1st = pd.read_csv(file_path_cur)

    vib.index = pd.to_datetime(vib['timestamp'])
    cur_1st.index = pd.to_datetime(cur_1st['timestamp'])

    start_time = max(vib.index.min(), cur_1st.index.min())
    end_time = min(vib.index.max(), cur_1st.index.max())

    vib_filtered = vib[(vib.index >= start_time) & (vib.index <= end_time)]
    cur_1st_filtered = cur_1st[(cur_1st.index >= start_time) & (cur_1st.index <= end_time)]

    merged_df = pd.merge_asof(vib_filtered.sort_index(), cur_1st_filtered.sort_index(), left_index=True, right_index=True, direction='nearest')

    save_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
    if save_path:
        merged_df.to_csv(save_path, index=False)
        messagebox.showinfo("Success", f"Merged file saved as {save_path}")
    else:
        messagebox.showinfo("Info", "Save operation cancelled.")

# Tkinter 主視窗
root = tk.Tk()
root.title("ITRI M100 Python Tool")
root.state('zoomed')
root.iconbitmap('EduCatCode.ico') 

# 左側功能按鍵欄位
left_frame = Frame(root)
left_frame.pack(side=tk.LEFT, padx=20, fill=tk.Y)

# 進度條
progress_var = tk.DoubleVar()
progress_bar = ttk.Progressbar(left_frame, orient="horizontal", length=200, mode="determinate", variable=progress_var)
progress_bar.pack(pady=10)

# 時域特徵處理按鈕
time_feature_button = tk.Button(left_frame, text="Process Time Domain Features", command=process_and_save_time_features)
time_feature_button.pack(pady=10)

# 頻域特徵處理按鈕
freq_feature_button = tk.Button(left_frame, text="Process Frequency Domain Features", command=process_and_save_frequency_features)
freq_feature_button.pack(pady=10)

# 繪製多個文件的圖表並儲存
multi_file_button = tk.Button(left_frame, text="Select Multiple Files and Plot", command=select_multiple_files_and_plot)
multi_file_button.pack(pady=10)

# 合併檔案
merge_button = tk.Button(left_frame, text="Merge Multiple CSV Files", command=merge_multiple_csv_files)
merge_button.pack(pady=10)

# 依時間搓記合併
align_merge_button = tk.Button(left_frame, text="Align and Merge CSV Files", command=align_and_merge_csv)
align_merge_button.pack(pady=10)

# 滾動軸欄位
canvas_frame = Canvas(root)
canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# 滾動條
scrollbar = tk.Scrollbar(root, command=canvas_frame.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
canvas_frame.configure(yscrollcommand=scrollbar.set)

# 在 Canvas 中新增 Frame 做為顯示圖的區塊
scrollable_frame = Frame(canvas_frame)
scrollable_frame.bind("<Configure>", lambda e: canvas_frame.configure(scrollregion=canvas_frame.bbox("all")))

canvas_frame.create_window((0, 0), window=scrollable_frame, anchor="nw")

def on_configure(event):
    canvas_frame.configure(scrollregion=canvas_frame.bbox('all'))

canvas_frame.bind('<Configure>', on_configure)

root.mainloop()
