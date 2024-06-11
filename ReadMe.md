# SignalVizTool 訊號處理與視覺化系統

Educatcode SignalViz Tool 是一個用於訊號處理與視覺化的圖形使用者介面（GUI），旨在降低訊號處理的入門門檻。此工具允許使用者透過簡單的點選操作進行各種訊號特徵的處理與分析，適合初學者與專業人員使用。

## 功能特點

- **時域特徵處理**：自動計算並儲存訊號的時域特徵。
- **頻域特徵處理**：自動計算並儲存訊號的頻域特徵。
- **多檔案圖表繪製**：選取多個檔案並生成圖表。
- **檔案合併**：合併多個 CSV 檔案。
- **時間戳對齊合併**：將兩個 CSV 檔案依據時間戳記對齊並合併。

## 環境建立

- 方法1: 使用 Anaconda yml檔案建置
```bash =
conda env create -f SignalVizTool_Environment.yml
```

- 方法2: 使用 PyPI requirements建置
```bash=
conda create -n SignalVizTool python=3.9
pip install -r SignalVizTool_Requirements.txt
```

- 方法3: 使用 Docker
```bash=
docker build -t SignalVizTool .
docker run SignalVizTool
```


## 使用方法

1. 下載此 GitHub 儲存庫：

    ```bash
    git clone https://github.com/yourusername/SignalVizTool.git
    cd SignalVizTool
    ```

2. 執行主程式：

    ```bash
    python SignalVizTool.py
    ```

3. 在 GUI 中，使用左側的功能按鈕進行相應的訊號處理操作。

## 功能描述

- **Process Time Domain Features**：選擇 CSV 檔案並計算時域特徵。
- **Process Frequency Domain Features**：選擇 CSV 檔案並計算頻域特徵。
- **Select Multiple Files and Plot**：選擇多個 CSV 檔案並繪製其圖表。
- **Merge Multiple CSV Files**：合併多個 CSV 檔案為一個。
- **Align and Merge CSV Files**：將兩個 CSV 檔案依據時間戳記對齊並合併。

## 目錄結構

SignalVizTool/
│
├── SignalVizTool.py # 主程式
├── SignalVizTool_Environment.yaml # 環境設定檔
├── SignalVizTool_Requirements.txt # 環境設定檔
├── ReadMe.md # 專案說明文件
├── EduCatCode.ico # 應用程式圖標
├── data/ # 範例數據資料夾
├── scripts/ # 輔助腳本資料夾
└── results/ # 處理結果存放資料夾




感謝您使用 SignalVizTool Python Tool！如有任何問題或建議，請隨時與我們聯繫。
