# 使用官方的 Python 基礎映像檔
FROM python:3.9

# 設置工作目錄
WORKDIR /app

# 複製 SignalVizTool_Requirements.txt  到容器中
COPY SignalVizTool_Requirements.txt /app/requirements.txt

# 安裝 Conda
RUN pip install conda-pack

# 使用 pip 安裝需要的套件
RUN pip install -r SignalVizTool_Requirements.txt

# 複製你的應用程式代碼到容器中
COPY . /app


CMD ["python", "SignalVizTool.py"]
