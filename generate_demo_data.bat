@echo off
echo ========================================
echo SignalVizTool v2.1 - 測試數據生成器
echo ========================================
echo.

python tests\data_generator.py

if %errorlevel% equ 0 (
    echo.
    echo ========================================
    echo ✓ 測試數據生成成功！
    echo ========================================
    echo.
    echo 數據位置: demo_data\
    echo.
    pause
) else (
    echo.
    echo ========================================
    echo ✗ 生成失敗，請確認已安裝 Python
    echo ========================================
    echo.
    pause
)
