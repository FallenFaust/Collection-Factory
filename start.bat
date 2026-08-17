@echo off
rem Двойной клик по этому файлу запускает инструмент и открывает страницу в браузере.
rem Перед запуском должен быть запущен ComfyUI (порт 8188) и задан ключ ANTHROPIC_API_KEY.
cd /d "%~dp0"
if exist ".env" (
  for /f "usebackq tokens=1,* delims==" %%a in (".env") do set "%%a=%%b"
)
python pipeline\studio.py --out runs
if errorlevel 1 pause
