@echo off
echo Starting Stock Price Predictor...

REM Start Redis
echo Starting Redis...
docker start redis 2>nul || docker run -d -p 6379:6379 --name redis redis
timeout /t 2 /nobreak >nul

REM Open Windows Terminal with all services as tabs
echo Opening services...
"%LOCALAPPDATA%\Microsoft\WindowsApps\wt.exe" ^
  new-tab --title "Django" powershell -NoExit -ExecutionPolicy Bypass -File "c:\stock price predictor\scripts\start_django.ps1" ^
  ; new-tab --title "Celery Worker" powershell -NoExit -ExecutionPolicy Bypass -File "c:\stock price predictor\scripts\start_worker.ps1" ^
  ; new-tab --title "Celery Beat" powershell -NoExit -ExecutionPolicy Bypass -File "c:\stock price predictor\scripts\start_beat.ps1" ^
  ; new-tab --title "Streamlit" powershell -NoExit -ExecutionPolicy Bypass -File "c:\stock price predictor\scripts\start_streamlit.ps1"

REM Wait for services then open browser
timeout /t 10 /nobreak >nul
start http://localhost:8501
echo Stock Price Predictor started. Check Windows Terminal tabs for status.
