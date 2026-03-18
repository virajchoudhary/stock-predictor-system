Set-Location "c:\stock price predictor\backend"
.\venv\Scripts\Activate.ps1
.\venv\Scripts\celery.exe -A quantvision worker --loglevel=info --pool=solo
