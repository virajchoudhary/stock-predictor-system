Set-Location C:\QuantVis\backend
.\venv\Scripts\Activate.ps1
.\venv\Scripts\celery.exe -A quantvision worker --loglevel=info --pool=solo
