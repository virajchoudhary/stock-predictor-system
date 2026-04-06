Set-Location "c:\stock price predictor\backend"
.\venv\Scripts\Activate.ps1
.\venv\Scripts\celery.exe -A quantvision beat --loglevel=info --scheduler django_celery_beat.schedulers:DatabaseScheduler
