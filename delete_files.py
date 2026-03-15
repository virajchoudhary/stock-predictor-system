import os
import shutil

files_to_delete = [
    # Root level
    "._frontend", "._quant_testing", "._README.md", "file_tree.txt",
    "github_pr_fix_quant_reporter.md", "github_pr_hpo.md", "github_pr_remove_tabs.md", "Technical_Documentation.txt",
    
    # backend/ level
    "backend/._api", "backend/._check_bsedata.py", "backend/._check_nsepython.py",
    "backend/._db.sqlite3", "backend/._debug_options.py", "backend/._debug_reporter.py",
    "backend/._debug_tickers.py", "backend/._manage.py", "backend/._manual_nse.py",
    "backend/._quantvision", "backend/._requirements.txt", "backend/celerybeat-schedule",
    "backend/celerybeat-schedule-shm", "backend/celerybeat-schedule-wal", "backend/check_bsedata.py",
    "backend/check_nsepython.py", "backend/debug_options.py", "backend/debug_reporter.py",
    "backend/debug_tickers.py", "backend/manual_nse.py",
    
    # backend/api/ level
    "backend/api/._.DS_Store", "backend/api/._admin.py", "backend/api/._ai_services.py",
    "backend/api/._apps.py", "backend/api/._migrations", "backend/api/._models.py",
    "backend/api/._tests.py", "backend/api/._urls.py", "backend/api/._views.py",
    "backend/api/.___init__.py", "backend/api/.___pycache__",
    
    # backend/api/migrations/ level
    "backend/api/migrations/._.DS_Store", "backend/api/migrations/.___init__.py", "backend/api/migrations/.___pycache__",
    
    # backend/quantvision/ level
    "backend/quantvision/._.DS_Store", "backend/quantvision/._asgi.py", "backend/quantvision/._settings.py",
    "backend/quantvision/._urls.py", "backend/quantvision/._wsgi.py", "backend/quantvision/.___init__.py",
    "backend/quantvision/.___pycache__",
    
    # frontend/ level
    "frontend/._.DS_Store", "frontend/._app.py", "frontend/._pages", "frontend/portfolio_report.html",
    
    # frontend/pages/ level
    "frontend/pages/._1_Dashboard.py", "frontend/pages/._2_Chat.py", "frontend/pages/._3_Portfolio.py", "frontend/pages/._4_Options_Analysis.py",
]

dirs_to_delete = [
    "quant_testing"
]

base_dir = r"c:\QuantVis"

deleted = []
not_found = []

for f in files_to_delete:
    path = os.path.join(base_dir, os.path.normpath(f))
    if os.path.exists(path):
        if os.path.isfile(path) or os.path.islink(path):
            try:
                os.remove(path)
                deleted.append(f)
            except Exception as e:
                print(f"Failed to delete {f}: {e}")
        else:
            try:
                shutil.rmtree(path)
                deleted.append(f)
            except Exception as e:
                print(f"Failed to delete {f}: {e}")
    else:
        not_found.append(f)

for d in dirs_to_delete:
    path = os.path.join(base_dir, os.path.normpath(d))
    if os.path.exists(path) and os.path.isdir(path):
        try:
            shutil.rmtree(path)
            deleted.append(d)
        except Exception as e:
            print(f"Failed to delete {d}: {e}")
    else:
        not_found.append(d)

print("DELETED:")
for x in deleted:
    print(f"- {x}")

print("\nNOT FOUND:")
for x in not_found:
    print(f"- {x}")
