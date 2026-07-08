# Chạy trang News/Updates. Mở http://127.0.0.1:8000 sau khi khởi động.
# Dùng: mở PowerShell tại thư mục gốc dự án rồi chạy:  .\news_site\run.ps1
$env:PYTHONUTF8 = "1"
$root = Split-Path -Parent $PSScriptRoot
& "$root\.venv\Scripts\python.exe" -m uvicorn news_site.app:app --reload --port 8000
