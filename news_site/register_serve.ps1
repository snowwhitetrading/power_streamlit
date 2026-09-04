# Tự bật website news lúc đăng nhập Windows (chạy ẩn, không cửa sổ).
# Chạy MỘT lần bằng quyền Administrator:
#   powershell -ExecutionPolicy Bypass -File news_site\register_serve.ps1
#
# Gỡ sau này:  Unregister-ScheduledTask -TaskName 'NewsSite' -TaskPath '\PowerDashboard\' -Confirm:$false

$ErrorActionPreference = 'Stop'
$dir = Split-Path -Parent $PSScriptRoot          # thư mục gốc dự án
$pyw = Join-Path $dir '.venv\Scripts\pythonw.exe' # python không cửa sổ
if (-not (Test-Path $pyw)) { $pyw = 'pythonw' }

# pythonw + "-X utf8" = chạy server ẩn, bật UTF-8 (khỏi cần biến môi trường).
$action = New-ScheduledTaskAction -Execute $pyw `
    -Argument '-X utf8 -m uvicorn news_site.app:app --port 8000 --log-level warning' `
    -WorkingDirectory $dir
$trigger = New-ScheduledTaskTrigger -AtLogOn
# Chạy trong phiên đăng nhập của bạn (để bind localhost); không giới hạn thời gian;
# tự khởi động lại tối đa 3 lần nếu chết.
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit ([TimeSpan]::Zero) -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1)
$principal = New-ScheduledTaskPrincipal `
    -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) `
    -LogonType Interactive -RunLevel Limited

Register-ScheduledTask -TaskName 'NewsSite' -TaskPath '\PowerDashboard\' `
    -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Force | Out-Null

Write-Host 'OK: da tao task "PowerDashboard\NewsSite".'
Write-Host 'Server se tu bat moi lan ban dang nhap Windows (cong 8000).'
Write-Host 'Muon bat NGAY bay gio (khong can dang xuat):'
Write-Host '  Start-ScheduledTask -TaskName "NewsSite" -TaskPath "\PowerDashboard\"'
