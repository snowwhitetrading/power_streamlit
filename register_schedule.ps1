# One-shot setup for the pipeline's scheduled tasks. Run once:
#   powershell -ExecutionPolicy Bypass -File register_schedule.ps1
#
# CSV pipeline  -> daily 09:00                    (scrape market data + upload CSV)
# JSON pipeline -> daily 09:00, 12:00, 15:00, 18:00 (documents+news+press + upload JSON)
#
# The 5 tasks run independently. They fire while you are logged in, and catch up a
# missed run when the machine next becomes available (StartWhenAvailable). Re-running
# this script is safe: -Force overwrites the tasks with the same names.

$ErrorActionPreference = 'Stop'
$dir = $PSScriptRoot
$py  = Join-Path $dir '.venv\Scripts\python.exe'
if (-not (Test-Path $py)) { $py = 'python' }

# Run in the background whether or not the user is logged on. S4U = "run only
# when logged on" is NOT used; the task survives logoff/lock instead of being
# Ctrl+C'd (exit 0xC000013A). WakeToRun wakes the machine at the scheduled time;
# the battery flags let it fire/keep running on laptop power.
$principal = New-ScheduledTaskPrincipal -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) `
    -LogonType S4U -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -WakeToRun `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries

function Register-PipelineTask($name, $group, $time) {
    $action = New-ScheduledTaskAction -Execute $py `
        -Argument ('"{0}\pipeline_run.py" --group {1} --continue-on-error --log' -f $dir, $group) `
        -WorkingDirectory $dir
    $trigger = New-ScheduledTaskTrigger -Daily -At $time
    Register-ScheduledTask -TaskName $name -TaskPath '\PowerDashboard\' `
        -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Force | Out-Null
    Write-Host ("  {0,-9} -> daily {1}  ({2})" -f $name, $time, $group)
}

Write-Host 'Registering PowerDashboard scheduled tasks...'
Register-PipelineTask 'CSV 09h'  'csv'  '09:00'
Register-PipelineTask 'JSON 09h' 'json' '09:00'
Register-PipelineTask 'JSON 12h' 'json' '12:00'
Register-PipelineTask 'JSON 15h' 'json' '15:00'
Register-PipelineTask 'JSON 18h' 'json' '18:00'
Write-Host 'Done. Manage them in Task Scheduler under the "PowerDashboard" folder.'
