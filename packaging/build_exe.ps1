# Build VibraStudio.exe (PyInstaller, onedir).
# Usage:  powershell -ExecutionPolicy Bypass -File packaging\build_exe.ps1
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root

$py = Join-Path $root ".venv312\Scripts\python.exe"
if (-not (Test-Path $py)) { Write-Error "venv not found: $py"; exit 1 }

& $py -m pip install --quiet pyinstaller
& $py packaging\make_icon.py

& (Join-Path $root ".venv312\Scripts\pyinstaller.exe") `
    --noconfirm --clean --windowed --onedir `
    --name VibraStudio `
    --icon packaging\vibrastudio.ico `
    run_app.py

if ($LASTEXITCODE -eq 0) {
    Write-Host "`nBuild OK -> $root\dist\VibraStudio\VibraStudio.exe"
} else {
    Write-Error "PyInstaller failed ($LASTEXITCODE)"
}
