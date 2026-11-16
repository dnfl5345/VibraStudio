# Build VibraStudio.exe (PyInstaller, onedir).
# Usage:  powershell -ExecutionPolicy Bypass -File packaging\build_exe.ps1 [-DistPath dist]
# Use another -DistPath while VibraStudio.exe is running from dist\VibraStudio.
param([string]$DistPath = "dist")
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root

$py = Join-Path $root ".venv312\Scripts\python.exe"
if (-not (Test-Path $py)) { Write-Error "venv not found: $py"; exit 1 }

& $py -m pip install --quiet pyinstaller
& $py packaging\make_icon.py

# PyInstaller wipes dist\VibraStudio. Recordings and outputs made with the
# exe live there, so park them outside and put them back after the build.
$app = Join-Path $root (Join-Path $DistPath "VibraStudio")
$park = Join-Path $root (Join-Path $DistPath "_userdata_keep")
$keep = @("recordings", "outputs", "vibrastudio_config.json", "selftest.json", "livecheck.txt", "livecheck.json")
New-Item -ItemType Directory -Force $park | Out-Null
foreach ($k in $keep) {
    $src = Join-Path $app $k
    if (Test-Path $src) { Move-Item $src (Join-Path $park $k) -Force }
}

# nidaqmx 1.x resolves its own and its dependencies' versions through
# importlib.metadata at import time, so the dist-info metadata must ship
# inside the bundle or `import nidaqmx` dies with
# "No package metadata was found for ...".
& (Join-Path $root ".venv312\Scripts\pyinstaller.exe") `
    --noconfirm --clean --windowed --onedir `
    --distpath $DistPath `
    --name VibraStudio `
    --icon packaging\vibrastudio.ico `
    --copy-metadata nidaqmx `
    --copy-metadata nitypes `
    --copy-metadata hightime `
    --copy-metadata deprecation `
    --copy-metadata tzlocal `
    --collect-submodules nidaqmx `
    --collect-submodules OpenGL `
    --hidden-import pyqtgraph.opengl `
    --hidden-import mpl_toolkits.mplot3d `
    --collect-all imageio_ffmpeg `
    --add-data "rigs;rigs" `
    run_app.py

$code = $LASTEXITCODE
New-Item -ItemType Directory -Force $app | Out-Null
foreach ($k in $keep) {
    $src = Join-Path $park $k
    if (Test-Path $src) { Move-Item $src (Join-Path $app $k) -Force }
}
if ((Get-ChildItem $park -Force | Measure-Object).Count -eq 0) { Remove-Item $park }

if ($code -eq 0) {
    Write-Host "`nBuild OK -> $app\VibraStudio.exe"
} else {
    Write-Error "PyInstaller failed ($code)"
}
