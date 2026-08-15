# Runs eval/evaluate.py against the held-out validation split.
# Usage: scripts\evaluate.ps1 [config-name] [adapter-dir]

param(
    [string]$Config = "qlora-3090-24gb.yml",
    [string]$AdapterDir = ""
)

$root = Split-Path $PSScriptRoot -Parent
$env:HF_HOME = Join-Path $root ".cache\huggingface"

if ($AdapterDir -eq "") {
    $base = [System.IO.Path]::GetFileNameWithoutExtension($Config)
    $AdapterDir = Join-Path $root "output\$base"
}

Push-Location $root
try {
    uv run python eval\evaluate.py --config "configs\$Config" --adapter-dir "$AdapterDir"
}
finally {
    Pop-Location
}
