# Quick manual prompt test against a trained adapter.
# Usage: scripts\inference.ps1 [config-name] [adapter-dir]

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
    uv run axolotl inference "configs\$Config" --lora-model-dir="$AdapterDir"
}
finally {
    Pop-Location
}
