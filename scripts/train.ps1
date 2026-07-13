# Usage: scripts\train.ps1 [config-name]
# Default config-name is qlora-3090-24gb.yml

param(
    [string]$Config = "qlora-3090-24gb.yml"
)

$root = Split-Path $PSScriptRoot -Parent
$env:HF_HOME = Join-Path $root ".cache\huggingface"

Push-Location $root
try {
    uv run axolotl train "configs\$Config"
}
finally {
    Pop-Location
}
