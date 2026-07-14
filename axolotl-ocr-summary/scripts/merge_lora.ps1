# Merges a trained LoRA/QLoRA adapter into a standalone full model.
# Usage: scripts\merge_lora.ps1 [config-name] [adapter-dir]
# adapter-dir defaults to output/<config-basename>/ (Axolotl's default output_dir)

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
    uv run axolotl merge-lora "configs\$Config" --lora-model-dir="$AdapterDir"
}
finally {
    Pop-Location
}
