# Bootstraps the uv-managed environment for this workspace.
# Every download (venv, python, pip cache, HF model weights, HF datasets)
# is redirected under .cache/ and .venv/ in this repo, both gitignored.
# `Remove-Item -Recurse -Force .venv, .cache` fully resets the workspace.

$root = $PSScriptRoot
$env:UV_CACHE_DIR = Join-Path $root ".cache\uv"
$env:HF_HOME = Join-Path $root ".cache\huggingface"
$env:HF_HUB_ENABLE_HF_TRANSFER = "1"

New-Item -ItemType Directory -Force -Path $env:UV_CACHE_DIR | Out-Null
New-Item -ItemType Directory -Force -Path $env:HF_HOME | Out-Null

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "uv not found. Install it first: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
}

Push-Location $root
try {
    uv venv --python 3.11
    uv sync

    # axolotl pulls in flash-attn / xformers style extras that need torch already
    # present and built without isolation — reinstall it explicitly after sync.
    uv pip install --no-build-isolation "axolotl[deepspeed]"

    # Pulls example configs + deepspeed configs into ./axolotl-examples (gitignored)
    uv run axolotl fetch examples --dest .cache/axolotl-examples
    uv run axolotl fetch deepspeed_configs --dest .cache/deepspeed_configs
}
finally {
    Pop-Location
}

Write-Host "Bootstrap complete. Activate with: .venv\Scripts\Activate.ps1"
