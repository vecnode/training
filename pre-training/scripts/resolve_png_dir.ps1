param(
    [Parameter(Mandatory = $true)]
    [string]$DatasetInput,

    [Parameter(Mandatory = $true)]
    [string]$ProjectRoot
)

# Resolves a user-typed dataset name or folder path to a PNG directory, trying
# (in order): the path as-is, the path under the project root, the legacy
# "<name>_PNG" convention, and finally the newest outputs/<timestamp>_<slug>
# folder produced by convert_pdf_to_png.ps1. Prints the resolved absolute path
# and exits 0, or exits 1 with no output if nothing matches.

function ConvertTo-SlugName {
    param([string]$Name)
    $slug = $Name.ToLowerInvariant()
    $slug = [System.Text.RegularExpressions.Regex]::Replace($slug, '[^a-z0-9]+', '-')
    $slug = $slug.Trim('-')
    if ([string]::IsNullOrEmpty($slug)) { $slug = 'file' }
    return $slug
}

function Resolve-IfDirectory {
    param([string]$Path)
    if ([string]::IsNullOrWhiteSpace($Path)) { return $null }
    if (Test-Path -LiteralPath $Path -PathType Container) {
        return (Resolve-Path -LiteralPath $Path).Path
    }
    return $null
}

$candidates = @(
    $DatasetInput,
    (Join-Path $ProjectRoot $DatasetInput),
    ($DatasetInput + '_PNG'),
    (Join-Path $ProjectRoot ($DatasetInput + '_PNG'))
)

foreach ($candidate in $candidates) {
    $resolved = Resolve-IfDirectory $candidate
    if ($resolved) {
        Write-Output $resolved
        exit 0
    }
}

$slug = ConvertTo-SlugName $DatasetInput
$outputsRoot = Join-Path $ProjectRoot 'outputs'
if (Test-Path -LiteralPath $outputsRoot) {
    $match = Get-ChildItem -LiteralPath $outputsRoot -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match ('_' + [regex]::Escape($slug) + '$') } |
        Sort-Object Name -Descending |
        Select-Object -First 1
    if ($match) {
        Write-Output $match.FullName
        exit 0
    }
}

exit 1
