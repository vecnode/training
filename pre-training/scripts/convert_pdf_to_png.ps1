param(
    [Parameter(Mandatory = $false)]
    [string]$DatasetPath = 'Release_1',

    [Parameter(Mandatory = $false)]
    [string]$OutputPath,

    [Parameter(Mandatory = $false)]
    [string]$LogPath,

    [Parameter(Mandatory = $false)]
    [double]$MaxMb = 1,

    [Parameter(Mandatory = $false)]
    [int]$MaxDim = 4000,

    [Parameter(Mandatory = $false)]
    [int]$Parallel = [Environment]::ProcessorCount,

    [Parameter(Mandatory = $false)]
    [string]$PythonExe
)

$ErrorActionPreference = 'Continue'
$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptRoot

function Resolve-PathFromRoot {
    param(
        [Parameter(Mandatory = $true)][string]$PathValue,
        [Parameter(Mandatory = $false)][switch]$MustExist
    )

    $candidate = $PathValue
    if (-not [System.IO.Path]::IsPathRooted($candidate)) {
        $candidate = Join-Path $projectRoot $candidate
    }

    if ($MustExist) {
        return (Resolve-Path -LiteralPath $candidate).Path
    }

    return $candidate
}

function ConvertTo-SlugName {
    param([string]$Name)
    $slug = $Name.ToLowerInvariant()
    $slug = [System.Text.RegularExpressions.Regex]::Replace($slug, '[^a-z0-9]+', '-')
    $slug = $slug.Trim('-')
    if ([string]::IsNullOrEmpty($slug)) { $slug = 'file' }
    return $slug
}

function Get-PdfPageCount {
    param([string]$PdfPath)
    $info = & pdfinfo $PdfPath 2>&1 | ForEach-Object { "$_" }
    foreach ($line in $info) {
        if ($line -match '^Pages:\s+(\d+)') {
            return [int]$Matches[1]
        }
    }
    return 0
}

function Get-PdfPageSizePts {
    param([string]$PdfPath)
    $info = & pdfinfo $PdfPath 2>&1 | ForEach-Object { "$_" }
    foreach ($line in $info) {
        if ($line -match 'Page size:\s+(\d+(?:\.\d+)?)\s+x\s+(\d+(?:\.\d+)?)\s+pts') {
            return [double]$Matches[1], [double]$Matches[2]
        }
    }
    return 612.0, 792.0
}

function Get-PdfNativeDpi {
    param([string]$PdfPath)
    $list = & pdfimages -list $PdfPath 2>&1 | ForEach-Object { "$_" }
    $ppis = @()
    foreach ($line in $list) {
        if ($line -match '^\s*\d+\s+\d+\s+image\s+') {
            $cols = ($line -split '\s+') | Where-Object { $_ -ne '' }
            if ($cols.Count -ge 14) {
                $xPpi = [int]$cols[12]
                if ($xPpi -gt 0) { $ppis += $xPpi }
            }
        }
    }
    if ($ppis.Count -gt 0) {
        return ($ppis | Measure-Object -Maximum).Maximum
    }
    return 72
}

function Get-RenderDpi {
    param([string]$PdfPath)
    $native = Get-PdfNativeDpi -PdfPath $PdfPath
    $wPts, $hPts = Get-PdfPageSizePts -PdfPath $PdfPath
    $maxPts = [Math]::Max($wPts, $hPts)
    $dpiForMaxDim = [int][Math]::Floor($MaxDim * 72.0 / $maxPts)
    if ($dpiForMaxDim -lt 72) { $dpiForMaxDim = 72 }
    return [Math]::Min($native, $dpiForMaxDim)
}

function Get-PagePngPath {
    param([string]$Slug, [int]$Page)
    return Join-Path $dst ($Slug + '-page-' + $Page + '.png')
}

function Test-PdfConverted {
    param([string]$Slug, [int]$PageCount)
    if ($PageCount -le 0) { return $false }
    for ($page = 1; $page -le $PageCount; $page++) {
        if (-not (Test-Path -LiteralPath (Get-PagePngPath -Slug $Slug -Page $page))) { return $false }
    }
    return $true
}

function Get-MissingPages {
    param([string]$Slug, [int]$PageCount)
    $missing = New-Object System.Collections.Generic.List[int]
    for ($page = 1; $page -le $PageCount; $page++) {
        if (-not (Test-Path -LiteralPath (Get-PagePngPath -Slug $Slug -Page $page))) {
            $missing.Add($page)
        }
    }
    return [int[]]$missing.ToArray()
}

function Get-ContiguousRanges {
    # Collapse a sorted page-number list into [start,end] runs so each run can be
    # rendered with a single pdftoppm call instead of one process per page.
    param([int[]]$Pages)
    $ranges = New-Object System.Collections.Generic.List[object]
    if ($Pages.Count -eq 0) { return $ranges }
    $start = $Pages[0]
    $prev = $Pages[0]
    for ($i = 1; $i -lt $Pages.Count; $i++) {
        if ($Pages[$i] -eq $prev + 1) {
            $prev = $Pages[$i]
            continue
        }
        $ranges.Add([pscustomobject]@{ Start = $start; End = $prev })
        $start = $Pages[$i]
        $prev = $Pages[$i]
    }
    $ranges.Add([pscustomobject]@{ Start = $start; End = $prev })
    return $ranges
}

function Count-DonePages {
    param([string]$Folder, [string]$Slug, [int]$PageCount)
    $have = 0
    for ($page = 1; $page -le $PageCount; $page++) {
        if (Test-Path -LiteralPath (Join-Path $Folder ($Slug + '-page-' + $page + '.png'))) { $have++ }
    }
    return $have
}

$src = Resolve-PathFromRoot -PathValue $DatasetPath -MustExist
if (-not (Test-Path -LiteralPath $src -PathType Container)) {
    throw "Dataset folder not found: $DatasetPath"
}

if ([string]::IsNullOrWhiteSpace($LogPath)) {
    $log = Join-Path $projectRoot 'conversion_log.txt'
} else {
    $log = Resolve-PathFromRoot -PathValue $LogPath
}

function Write-Log {
    param([string]$Message)
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $Message"
    Add-Content -LiteralPath $log -Value $line
    Write-Host $Message
}

$timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$srcSlug = ConvertTo-SlugName (Split-Path -Leaf $src)
$outputsRoot = Join-Path $projectRoot 'outputs'

# Precompute page counts once; reused for the resume-candidate scan below and
# for the main pending-work loop further down (avoids calling pdfinfo twice).
$pdfFiles = @(Get-ChildItem -LiteralPath $src -Filter '*.pdf' -File | Sort-Object Name)
$pdfPageCounts = @{}
foreach ($pdf in $pdfFiles) { $pdfPageCounts[$pdf.FullName] = Get-PdfPageCount -PdfPath $pdf.FullName }
$totalPagesAll = 0
if ($pdfPageCounts.Values.Count -gt 0) { $totalPagesAll = ($pdfPageCounts.Values | Measure-Object -Sum).Sum }

$dst = $null

if ([string]::IsNullOrWhiteSpace($OutputPath)) {
    $resumeCandidates = @()
    if (Test-Path -LiteralPath $outputsRoot) {
        $resumeCandidates = @(
            Get-ChildItem -LiteralPath $outputsRoot -Directory -ErrorAction SilentlyContinue |
                Where-Object { $_.Name -match ('_' + [regex]::Escape($srcSlug) + '$') } |
                Sort-Object Name -Descending
        )
    }

    if ($resumeCandidates.Count -gt 0) {
        Write-Log "Found $($resumeCandidates.Count) previous run(s) for dataset '$srcSlug':"
        $summaries = @()
        for ($i = 0; $i -lt $resumeCandidates.Count; $i++) {
            $cand = $resumeCandidates[$i]
            $donePages = 0
            $donePdfs = 0
            foreach ($pdf in $pdfFiles) {
                $slug = ConvertTo-SlugName $pdf.BaseName
                $pc = $pdfPageCounts[$pdf.FullName]
                $have = Count-DonePages -Folder $cand.FullName -Slug $slug -PageCount $pc
                $donePages += $have
                if ($pc -gt 0 -and $have -eq $pc) { $donePdfs++ }
            }
            $pct = if ($totalPagesAll -gt 0) { [Math]::Round(($donePages * 100.0) / $totalPagesAll, 1) } else { 0 }
            $summaries += [pscustomobject]@{ Folder = $cand; DonePdfs = $donePdfs; DonePages = $donePages; Percent = $pct }
            Write-Log ("  [{0}] {1}  ->  {2}/{3} PDFs, {4}/{5} pages ({6}%) done" -f ($i + 1), $cand.Name, $donePdfs, $pdfFiles.Count, $donePages, $totalPagesAll, $pct)
        }

        Write-Host ''
        $resumeChoice = Read-Host "Resume one of these runs? Enter a number, or press Enter to start a new run"
        if ($resumeChoice) {
            $chosenIndex = 0
            if ([int]::TryParse($resumeChoice, [ref]$chosenIndex) -and $chosenIndex -ge 1 -and $chosenIndex -le $summaries.Count) {
                $chosen = $summaries[$chosenIndex - 1]
                $dst = $chosen.Folder.FullName
                Write-Log "Resuming run: $dst ($($chosen.Percent)% already done, $($chosen.DonePdfs)/$($pdfFiles.Count) PDFs complete)"
            } else {
                Write-Log "Invalid selection '$resumeChoice'. Starting a new run instead."
            }
        }
    }

    if (-not $dst) {
        $dst = Join-Path $outputsRoot ($timestamp + '_' + $srcSlug)
    }
} else {
    $dst = Resolve-PathFromRoot -PathValue $OutputPath
}

$processPy = Join-Path $scriptRoot 'compress_png_max.py'
$parallel = [Math]::Max(1, $Parallel)
$pdfToPpm = (Get-Command pdftoppm -ErrorAction Stop).Source

if ([string]::IsNullOrWhiteSpace($PythonExe)) {
    $venvPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
    if (Test-Path -LiteralPath $venvPython -PathType Leaf) {
        $pythonExe = $venvPython
    } else {
        $pythonExe = (Get-Command python -ErrorAction Stop).Source
    }
} else {
    $pythonExe = Resolve-PathFromRoot -PathValue $PythonExe -MustExist
}

if (-not (Test-Path $dst)) {
    New-Item -ItemType Directory -Path $dst | Out-Null
}

Write-Log "=== max ${MaxDim}px / ${MaxMb}MB run ($parallel workers) -> $dst ==="

$total = $pdfFiles.Count
$skipped = 0
$failures = @()
$pending = New-Object System.Collections.Generic.List[object]
$index = 0

foreach ($pdf in $pdfFiles) {
    $index++
    $slug = ConvertTo-SlugName $pdf.BaseName
    $pageCount = $pdfPageCounts[$pdf.FullName]

    if (Test-PdfConverted -Slug $slug -PageCount $pageCount) {
        $skipped++
        Write-Log "[$index/$total] SKIP $($pdf.Name) (complete)"
        continue
    }

    $missingPages = @(Get-MissingPages -Slug $slug -PageCount $pageCount)
    if ($missingPages.Count -eq 0) { continue }

    $ranges = Get-ContiguousRanges -Pages $missingPages
    $rangesCsv = ($ranges | ForEach-Object { "$($_.Start)-$($_.End)" }) -join ';'

    $dpi = Get-RenderDpi -PdfPath $pdf.FullName
    $pending.Add([pscustomobject]@{
        Index        = $index
        Name         = $pdf.Name
        Path         = $pdf.FullName
        Slug         = $slug
        Dpi          = $dpi
        PageCount    = $pageCount
        MissingPages = $missingPages
        RangesCsv    = $rangesCsv
    })
}

Write-Log "Pending PDFs: $($pending.Count) | Skipped complete: $skipped | Max ${MaxDim}px @ ${MaxMb}MB"

function Show-ConversionProgress {
    param([int]$Done, [int]$Total, [datetime]$StartTime, [int]$SkippedAtStart)
    if ($Total -le 0) { return }
    $percent = [Math]::Min(100, [int](($Done * 100) / $Total))
    $elapsed = (Get-Date) - $StartTime
    $doneThisRun = [Math]::Max(0, $Done - $SkippedAtStart)
    $remaining = $Total - $Done
    $status = "$Done/$Total PDFs ($percent%)"
    if ($doneThisRun -gt 0 -and $remaining -gt 0) {
        $avgSeconds = $elapsed.TotalSeconds / $doneThisRun
        $eta = [TimeSpan]::FromSeconds([Math]::Round($avgSeconds * $remaining))
        $status += " - elapsed $($elapsed.ToString('hh\:mm\:ss')) - ETA $($eta.ToString('hh\:mm\:ss'))"
    } else {
        $status += " - elapsed $($elapsed.ToString('hh\:mm\:ss'))"
    }
    Write-Progress -Activity "Converting PDFs to PNG" -Status $status -PercentComplete $percent
}

# Balance outer per-PDF parallelism against the compress step's own worker pool
# so the two layers don't oversubscribe the machine's logical processors.
$compressJobs = [Math]::Max(1, [Math]::Floor([Environment]::ProcessorCount / $parallel))

$startTime = Get-Date
$converted = 0
$completed = $skipped
Show-ConversionProgress -Done $completed -Total $total -StartTime $startTime -SkippedAtStart $skipped
$queue = [System.Collections.Queue]::new()
foreach ($item in $pending) { [void]$queue.Enqueue($item) }
$running = @()

while ($queue.Count -gt 0 -or $running.Count -gt 0) {
    while ($running.Count -lt $parallel -and $queue.Count -gt 0) {
        $item = $queue.Dequeue()
        $missCount = $item.MissingPages.Count
        $rangeCount = ($item.RangesCsv -split ';').Count
        Write-Log "[$($item.Index)/$total] START $($item.Name) @ $($item.Dpi)dpi ($missCount missing pages, $rangeCount pdftoppm call(s))"
        $job = Start-Job -ScriptBlock {
            param($PdfPath, $Dst, $Slug, $Dpi, $RangesCsv, $ProcessPy, $MaxMb, $MaxDim, $PdfToPpm, $PythonExe, $CompressJobs)
            $prefix = Join-Path $Dst ($Slug + '-page')
            $saved = New-Object System.Collections.Generic.List[string]

            $ranges = @()
            if ($RangesCsv) {
                $ranges = $RangesCsv -split ';' | ForEach-Object {
                    $parts = $_ -split '-'
                    [pscustomobject]@{ Start = [int]$parts[0]; End = [int]$parts[1] }
                }
            }

            foreach ($range in $ranges) {
                # One pdftoppm call renders the whole contiguous run of missing pages
                # instead of re-spawning a process per page.
                $null = & $PdfToPpm -png -r $Dpi -f $range.Start -l $range.End $PdfPath $prefix 2>&1
                if ($LASTEXITCODE -ne 0) { return 1 }

                # pdftoppm zero-pads the page suffix based on the -l value's digit width;
                # normalize every rendered file in this range back to "<slug>-page-<n>.png".
                $rendered = Get-ChildItem -LiteralPath $Dst -Filter ($Slug + '-page-*.png')
                foreach ($file in $rendered) {
                    if ($file.BaseName -notmatch ('^' + [regex]::Escape($Slug) + '-page-0*(\d+)$')) { continue }
                    $pageNum = [int]$Matches[1]
                    if ($pageNum -lt $range.Start -or $pageNum -gt $range.End) { continue }
                    $canonical = Join-Path $Dst ($Slug + '-page-' + $pageNum + '.png')
                    if ($file.FullName -ne $canonical) {
                        Move-Item -LiteralPath $file.FullName -Destination $canonical -Force
                    }
                    if (-not $saved.Contains($canonical)) { $saved.Add($canonical) }
                }
            }

            if ($saved.Count -gt 0) {
                # Pass the (potentially hundreds of) page paths via a list file rather than
                # command-line args - Windows' argument-length limit is easy to blow past
                # once a single PDF has 100+ pages.
                $listFile = Join-Path ([System.IO.Path]::GetTempPath()) ([System.IO.Path]::GetRandomFileName() + '.txt')
                try {
                    Set-Content -LiteralPath $listFile -Value $saved -Encoding UTF8
                    $null = & $PythonExe $ProcessPy --max-mb $MaxMb --max-dim $MaxDim --jobs $CompressJobs --list-file $listFile 2>&1
                    if ($LASTEXITCODE -ne 0) { return 2 }
                } finally {
                    Remove-Item -LiteralPath $listFile -Force -ErrorAction SilentlyContinue
                }
            }
            return 0
        } -ArgumentList $item.Path, $dst, $item.Slug, $item.Dpi, $item.RangesCsv, $processPy, $maxMb, $maxDim, $pdfToPpm, $pythonExe, $compressJobs
        $running += [pscustomobject]@{ Job = $job; Item = $item }
    }

    if ($running.Count -eq 0) { break }

    $done = Wait-Job -Job ($running.Job) -Any
    $slot = $running | Where-Object { $_.Job.Id -eq $done.Id }
    $running = $running | Where-Object { $_.Job.Id -ne $done.Id }

    $code = Receive-Job -Job $done
    Remove-Job -Job $done

    if ($code -ne 0) {
        $failures += $slot.Item.Name
        Write-Log "[$($slot.Item.Index)/$total] FAILED $($slot.Item.Name) (code $code)"
    } else {
        $converted++
        Write-Log "[$($slot.Item.Index)/$total] DONE $($slot.Item.Name)"
    }
    $completed++
    Show-ConversionProgress -Done $completed -Total $total -StartTime $startTime -SkippedAtStart $skipped
}

Write-Progress -Activity "Converting PDFs to PNG" -Completed
Write-Log ''
Write-Log 'Done.'
Write-Log "Output folder: $dst"
Write-Log "PDFs skipped (already complete): $skipped"
Write-Log "PDFs processed this run: $converted"
Write-Log "PDFs failed: $($failures.Count)"
if ($failures.Count -gt 0) {
    $failures | ForEach-Object { Write-Log "  FAILED $_" }
}
