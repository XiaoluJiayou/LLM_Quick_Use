param([switch]$DownloadSources)
$ErrorActionPreference = 'Stop'
if ($DownloadSources) {
    & (Join-Path $PSScriptRoot 'download_a2_sources.ps1')
}
python (Join-Path $PSScriptRoot 'curate_a2.py') candidates
if ($LASTEXITCODE -ne 0) { throw 'Candidate extraction failed' }
python (Join-Path $PSScriptRoot 'curate_a2.py') select
if ($LASTEXITCODE -ne 0) { throw 'Candidate selection failed' }
python (Join-Path $PSScriptRoot 'augment_a2_short_sources.py')
if ($LASTEXITCODE -ne 0) { throw 'Concise source selection failed' }
python (Join-Path $PSScriptRoot 'finalize_a2.py')
if ($LASTEXITCODE -ne 0) { throw 'Dataset construction failed' }
python (Join-Path $PSScriptRoot 'validate_a2_dataset.py') --write-report
if ($LASTEXITCODE -ne 0) { throw 'Validation failed' }
