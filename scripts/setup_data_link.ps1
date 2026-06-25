$ErrorActionPreference = 'Stop'

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$target = (Resolve-Path (Join-Path $projectRoot 'data\raw\mat')).Path
$link = Join-Path $projectRoot 'code\SHU_Dataset'

if (Test-Path -LiteralPath $link) {
    $item = Get-Item -LiteralPath $link -Force
    if ($item.LinkType -eq 'Junction' -and $item.Target -contains $target) {
        Write-Host "Data link already points to $target"
        exit 0
    }

    throw "Cannot create data link because the path already exists: $link"
}

New-Item -ItemType Junction -Path $link -Target $target | Out-Null
Write-Host "Created $link -> $target"
