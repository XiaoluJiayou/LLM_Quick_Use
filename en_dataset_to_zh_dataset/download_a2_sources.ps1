param()
$ErrorActionPreference = 'Stop'
$destination = Join-Path $PSScriptRoot 'en_data/a2_sources'
New-Item -ItemType Directory -Path $destination -Force | Out-Null
$sources = @(
    @{ Name='CoProv2_train.csv'; Repository='Visualignment/SafetyDPO'; Revision='5f71f8db885f096f84abd391f48e49e78b32e8d1'; File='data/CoProv2_train.csv' },
    @{ Name='detonate_race_hateful.csv'; Repository='abhilekhborah/Detonate'; Revision='e6c3214948eef85e33cf5b8175c741dd79db161a'; File='race_hateful.csv' },
    @{ Name='detonate_gender_hateful.csv'; Repository='abhilekhborah/Detonate'; Revision='e6c3214948eef85e33cf5b8175c741dd79db161a'; File='gender_hateful.csv' },
    @{ Name='detonate_disability_hateful.csv'; Repository='abhilekhborah/Detonate'; Revision='e6c3214948eef85e33cf5b8175c741dd79db161a'; File='disability_hateful.csv' },
    @{ Name='jade_v1_zh.csv'; Repository='whitzard-ai/jade-db'; Revision='e8b949936b08e94434f54f8c20de6c1516122257'; File='jade-t2i-v1.0/jade_benchmark_zh.csv' },
    @{ Name='jade_v2_zh.csv'; Repository='whitzard-ai/jade-db'; Revision='cfb889959e15d49e79b00e3e7dc52e0e232e1ca8'; File='jade-t2i-v2.0/jade_htg_benchmark_zh.csv' },
    @{ Name='CoProV2_LICENSE.txt'; Repository='Visualignment/SafetyDPO'; Revision='5f71f8db885f096f84abd391f48e49e78b32e8d1'; File='LICENSE' },
    @{ Name='DETONATE_LICENSE.txt'; Repository='abhilekhborah/Detonate'; Revision='e6c3214948eef85e33cf5b8175c741dd79db161a'; File='LICENSE' },
    @{ Name='JADE_LICENSE.txt'; Repository='whitzard-ai/jade-db'; Revision='cfb889959e15d49e79b00e3e7dc52e0e232e1ca8'; File='LICENSE' }
)
$manifest = foreach ($source in $sources) {
    $url = "https://raw.githubusercontent.com/$($source.Repository)/$($source.Revision)/$($source.File)"
    $file = Join-Path $destination $source.Name
    if (!(Test-Path -LiteralPath $file)) {
        Invoke-WebRequest -Uri $url -OutFile $file -TimeoutSec 60
    }
    $item = Get-Item -LiteralPath $file
    $hash = (Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash.ToLowerInvariant()
    Write-Host "$($source.Name): $($item.Length) bytes"
    @{ file=$source.Name; url=$url; repository=$source.Repository; revision=$source.Revision; bytes=$item.Length; sha256=$hash }
}
$manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $destination 'manifest.json') -Encoding utf8
