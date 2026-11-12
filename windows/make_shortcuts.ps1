# Builds the desktop preset tree for llama-modchip:
#   Desktop\DSV4 пресеты\<ctx>\быстрый|точный.lnk
# Re-run any time: powershell -ExecutionPolicy Bypass -File windows\make_shortcuts.ps1

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PresetDir = Join-Path $ScriptDir "presets"
$Desktop   = [Environment]::GetFolderPath("Desktop")
$Root      = Join-Path $Desktop "DSV4 пресеты"
$ServerExe = Join-Path $ScriptDir "bin-modchip\llama-server.exe"

New-Item -ItemType Directory -Path $PresetDir -Force | Out-Null
New-Item -ItemType Directory -Path $Root -Force | Out-Null

$contexts = @(32768, 65536, 131072, 262144, 524288, 1048576)

function Write-PresetBat([string]$file, [string]$ctx, [string]$extra, [string]$title) {
@"
@echo off
chcp 65001 >nul
title $title
cd /d "%~dp0.."

taskkill /IM llama-server.exe /F >nul 2>&1
ping -n 3 127.0.0.1 >nul

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0..\modchip_dsv4.ps1" -Ctx $ctx $extra
if errorlevel 1 (
    echo.
    echo [!] Server exited with code %errorlevel% - see the log above.
)
echo.
echo Window closes in 5 seconds ...
ping -n 6 127.0.0.1 >nul
"@ | Set-Content -LiteralPath (Join-Path $PresetDir $file) -Encoding ASCII
}

# stop helper
@"
@echo off
chcp 65001 >nul
title Stop DSV4 server
taskkill /IM llama-server.exe /F >nul 2>&1
if errorlevel 1 (echo Server was not running.) else (echo Server stopped.)
ping -n 3 127.0.0.1 >nul
"@ | Set-Content -LiteralPath (Join-Path $PresetDir "STOP_server.bat") -Encoding ASCII

$wsh = New-Object -ComObject WScript.Shell

function New-Link([string]$lnk, [string]$target, [string]$desc) {
    $sc = $wsh.CreateShortcut($lnk)
    $sc.TargetPath = $target
    $sc.WorkingDirectory = $PresetDir
    if (Test-Path $ServerExe) { $sc.IconLocation = "$ServerExe,0" }
    $sc.Description = $desc
    $sc.Save()
}

$made = @()
foreach ($ctx in $contexts) {
    $label = if ($ctx -ge 1048576) { "1M" } else { "{0}K" -f ($ctx / 1024) }
    $sub = Join-Path $Root $label
    New-Item -ItemType Directory -Path $sub -Force | Out-Null

    # fast preset: 4 of 6 experts
    Write-PresetBat -file ("ctx{0}_fast.bat" -f $ctx) -ctx "$ctx" -extra "-Fast" -title "DSV4 $label fast"
    New-Link -lnk (Join-Path $sub "быстрый — 8 tok-s.lnk") -target (Join-Path $PresetDir ("ctx{0}_fast.bat" -f $ctx)) -desc "ctx $ctx, MoE Top-K 4 (fast)"

    # exact preset: all 6 experts
    Write-PresetBat -file ("ctx{0}_exact.bat" -f $ctx) -ctx "$ctx" -extra "" -title "DSV4 $label exact"
    New-Link -lnk (Join-Path $sub "точный — 6.3 tok-s.lnk") -target (Join-Path $PresetDir ("ctx{0}_exact.bat" -f $ctx)) -desc "ctx $ctx, MoE Top-K 6 (unpruned)"

    $made += (Join-Path $sub "быстрый — 8 tok-s.lnk")
    $made += (Join-Path $sub "точный — 6.3 tok-s.lnk")
}

New-Link -lnk (Join-Path $Root "ОСТАНОВИТЬ сервер.lnk") -target (Join-Path $PresetDir "STOP_server.bat") -desc "Stops llama-server.exe"
$made += (Join-Path $Root "ОСТАНОВИТЬ сервер.lnk")

$readme = @"
DSV4 — DeepSeek-V4-Flash Fractal (локальная модель, 284B, 56.5 GiB)

Папки = размер контекста (сколько токенов модель держит в памяти разговора).
 32K / 64K / 128K / 256K / 512K / 1M

Внутри каждой папки два ярлыка:
  быстрый — 4 из 6 экспертов на токен. ~8 tok/s. Качество чуть ниже, годится для рутины.
  точный  — все 6 экспертов, как в оригинальной модели. ~6.3 tok/s. Максимальное качество.

Скорость от контекста почти не зависит, контекст влияет только на память:
  32K -> 512K: всё влезает в 64 ГБ RAM спокойно
  1M: ~16 ГБ свободной RAM остаётся, работать можно

Запуск: двойной клик по ярлыку. Первый запуск ждёт 30-60 секунд (модель читается с диска).
Сервер слушает http://127.0.0.1:8080 — это же адрес, который использует DeepSeek Harness.
Одновременно может работать только один пресет: новый запуск сам убивает старый сервер.
"ОСТАНОВИТЬ сервер" — выключает модель.
"@
Set-Content -LiteralPath (Join-Path $Root "ЧИТАЙ МЕНЯ.txt") -Value $readme -Encoding UTF8

Write-Host "Preset tree created:" -ForegroundColor Green
Get-ChildItem $Root -Recurse | Where-Object { $_.Extension -in '.lnk', '.txt' } | ForEach-Object { Write-Host ("  " + $_.FullName.Replace($Root, "")) }
