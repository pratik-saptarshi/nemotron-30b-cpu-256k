param(
    [switch]$Cuda,
    [switch]$Clean,
    [ValidateSet("dsv4", "nemotron")]
    [string]$Target = "dsv4",
    [string]$Arch = "89",
    [string]$SrcRoot = ""
)

# Builds the patched llama.cpp engine needed by Fractal-64 (GGML_TYPE_Q2_0 = 42) and
# LUT-VQ3-64 (GGML_TYPE_VQ3_0 = 43) GGUFs on Windows.
# Stock llama.cpp cannot load them: the patch changes Q2_0 to 30 bytes / 64 weights
# and adds VQ3_0 at 24 bytes / 64 weights.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File windows\build_patched_engine.ps1
#   powershell -ExecutionPolicy Bypass -File windows\build_patched_engine.ps1 -Target nemotron
#   powershell -ExecutionPolicy Bypass -File windows\build_patched_engine.ps1 -Cuda

$ErrorActionPreference = "Stop"

$WindowsDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot   = Split-Path -Parent $WindowsDir
if (-not $SrcRoot) { $SrcRoot = Join-Path $RepoRoot "workspace" }

$BaseCommit = "d2f83055d6e3b379b5d34c4837122a918cf402c2"
$PatchFile  = if ($Target -eq "nemotron") {
    Join-Path $RepoRoot "patches\zen5-nemotron-256k-optimizations.patch"
} else {
    Join-Path $RepoRoot "patches\deepseek-v4-flash-fractal-256k.patch"
}
$Vq3Patch   = Join-Path $RepoRoot "patches\lut-vq3-64-avx512-vbmi.patch"
$HeadersDir = Join-Path $RepoRoot "speculative\cpp"
$LlamaDir   = Join-Path $SrcRoot "llama.cpp-src"
$OutDir     = Join-Path $WindowsDir "bin-modchip"
$BuildDir   = if ($Cuda) { Join-Path $LlamaDir "build-cuda" } else { Join-Path $LlamaDir "build-mingw" }

function Fail([string]$Msg) { Write-Host "[!] $Msg" -ForegroundColor Red; exit 1 }

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "  llama-modchip v2.5.0 Windows engine build (Fractal-64 Q2_0 + LUT-VQ3-64 VQ3_0)" -ForegroundColor Green
Write-Host ("  Target : {0} | Backend: {1}" -f $Target, $(if ($Cuda) { "CUDA sm_$Arch + CPU (MSVC)" } else { "CPU AVX-512 (MinGW-w64)" })) -ForegroundColor Green
Write-Host "================================================================================" -ForegroundColor Cyan

$git = (Get-Command git.exe -ErrorAction SilentlyContinue).Source
if (-not $git) { Fail "git.exe not in PATH." }
if (-not (Test-Path $PatchFile)) { Fail "Patch missing: $PatchFile" }

New-Item -ItemType Directory -Path $SrcRoot -Force | Out-Null

if (-not (Test-Path (Join-Path $LlamaDir ".git"))) {
    Write-Host "[*] Cloning llama.cpp ..." -ForegroundColor Cyan
    & $git clone --depth 1 https://github.com/ggml-org/llama.cpp $LlamaDir
    if ($LASTEXITCODE -ne 0) { Fail "git clone failed." }
}
Push-Location $LlamaDir
try {
    $head = (& $git rev-parse HEAD).Trim()
    if ($head -ne $BaseCommit) {
        Write-Host "[*] Fetching pinned commit $BaseCommit ..." -ForegroundColor Cyan
        & $git fetch --depth 1 origin $BaseCommit
        if ($LASTEXITCODE -ne 0) { Fail "git fetch of the pinned commit failed (network?)." }
        & $git checkout -q FETCH_HEAD
    }
    Write-Host "[*] Resetting working tree ..." -ForegroundColor Cyan
    & $git checkout -q -- .
    Remove-Item -Force -ErrorAction SilentlyContinue (Join-Path $LlamaDir "ggml\src\vq3_codebook.h")
    Remove-Item -Force -ErrorAction SilentlyContinue (Join-Path $LlamaDir "src\llama-kv-cache-dsv4.cpp")
    Remove-Item -Force -ErrorAction SilentlyContinue (Join-Path $LlamaDir "src\llama-kv-cache-dsv4.h")
    Remove-Item -Force -ErrorAction SilentlyContinue (Join-Path $LlamaDir "src\models\deepseek4.cpp")

    Write-Host ("[*] Applying {0} ..." -f (Split-Path -Leaf $PatchFile)) -ForegroundColor Cyan
    & $git apply $PatchFile
    if ($LASTEXITCODE -ne 0) { Fail "Patch does not apply. Expected llama.cpp at $BaseCommit." }

    if (Test-Path $Vq3Patch) {
        Write-Host "[*] Applying patches\lut-vq3-64-avx512-vbmi.patch (LUT-VQ3-64 VQ3_0) ..." -ForegroundColor Cyan
        & $git apply $Vq3Patch
        if ($LASTEXITCODE -ne 0) { Fail "LUT-VQ3-64 patch does not apply." }
    }

    Write-Host "[*] Copying RouteCrack headers into common\ ..." -ForegroundColor Cyan
    foreach ($h in "rc_predictor.h", "rc_stockfish.h", "rc_trie_v2.h") {
        Copy-Item (Join-Path $HeadersDir $h) (Join-Path $LlamaDir "common\$h") -Force
    }
    if ($Target -eq "nemotron") {
        Copy-Item (Join-Path $HeadersDir "ngram-map.cpp") (Join-Path $LlamaDir "common\ngram-map.cpp") -Force
    }

    # --- Windows / MSVC portability fixes ------------------------------------
    # 1) ops.cpp reads a Linux /dev/shm block; Windows uses the env vars instead.
    Write-Host "[*] Applying Windows portability fixes ..." -ForegroundColor Cyan
    $ops = Join-Path $LlamaDir "ggml\src\ggml-cpu\ops.cpp"
    $c = Get-Content -LiteralPath $ops -Raw
    $c = $c -replace '(?m)^#if defined\(__AVX2__\) \|\| defined\(__AVX512F__\)\r?\n#include <sys/mman\.h>',
                      "#if defined(__AVX2__) || defined(__AVX512F__)`r`n#ifndef _WIN32`r`n#include <sys/mman.h>"
    $c = $c -replace '(?m)^#include <unistd\.h>\r?\n\r?\nstruct llama_modchip_shm_t',
                      "#include <unistd.h>`r`n#endif`r`n`r`nstruct llama_modchip_shm_t"
    $c = $c -replace '(?ms)^(        int fd = open\("/dev/shm/llama_modchip\.bin", O_RDONLY\);.*?^        \})',
                      "#ifndef _WIN32`r`n`$1`r`n#endif"
    Set-Content -LiteralPath $ops -Value $c -Encoding UTF8 -NoNewline

    # 2) quants.c: MSVC has no __builtin_ctz
    $q = Join-Path $LlamaDir "ggml\src\ggml-cpu\arch\x86\quants.c"
    $c = Get-Content -LiteralPath $q -Raw
    if ($c -notmatch 'mc_ctz32') {
        $c = $c.Replace("#define UNUSED GGML_UNUSED",
"#define UNUSED GGML_UNUSED

// MSVC has no __builtin_ctz
#if defined(_MSC_VER)
#include <intrin.h>
static inline int mc_ctz32(unsigned int v) {
    unsigned long idx = 0;
    _BitScanForward(&idx, v);
    return (int) idx;
}
#else
#define mc_ctz32(v) __builtin_ctz(v)
#endif
")
        $c = $c.Replace('__builtin_ctz(m)', 'mc_ctz32(m)')
        Set-Content -LiteralPath $q -Value $c -Encoding UTF8 -NoNewline
    }

    # 3) ggml-cpu.c: __thread / __attribute__((aligned)) are GCC-only
    $gc = Join-Path $LlamaDir "ggml\src\ggml-cpu\ggml-cpu.c"
    $lines = [System.Collections.Generic.List[string]](Get-Content -LiteralPath $gc)
    if (-not ($lines -join "`n").Contains("MC_TLS")) {
        $idx = -1
        for ($i = 0; $i -lt $lines.Count; $i++) { if ($lines[$i] -match '^#define GGML_CACHE_ALIGN __attribute__') { $idx = $i; break } }
        if ($idx -ge 0) {
            $end = $idx + 1
            while ($lines[$end] -notmatch '^#endif') { $end++ }
            $lines.InsertRange($end + 1, [string[]]@(
                '',
                '// MSVC-compatible counterparts for the modchip additions',
                '#if defined(_WIN32)',
                '#include <io.h>',
                '#ifndef F_OK',
                '#define F_OK 0',
                '#endif',
                '#endif',
                '#if defined(_MSC_VER)',
                '#define MC_TLS __declspec(thread)',
                '#else',
                '#define MC_TLS __thread',
                '#endif'
            ))
        }
        for ($i = 0; $i -lt $lines.Count; $i++) {
            if ($lines[$i] -match 'static __thread ') { $lines[$i] = $lines[$i].Replace('static __thread ', 'static MC_TLS ') }
            if ($lines[$i] -match 's_q8_cache_buf\[GGML_Q8_CACHE_SLOTS\]' -and $lines[$i] -match '__attribute__') {
                $lines[$i] = $lines[$i].Replace(
                    'static uint8_t s_q8_cache_buf[GGML_Q8_CACHE_SLOTS][GGML_Q8_CACHE_SIZE] __attribute__((aligned(64)));',
                    '#if defined(_MSC_VER)' + "`r`n" +
                    '__declspec(align(64)) static uint8_t s_q8_cache_buf[GGML_Q8_CACHE_SLOTS][GGML_Q8_CACHE_SIZE];' + "`r`n" +
                    '#else' + "`r`n" +
                    'static uint8_t s_q8_cache_buf[GGML_Q8_CACHE_SLOTS][GGML_Q8_CACHE_SIZE] __attribute__((aligned(64)));' + "`r`n" +
                    '#endif')
            }
        }
        Set-Content -LiteralPath $gc -Value $lines -Encoding UTF8
    }

    # 4) llama-graph.cpp: patch references a C global from C++ code
    $lg = Join-Path $LlamaDir "src\llama-graph.cpp"
    $c = Get-Content -LiteralPath $lg -Raw
    if ($c -notmatch 'extern "C" int ggml_dsv4_skip_csa_comp') {
        $c = $c.Replace(
            '        extern int ggml_dsv4_skip_csa_comp;',
"#if defined(_MSC_VER)
        extern `"C`" int ggml_dsv4_skip_csa_comp;
#else
        extern int ggml_dsv4_skip_csa_comp;
#endif
")
        Set-Content -LiteralPath $lg -Value $c -Encoding UTF8 -NoNewline
    }

    if ($Clean -and (Test-Path $BuildDir)) { Remove-Item $BuildDir -Recurse -Force }

    $jobs = [math]::Max(1, (Get-CimInstance Win32_Processor | Select-Object -First 1).NumberOfCores)

    if ($Cuda) {
        $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
        if (-not (Test-Path $vswhere)) { Fail "vswhere.exe not found: install Visual Studio Build Tools with the C++ workload." }
        $vsPath = (& $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath).Trim()
        if (-not $vsPath) { Fail "No MSVC C++ toolset found." }
        $vcvars = Join-Path $vsPath "VC\Auxiliary\Build\vcvars64.bat"
        if (-not (Test-Path $vcvars)) { Fail "vcvars64.bat not found under $vsPath" }

        $cudaRoot = $env:CUDA_PATH
        if (-not $cudaRoot -or -not (Test-Path (Join-Path $cudaRoot "bin\nvcc.exe"))) {
            $cudaRoot = Get-ChildItem "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA" -Directory -ErrorAction SilentlyContinue |
                        Sort-Object Name -Descending | Select-Object -First 1 -ExpandProperty FullName
        }
        if (-not $cudaRoot) { Fail "CUDA Toolkit not found (winget install Nvidia.CUDA)." }

        $cmd = @"
call "$vcvars" >nul 2>&1
set "PATH=$cudaRoot\bin;%PATH%"
cd /d "$LlamaDir"
cmake -B "$BuildDir" -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=$Arch -DGGML_NATIVE=ON -DGGML_AVX512=ON -DGGML_AVX512_VBMI=ON -DGGML_AVX512_VNNI=ON -DGGML_OPENMP=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DBUILD_SHARED_LIBS=OFF "-DCMAKE_EXE_LINKER_FLAGS=/STACK:16777216"
if errorlevel 1 exit /b 1
cmake --build "$BuildDir" -j $jobs --target llama-server llama-cli llama-bench
"@
        $bat = Join-Path $env:TEMP "mcwin_build_cuda.bat"
        Set-Content -LiteralPath $bat -Value $cmd -Encoding ASCII
        Write-Host "[*] Building with MSVC + CUDA sm_$Arch ..." -ForegroundColor Cyan
        & cmd /c $bat
        if ($LASTEXITCODE -ne 0) { Fail "CUDA build failed." }
    } else {
        $cmake = (Get-Command cmake.exe -ErrorAction SilentlyContinue).Source
        $ninja = (Get-Command ninja.exe -ErrorAction SilentlyContinue).Source
        $gcc   = (Get-Command g++.exe   -ErrorAction SilentlyContinue).Source
        if (-not $cmake -or -not $ninja -or -not $gcc) { Fail "cmake.exe / ninja.exe / g++.exe not in PATH (install MinGW-w64 WinLibs + CMake + Ninja)." }
        $winFlags = "-D_WIN32_WINNT=0x0A00 -DNTDDI_VERSION=0x0A000000"
        Write-Host "[*] Configuring ..." -ForegroundColor Cyan
        & cmake -B $BuildDir -G Ninja `
            -DCMAKE_BUILD_TYPE=Release `
            -DGGML_NATIVE=ON `
            -DGGML_OPENMP=OFF `
            -DLLAMA_CURL=OFF `
            -DLLAMA_BUILD_TESTS=OFF `
            -DBUILD_SHARED_LIBS=OFF `
            "-DCMAKE_EXE_LINKER_FLAGS=-static -Wl,--stack,16777216" `
            "-DCMAKE_C_FLAGS=$winFlags" `
            "-DCMAKE_CXX_FLAGS=$winFlags"
        if ($LASTEXITCODE -ne 0) { Fail "cmake configure failed." }
        Write-Host "[*] Building (a few minutes) ..." -ForegroundColor Cyan
        & cmake --build $BuildDir -j $jobs --target llama-server llama-cli llama-bench
        if ($LASTEXITCODE -ne 0) { Fail "Build failed." }
    }

    New-Item -ItemType Directory -Path $OutDir -Force | Out-Null
    foreach ($b in "llama-server.exe", "llama-cli.exe", "llama-bench.exe") {
        $p = Join-Path $BuildDir "bin\$b"
        if (Test-Path $p) { Copy-Item $p $OutDir -Force }
    }
    Get-ChildItem (Join-Path $BuildDir "bin") -Filter "*.dll" -ErrorAction SilentlyContinue | Copy-Item -Destination $OutDir -Force

    Write-Host ("[OK] Engine written to {0}" -f $OutDir) -ForegroundColor Green
} finally {
    Pop-Location
}
