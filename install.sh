#!/usr/bin/env bash
# ==============================================================================
# install.sh — Interactive Zero-to-Running Installer for llama-modchip (v2.5.3)
# ==============================================================================
# Usage:
#   One-liner (from scratch):
#     curl -sL https://raw.githubusercontent.com/synchopate/llama-modchip/main/install.sh | bash
#
#   From cloned repo:
#     ./install.sh [options]
#
# Options:
#   --workspace <dir>      Target build/model directory (default: ./workspace)
#   --preset <id>          Model preset: dsv4-maxq | dsv4-vq3 | dsv4-fractal | dsv4-ablit |
#                                        nemotron-vq3 | nemotron-fractal | nemotron-ablit | nemotron-q4
#   --model <path>         Use an existing local GGUF file instead of downloading
#   --skip-model           Install engine & scripts without downloading GGUF weights
#   --prebuilt             Download CPU-matched Linux x86_64 binary (-avx512 / -avx2)
#   --source               Compile patched engine locally from source (requires cmake/g++)
#   --yes, -y              Non-interactive mode: accept hardware-matched aligned defaults
#   --accept-abliterated   Confirm opt-in consent when selecting an abliterated preset in --yes mode
#   --corpus <dir>         Optional path to your codebase for Nemotron RouteCrack n-grams
#   --port <port>          llama-server port (default: 8789)
# ==============================================================================
set -euo pipefail

# If piped via `curl ... | bash`, bootstrap the repository first and re-exec
SCRIPT_SOURCE="${BASH_SOURCE[0]:-}"
if [[ -z "$SCRIPT_SOURCE" ]] || [[ ! -f "$(dirname "$SCRIPT_SOURCE")/patches/zen5-nemotron-256k-optimizations.patch" ]]; then
    echo "=============================================================================="
    echo " [*] Bootstrapping llama-modchip repository..."
    echo "=============================================================================="
    if [[ ! -d "./llama-modchip" ]]; then
        if command -v git >/dev/null 2>&1; then
            git clone https://github.com/synchopate/llama-modchip.git ./llama-modchip
        else
            mkdir -p ./llama-modchip
            curl -fsSL https://github.com/synchopate/llama-modchip/archive/refs/heads/main.tar.gz \
                | tar -xz --strip-components=1 -C ./llama-modchip
        fi
    fi
    cd ./llama-modchip
    exec bash ./install.sh "$@"
fi

REPO_ROOT="$(cd "$(dirname "$SCRIPT_SOURCE")" && pwd)"
WORKSPACE_DIR="${WORKSPACE_DIR:-${REPO_ROOT}/workspace}"
MODEL_PATH="${MODEL_PATH:-}"
MODEL_EXPLICIT=0
[[ -n "$MODEL_PATH" ]] && MODEL_EXPLICIT=1
MODEL_PRESET="${MODEL_PRESET:-}"
SKIP_MODEL="${SKIP_MODEL:-0}"
USER_TARGET_MODEL="${TARGET_MODEL:-}"
ENGINE_MODE_EXPLICIT=0
if [[ -n "${USE_PREBUILT:-}" || -n "$USER_TARGET_MODEL" ]]; then
    ENGINE_MODE_EXPLICIT=1
fi
USE_PREBUILT="${USE_PREBUILT:-0}"
NONINTERACTIVE="${NONINTERACTIVE:-0}"
ACCEPT_ABLITERATED="${ACCEPT_ABLITERATED:-0}"
USER_CORPUS_DIR=""
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${SERVER_PORT:-8789}"
GATEWAY_HOST="${GATEWAY_HOST:-127.0.0.1}"
GATEWAY_PORT="${GATEWAY_PORT:-8787}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --workspace)
            WORKSPACE_DIR="$(realpath -m "$2")"
            shift 2
            ;;
        --preset)
            MODEL_PRESET="$2"
            shift 2
            ;;
        --model)
            MODEL_PATH="$(realpath -m "$2")"
            MODEL_EXPLICIT=1
            shift 2
            ;;
        --skip-model)
            SKIP_MODEL=1
            shift
            ;;
        --prebuilt)
            USE_PREBUILT=1
            ENGINE_MODE_EXPLICIT=1
            shift
            ;;
        --source)
            USE_PREBUILT=0
            ENGINE_MODE_EXPLICIT=1
            shift
            ;;
        --yes|-y)
            NONINTERACTIVE=1
            shift
            ;;
        --accept-abliterated)
            ACCEPT_ABLITERATED=1
            shift
            ;;
        --corpus)
            USER_CORPUS_DIR="$(realpath -m "$2")"
            shift 2
            ;;
        --port)
            SERVER_PORT="$2"
            shift 2
            ;;
        -h|--help)
            sed -n '2,24p' "$0"
            exit 0
            ;;
        *)
            echo "[!] Unknown option: $1 (use --help for usage)"
            exit 1
            ;;
    esac
done

mkdir -p "$WORKSPACE_DIR/bin" "$WORKSPACE_DIR/models" "$WORKSPACE_DIR/routecrack" "$WORKSPACE_DIR/cache"

check_required_commands() {
    local missing=()
    local cmd
    for cmd in "$@"; do
        if ! command -v "$cmd" >/dev/null 2>&1; then
            missing+=("$cmd")
        fi
    done
    if [[ ${#missing[@]} -eq 0 ]]; then
        return 0
    fi
    echo "[!] Missing required system tools: ${missing[*]}"
    if [[ "${INSTALL_DEPS:-0}" == "1" ]]; then
        if command -v apt-get >/dev/null 2>&1; then
            sudo apt-get update && sudo apt-get install -y build-essential cmake git curl python3 libomp-dev
        elif command -v dnf >/dev/null 2>&1; then
            sudo dnf install -y gcc-c++ cmake git curl python3 libomp-devel
        elif command -v pacman >/dev/null 2>&1; then
            sudo pacman -Sy --noconfirm base-devel cmake git curl python openmp
        fi
    else
        echo "    Install them with your package manager (or re-run with INSTALL_DEPS=1 ./install.sh):"
        echo "      Debian/Ubuntu : sudo apt-get install -y curl python3 build-essential cmake git libomp-dev"
        echo "      Fedora/RHEL   : sudo dnf install -y curl python3 gcc-c++ cmake git libomp-devel"
        echo "      Arch Linux    : sudo pacman -S curl python base-devel cmake git openmp"
        exit 1
    fi
}

echo "=============================================================================="
echo " STEP 1/5: HARDWARE READINESS AUDIT & CONFIGURATION"
echo "=============================================================================="
# Base runtime dependencies (curl + python3 only; compiler toolchain checked only if building from source)
check_required_commands curl python3

python3 "${REPO_ROOT}/scripts/check_cpu_readiness.py" || true

# Detect physical CPU core count, total host RAM, and x86_64 SIMD prebuilt compatibility
PHYS_CORES="$(python3 -c '
import os, platform, subprocess
cores = set()
try:
    with open("/proc/cpuinfo") as f:
        phys_id, core_id = "0", None
        for line in f:
            if line.startswith("physical id"): phys_id = line.split(":")[1].strip()
            elif line.startswith("core id"):
                core_id = line.split(":")[1].strip()
                cores.add((phys_id, core_id))
except Exception:
    pass
n = len(cores)
if n == 0 and platform.system() == "Darwin":
    for key in ("hw.perflevel0.physicalcpu", "hw.physicalcpu"):
        try:
            v = int(subprocess.check_output(["sysctl", "-n", key], stderr=subprocess.DEVNULL).strip())
            if v > 0:
                n = v
                break
        except Exception:
            pass
print(n if n > 0 else (os.cpu_count() or 8))
')"

TOTAL_RAM_GB="$(python3 -c '
import glob, platform, subprocess
ram_gb = 0
try:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemTotal:"):
                ram_gb = int(line.split()[1]) // (1024 * 1024)
                break
except Exception:
    pass
if ram_gb == 0 and platform.system() == "Darwin":
    try:
        ram_gb = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], stderr=subprocess.DEVNULL).strip()) // (1024 ** 3)
    except Exception:
        pass
if ram_gb <= 0:
    ram_gb = 32
if ram_gb > 65536:
    try:
        numa_kb = 0
        for p in glob.glob("/sys/devices/system/node/node*/meminfo"):
            with open(p) as nf:
                for nline in nf:
                    if "MemTotal:" in nline:
                        numa_kb += int(nline.split()[3])
                        break
        if numa_kb > 0:
            ram_gb = numa_kb // (1024 * 1024)
    except Exception:
        pass
print(ram_gb)
')"

detect_x86_prebuilt_asset() {
    local flags=""
    if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
        printf '%s\n' ""
        return
    fi
    if [[ -r /proc/cpuinfo ]]; then
        flags="$(grep -m1 -E '^flags' /proc/cpuinfo || true)"
    fi
    if echo "$flags" | grep -q "avx512f" && echo "$flags" | grep -qE "avx512_?vbmi"; then
        printf '%s\n' "llama-server-modchip-linux-x86_64-avx512"
    elif echo "$flags" | grep -q "avx2"; then
        printf '%s\n' "llama-server-modchip-linux-x86_64-avx2"
    else
        printf '%s\n' ""
    fi
}

X86_PREBUILT_ASSET="$(detect_x86_prebuilt_asset)"
HAS_AVX512_VBMI=0
[[ "$X86_PREBUILT_ASSET" == *-avx512 ]] && HAS_AVX512_VBMI=1

# Detect AMD Zen 1 / Zen 2 (CPU family 23), where BMI2 PDEP is microcoded (~18 cycles)
IS_AMD_ZEN12=0
if [[ -r /proc/cpuinfo ]] && grep -qi "AuthenticAMD" /proc/cpuinfo && grep -qE '^cpu family[[:space:]]*:[[:space:]]*23$' /proc/cpuinfo; then
    IS_AMD_ZEN12=1
fi

IS_INTERACTIVE=0
if [[ "$NONINTERACTIVE" != "1" && -t 1 && -r /dev/tty ]]; then
    IS_INTERACTIVE=1
fi

# Hardware-aware safety-aligned default preset:
#   >= 60 GB RAM -> DeepSeek-V4-Flash-0731-MaxQ-Fractal (safety-aligned single-pass 284B default)
#   <  60 GB RAM -> Nemotron-3.5-30B-A3B-VQ3-64 (on AVX-512 VBMI, or <=18 GB RAM except Zen 1/2)
#                   else Nemotron-3.5-30B-A3B-Fractal-64 (AVX2 / AMD Zen 1-2 / ARM64)
if [[ "$TOTAL_RAM_GB" -ge 60 ]]; then
    DEFAULT_PRESET="dsv4-maxq"
    DEFAULT_MENU_CHOICE="1"
elif [[ "$HAS_AVX512_VBMI" == "1" ]] || [[ "$TOTAL_RAM_GB" -le 18 && "$IS_AMD_ZEN12" == "0" ]]; then
    DEFAULT_PRESET="nemotron-vq3"
    DEFAULT_MENU_CHOICE="5"
else
    DEFAULT_PRESET="nemotron-fractal"
    DEFAULT_MENU_CHOICE="6"
fi

if [[ "$MODEL_EXPLICIT" == "0" && -z "$MODEL_PRESET" ]]; then
    if [[ "$SKIP_MODEL" == "0" && "$IS_INTERACTIVE" == "1" ]]; then
        echo ""
        echo "------------------------------------------------------------------------------"
        echo " MODEL SELECTION (Detected Host RAM: ~${TOTAL_RAM_GB} GiB | Physical Cores: ${PHYS_CORES})"
        echo "------------------------------------------------------------------------------"
        echo "  1) DeepSeek-V4-Flash-0731-MaxQ-Fractal        [284B MoE · 56.93 GiB · 3 shards]"
        echo "     Default 284B (64 GB+ RAM, safety-aligned) — Single-pass Q6_K + Fractal-64"
        echo "  2) DeepSeek-V4-Flash-0731-VQ3                 [284B MoE · 56.40 GiB · 3 shards]"
        echo "     Safety-aligned 284B — Hybrid LUT-VQ3-64 + Fractal-64 (AVX-512 VBMI, 64 GB+)"
        echo "  3) DeepSeek-V4-Flash-0731-Fractal             [284B MoE · 56.57 GiB · 3 shards]"
        echo "     Safety-aligned 284B — Pure Fractal-64 baseline (AVX2 / Apple Silicon 64 GB+)"
        echo "  4) DeepSeek-V4-Flash-0731-Abliterated-Fractal [284B MoE · 56.93 GiB · 3 shards]"
        echo "     [OPT-IN UNCENSORED] 284B single-pass Q6_K + Fractal-64 (refusal removed)"
        echo "  5) NVIDIA-Nemotron-3.5-30B-A3B-VQ3-64         [30B MoE  · 13.30 GiB · 1 file]"
        echo "     Default 30B for AVX-512 VBMI & 16–32 GB RAM (safety-aligned, 100/100 Gate)"
        echo "  6) NVIDIA-Nemotron-3.5-30B-A3B-Fractal-64     [30B MoE  · 14.36 GiB · 1 file]"
        echo "     Default 30B for AVX2 / AMD Zen 1–2 / ARM64 CPUs (safety-aligned, 100/100 Gate)"
        echo "  7) NVIDIA-Nemotron-3.5-30B-A3B-Abliterated-VQ3[30B MoE  · 13.30 GiB · 1 file]"
        echo "     [OPT-IN UNCENSORED] 30B VQ3-64 (Darkstar BF16 abliterated, refusal removed)"
        echo "  8) NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0 [30B MoE  · 17.60 GiB · 1 file]"
        echo "     Official 4.50-bpw Q4_0 baseline checkpoint (24 GB+ RAM)"
        echo "------------------------------------------------------------------------------"
        while true; do
            printf " Select model [1-8, Enter=%s]: " "$DEFAULT_MENU_CHOICE"
            read -r USER_MODEL_CHOICE < /dev/tty || USER_MODEL_CHOICE="$DEFAULT_MENU_CHOICE"
            USER_MODEL_CHOICE="${USER_MODEL_CHOICE:-$DEFAULT_MENU_CHOICE}"
            case "$USER_MODEL_CHOICE" in
                1|dsv4-maxq|deepseek)                   MODEL_PRESET="dsv4-maxq"; break ;;
                2|dsv4-vq3)                             MODEL_PRESET="dsv4-vq3"; break ;;
                3|dsv4-fractal)                         MODEL_PRESET="dsv4-fractal"; break ;;
                4|dsv4-ablit|dsv4-abliterated)          MODEL_PRESET="dsv4-ablit"; break ;;
                5|nemotron-vq3)                         MODEL_PRESET="nemotron-vq3"; break ;;
                6|nemotron-fractal)                     MODEL_PRESET="nemotron-fractal"; break ;;
                7|nemotron-ablit|nemotron-abliterated)  MODEL_PRESET="nemotron-ablit"; break ;;
                8|nemotron-q4)                          MODEL_PRESET="nemotron-q4"; break ;;
                *)
                    echo "[!] Unrecognized choice '${USER_MODEL_CHOICE}'. Please enter a number from 1 to 8."
                    ;;
            esac
        done
    else
        MODEL_PRESET="$DEFAULT_PRESET"
    fi
fi

MODEL_FAMILY="nemotron"
IS_ABLITERATED=0
MODEL_SHARDS=()
MODEL_SHARD_SIZES=()
MODEL_SHARD_SHA256S=()
MODEL_SUBDIR=""

if [[ "$MODEL_EXPLICIT" == "1" ]]; then
    MODEL_LABEL="$(basename "$MODEL_PATH")"
    if [[ "$MODEL_PATH" == *[Dd]eep[Ss]eek* || "$MODEL_PATH" == *dsv4* ]]; then
        MODEL_FAMILY="dsv4"
    fi
    if [[ "$MODEL_PATH" == *[Aa]bliterated* ]]; then
        IS_ABLITERATED=1
    fi
else
    case "$MODEL_PRESET" in
        1|dsv4-maxq|deepseek)
            MODEL_PRESET="dsv4-maxq"
            MODEL_FAMILY="dsv4"
            MODEL_LABEL="DeepSeek-V4-Flash-0731-MaxQ-Fractal (284B · 56.93 GiB · 3 shards)"
            MODEL_REPO="Synchopate/DeepSeek-V4-Flash-0731-MaxQ-Fractal-64GB-CPU-GGUF"
            MODEL_REV="3619ba39ae7ac349548ce2cdff1e1c2ca3a71cb5"
            MODEL_SUBDIR="dsv4-maxq-fractal"
            MODEL_SHARDS=(
                "DeepSeek-V4-Flash-0731-MaxQ-Fractal-00001-of-00003.gguf"
                "DeepSeek-V4-Flash-0731-MaxQ-Fractal-00002-of-00003.gguf"
                "DeepSeek-V4-Flash-0731-MaxQ-Fractal-00003-of-00003.gguf"
            )
            MODEL_SHARD_SIZES=(5257952 36817932704 24306173504)
            MODEL_SHARD_SHA256S=(
                "0ae64076077f88f5750d7e78403bbf0304a386c86912a47655eb3aaf627d2ce4"
                "155fc6c239969ed2c470c6a7b9c474bc0c17922980e8111f614c88dd65e63aa5"
                "15f2e256b9b1c3e106a52ddc2e093f2264571ffdbce3ba0010f54f9ecce57252"
            )
            ;;
        2|dsv4-vq3)
            MODEL_PRESET="dsv4-vq3"
            MODEL_FAMILY="dsv4"
            MODEL_LABEL="DeepSeek-V4-Flash-0731-VQ3 (284B · 56.40 GiB · 3 shards)"
            MODEL_REPO="Synchopate/DeepSeek-V4-Flash-0731-VQ3-64GB-CPU-GGUF"
            MODEL_REV="dd470e619f2a4926adbdd471010423d4b686c1b4"
            MODEL_SUBDIR="dsv4-vq3"
            MODEL_SHARDS=(
                "DeepSeek-V4-Flash-0731-VQ3-00001-of-00003.gguf"
                "DeepSeek-V4-Flash-0731-VQ3-00002-of-00003.gguf"
                "DeepSeek-V4-Flash-0731-VQ3-00003-of-00003.gguf"
            )
            MODEL_SHARD_SIZES=(5258016 36362375584 24191223360)
            MODEL_SHARD_SHA256S=(
                "00ae0e9b0316ece06cd75afb7d0d49bcfa01fcfca3f3eb8f37d1e2b7c65e25bb"
                "6b1d75cbd3d7fa7668820d259ae205c2ad5078123b92b01bc1d3f8863dd7e957"
                "5f811f21a60cc1aa210afa2eda8aacc53205815f429f8bacdc28cc6371da8ec5"
            )
            ;;
        3|dsv4-fractal)
            MODEL_PRESET="dsv4-fractal"
            MODEL_FAMILY="dsv4"
            MODEL_LABEL="DeepSeek-V4-Flash-0731-Fractal (284B · 56.57 GiB · 3 shards)"
            MODEL_REPO="Synchopate/DeepSeek-V4-Flash-0731-Fractal-64GB-CPU-GGUF"
            MODEL_REV="a0758c031751ef02e3cd63d330183e5525bd8cc9"
            MODEL_SUBDIR="dsv4-fractal"
            MODEL_SHARDS=(
                "DeepSeek-V4-Flash-0731-Fractal-00001-of-00003.gguf"
                "DeepSeek-V4-Flash-0731-Fractal-00002-of-00003.gguf"
                "DeepSeek-V4-Flash-0731-Fractal-00003-of-00003.gguf"
            )
            MODEL_SHARD_SIZES=(5258016 36472476064 24261609024)
            MODEL_SHARD_SHA256S=(
                "6f8e5e93faf86e61d2f378170139299477dee53a8bb8798011cf35566e30db25"
                "f191c7d2cf34cc9ca7786bfb41d1b3a2b1bf18593db5594ae26e5b25087f1030"
                "2ad07ab072be090bea938b0c407ac03bc6f9fb825ea289eda43a8dbac4844182"
            )
            ;;
        4|dsv4-ablit|dsv4-abliterated)
            MODEL_PRESET="dsv4-ablit"
            MODEL_FAMILY="dsv4"
            IS_ABLITERATED=1
            MODEL_LABEL="DeepSeek-V4-Flash-0731-Abliterated-Fractal (284B · 56.93 GiB · 3 shards)"
            MODEL_REPO="Synchopate/DeepSeek-V4-Flash-0731-Abliterated-Fractal-64GB-CPU-GGUF"
            MODEL_REV="2ebd8a1e5730b1e43ab6a84178e9b949d223f3e9"
            MODEL_SUBDIR="dsv4-abliterated-fractal"
            MODEL_SHARDS=(
                "DeepSeek-V4-Flash-0731-Abliterated-Fractal-00001-of-00003.gguf"
                "DeepSeek-V4-Flash-0731-Abliterated-Fractal-00002-of-00003.gguf"
                "DeepSeek-V4-Flash-0731-Abliterated-Fractal-00003-of-00003.gguf"
            )
            MODEL_SHARD_SIZES=(5258016 36817932704 24306173504)
            MODEL_SHARD_SHA256S=(
                "69b5bf614a56c119670b1b9535b71e46987e917928fbf09ac4631c3d202ef87f"
                "6e3f499c5b6311a8cdd225243c457f45e7672fded595aba37dc3f15720db8f10"
                "0063f3a9566cdc5a1c1e0b1dc14fefd0da408a9dac3f0b692d676b86cc8f46e3"
            )
            ;;
        5|nemotron-vq3)
            MODEL_PRESET="nemotron-vq3"
            MODEL_FAMILY="nemotron"
            MODEL_LABEL="NVIDIA-Nemotron-3.5-30B-A3B-VQ3-64 (30B · 13.30 GiB)"
            MODEL_REPO="Synchopate/NVIDIA-Nemotron-3.5-30B-A3B-VQ3-64-CPU-GGUF"
            MODEL_REV="c3dd3a506502fd9accb5c6dc9055a7c3dd4aba9a"
            MODEL_SHARDS=("NVIDIA-Nemotron-3.5-30B-A3B-VQ3-64.gguf")
            MODEL_SHARD_SIZES=(14282559040)
            MODEL_SHARD_SHA256S=("d273709749aa88fd5a5af59ed3a2d55354283e1ae8ae7483f85737dd01d42b76")
            ;;
        6|nemotron-fractal)
            MODEL_PRESET="nemotron-fractal"
            MODEL_FAMILY="nemotron"
            MODEL_LABEL="NVIDIA-Nemotron-3.5-30B-A3B-Fractal-64 (30B · 14.36 GiB)"
            MODEL_REPO="Synchopate/NVIDIA-Nemotron-3.5-30B-A3B-Fractal-64-CPU-GGUF"
            MODEL_REV="fab73258754e9f4a2d21077af5300883bae201c0"
            MODEL_SHARDS=("NVIDIA-Nemotron-3.5-30B-A3B-Fractal-64.gguf")
            MODEL_SHARD_SIZES=(15420034624)
            MODEL_SHARD_SHA256S=("71c22562ab908d5d3373e3a010544e977af2499e3cc476fc7646d402a8a550ee")
            ;;
        7|nemotron-ablit|nemotron-abliterated)
            MODEL_PRESET="nemotron-ablit"
            MODEL_FAMILY="nemotron"
            IS_ABLITERATED=1
            MODEL_LABEL="NVIDIA-Nemotron-3.5-30B-A3B-Abliterated-VQ3-64 (30B · 13.30 GiB)"
            MODEL_REPO="Synchopate/NVIDIA-Nemotron-3.5-30B-A3B-Abliterated-VQ3-64-CPU-GGUF"
            MODEL_REV="db9a06beb7c90fc1e5fd5570a874ac7f476a85bb"
            MODEL_SHARDS=("NVIDIA-Nemotron-3.5-30B-A3B-Abliterated-VQ3-64.gguf")
            MODEL_SHARD_SIZES=(14282559488)
            MODEL_SHARD_SHA256S=("456d5561acf609850c816fa15de7dbff93bd92430f64aadd85682dacc6f02fa4")
            ;;
        8|nemotron-q4)
            MODEL_PRESET="nemotron-q4"
            MODEL_FAMILY="nemotron"
            MODEL_LABEL="NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0 (30B · 17.60 GiB)"
            MODEL_REPO="ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF"
            MODEL_REV="8cd082170ed313484be1cf44f256a5caa2cfdad3"
            MODEL_SHARDS=("NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0.gguf")
            MODEL_SHARD_SIZES=(18898091584)
            MODEL_SHARD_SHA256S=("61f87e75974e4b535dcdf9aad056541a9514f1dfa4538b463b081d19b7a00e3c")
            ;;
        *)
            echo "[!] ERROR: Unknown --preset '${MODEL_PRESET}'. Valid presets: dsv4-maxq, dsv4-vq3, dsv4-fractal, dsv4-ablit, nemotron-vq3, nemotron-fractal, nemotron-ablit, nemotron-q4" >&2
            exit 1
            ;;
    esac
    DEFAULT_MODEL_NAME="${MODEL_SHARDS[0]}"
fi

if [[ "$MODEL_FAMILY" == "dsv4" && "$TOTAL_RAM_GB" -lt 60 ]]; then
    echo "[!] WARNING: DeepSeek-V4 284B requires ~58.3–59.1 GiB anonymous-heap VmRSS (64 GB+ host RAM required)."
fi

if [[ "$IS_ABLITERATED" == "1" ]]; then
    ALIGNED_ALT="dsv4-maxq"
    [[ "$MODEL_FAMILY" == "nemotron" ]] && ALIGNED_ALT="nemotron-vq3"
    echo ""
    echo "=============================================================================="
    echo " [!] DISCLAIMER — OPT-IN ABLITERATED / UNCENSORED MODEL CHECKPOINT"
    echo "=============================================================================="
    echo " Selected model (${MODEL_LABEL}) derives from an abliterated base"
    echo " checkpoint in which refusal-mediating activation directions were removed."
    echo " Standard upstream safety refusal filters are disabled."
    echo "   • Scope: Authorized security research, red-teaming, CTF, code auditing,"
    echo "     and local offline development only."
    echo "   • Deployment: Do NOT expose this model to unmoderated public networks"
    echo "     without external access control and safety filtering."
    echo "   • Responsibility: You assume full legal and operational responsibility for"
    echo "     all generated outputs. For standard safety alignment, use"
    echo "     './install.sh --preset ${ALIGNED_ALT}'."
    echo "=============================================================================="
    if [[ "$ACCEPT_ABLITERATED" != "1" ]]; then
        if [[ "$IS_INTERACTIVE" == "1" ]]; then
            printf " Proceed with uncensored/abliterated checkpoint? [y/N]: "
            read -r ABLIT_CONFIRM < /dev/tty || ABLIT_CONFIRM=""
            if [[ ! "$ABLIT_CONFIRM" =~ ^[Yy]([Ee][Ss])?$ ]]; then
                echo "[i] Aborted. Re-run './install.sh --preset ${ALIGNED_ALT}' for the standard safety-aligned model."
                exit 1
            fi
        else
            echo "[!] ERROR: Abliterated/uncensored presets require explicit consent." >&2
            echo "    Pass '--accept-abliterated' when running with --yes / non-interactive mode," >&2
            echo "    or use the safety-aligned preset '--preset ${ALIGNED_ALT}'." >&2
            exit 1
        fi
    fi
fi

# Engine Installation Mode Selection (Prebuilt binary vs Source compile + Patch target)
BUILD_TARGET_MODEL="${USER_TARGET_MODEL:-}"
if [[ -z "$BUILD_TARGET_MODEL" ]]; then
    if [[ "$(uname -s)" == "Darwin" && "$MODEL_FAMILY" == "nemotron" ]]; then
        BUILD_TARGET_MODEL="nemotron"
    else
        BUILD_TARGET_MODEL="dsv4"
    fi
fi

if [[ "$ENGINE_MODE_EXPLICIT" == "0" ]]; then
    if [[ -n "$X86_PREBUILT_ASSET" ]]; then
        if [[ "$IS_INTERACTIVE" == "1" ]]; then
            echo ""
            echo "------------------------------------------------------------------------------"
            echo " ENGINE INSTALLATION MODE (Linux x86_64 — CPUID matched: ${X86_PREBUILT_ASSET})"
            echo "------------------------------------------------------------------------------"
            echo "  1) Download precompiled binary (${X86_PREBUILT_ASSET}) [Recommended, ~5s]"
            echo "     Supports BOTH DeepSeek-V4 284B & Nemotron-30B (Fractal-64 + LUT-VQ3-64);"
            echo "     no C++ compiler (cmake/g++) required."
            echo "  2) Compile from source — Unified DeepSeek-V4 284B + Nemotron-30B engine"
            echo "     Applies deepseek-v4-flash-fractal-256k.patch + dsv4-ipc-hardening.patch + lut-vq3-64-avx512-vbmi.patch."
            echo "  3) Compile from source — Dedicated Nemotron-30B + offline RouteCrack builder"
            echo "     Applies zen5-nemotron-256k-optimizations.patch + lut-vq3-64-avx512-vbmi.patch."
            echo "------------------------------------------------------------------------------"
            printf " Select engine mode [1-3, Enter=1]: "
            read -r USER_ENGINE_CHOICE < /dev/tty || USER_ENGINE_CHOICE=""
            USER_ENGINE_CHOICE="${USER_ENGINE_CHOICE:-1}"
            case "$USER_ENGINE_CHOICE" in
                1) USE_PREBUILT=1 ;;
                2) USE_PREBUILT=0; BUILD_TARGET_MODEL="dsv4" ;;
                3) USE_PREBUILT=0; BUILD_TARGET_MODEL="nemotron" ;;
                *)
                    echo "[!] Unrecognized choice '${USER_ENGINE_CHOICE}'; using precompiled binary."
                    USE_PREBUILT=1
                    ;;
            esac
        else
            # Non-interactive default on supported Linux x86_64: fast prebuilt binary (zero compiler requirement)
            USE_PREBUILT=1
        fi
    else
        USE_PREBUILT=0
        if [[ "$IS_INTERACTIVE" == "1" && "$(uname -s)" != "Darwin" ]]; then
            DEF_SRC_OPT="1"
            [[ "$BUILD_TARGET_MODEL" == "nemotron" ]] && DEF_SRC_OPT="2"
            echo ""
            echo "------------------------------------------------------------------------------"
            echo " SOURCE ENGINE TARGET SELECTION ($(uname -s)/$(uname -m))"
            echo "------------------------------------------------------------------------------"
            echo "  1) Unified DeepSeek-V4 284B + Nemotron-30B + Fractal-64/VQ3 (TARGET_MODEL=dsv4)"
            echo "  2) Dedicated Nemotron-30B + offline RouteCrack builder      (TARGET_MODEL=nemotron)"
            echo "------------------------------------------------------------------------------"
            printf " Select patch target [1-2, Enter=%s]: " "$DEF_SRC_OPT"
            read -r USER_SRC_CHOICE < /dev/tty || USER_SRC_CHOICE=""
            USER_SRC_CHOICE="${USER_SRC_CHOICE:-$DEF_SRC_OPT}"
            case "$USER_SRC_CHOICE" in
                1) BUILD_TARGET_MODEL="dsv4" ;;
                2) BUILD_TARGET_MODEL="nemotron" ;;
            esac
        fi
    fi
fi

# Enforce dsv4 patch target if compiling from source for a DeepSeek-V4 model
if [[ "$USE_PREBUILT" == "0" && "$MODEL_FAMILY" == "dsv4" && "$BUILD_TARGET_MODEL" != "dsv4" ]]; then
    echo "[i] Switching source build target to TARGET_MODEL=dsv4 (required for DeepSeek-V4 architecture)."
    BUILD_TARGET_MODEL="dsv4"
fi

# Check compiler toolchain ONLY when compiling from source
if [[ "$USE_PREBUILT" == "0" ]]; then
    check_required_commands git cmake g++
fi

# Check 2MB Transparent HugePages on Linux (our patch uses madvise(MADV_HUGEPAGE), so [madvise] or [always] is sufficient)
if [[ -f /sys/kernel/mm/transparent_hugepage/enabled ]]; then
    THP_STATE="$(cat /sys/kernel/mm/transparent_hugepage/enabled)"
    if [[ "$THP_STATE" == *"[madvise]"* ]] || [[ "$THP_STATE" == *"[always]"* ]]; then
        echo "[+] 2MB Transparent HugePages (THP) active: ${THP_STATE}"
    else
        echo "[i] Tip: Enable madvise Transparent HugePages for lower TLB overhead:"
        echo "    echo madvise | sudo tee /sys/kernel/mm/transparent_hugepage/enabled"
        echo "    echo defer+madvise | sudo tee /sys/kernel/mm/transparent_hugepage/defrag"
    fi
fi

echo ""
echo "=============================================================================="
echo " STEP 2/5: INSTALLING PATCHED LLAMA-MODCHIP ENGINE"
echo "=============================================================================="
LLAMA_BUILD_DIR="${WORKSPACE_DIR}/llama.cpp"

PREBUILT_OK=0
if [[ "$USE_PREBUILT" == "1" && -n "$X86_PREBUILT_ASSET" ]]; then
    ASSET_URL="https://github.com/synchopate/llama-modchip/releases/download/v2.5.3/${X86_PREBUILT_ASSET}"
    FALLBACK_ASSET_URL="https://huggingface.co/Synchopate/DeepSeek-V4-Flash-0731-MaxQ-Fractal-64GB-CPU-GGUF/resolve/3619ba39ae7ac349548ce2cdff1e1c2ca3a71cb5/${X86_PREBUILT_ASSET}"
    EXPECTED_BIN_SHA256="09f2a1accacb1454aaf0d9a8d75dda8f7cca36df81138eb57bfee45137ef07e4"
    [[ "$X86_PREBUILT_ASSET" == *-avx512 ]] && EXPECTED_BIN_SHA256="85fe1e2a780ce7c88d99cf5f6fa384c2726dfeee247460924d6bf937efd19597"
    echo "[*] CPUID selected portable precompiled binary: ${X86_PREBUILT_ASSET}"
    if curl -fL --progress-bar -o "${WORKSPACE_DIR}/bin/llama-server-nemotron.part" "$ASSET_URL" \
        || curl -fL --progress-bar -o "${WORKSPACE_DIR}/bin/llama-server-nemotron.part" "$FALLBACK_ASSET_URL"; then
        ACTUAL_BIN_SHA256="$(sha256sum "${WORKSPACE_DIR}/bin/llama-server-nemotron.part" | awk '{print $1}')"
        chmod +x "${WORKSPACE_DIR}/bin/llama-server-nemotron.part"
        SMOKE_ERR=""
        if [[ "$ACTUAL_BIN_SHA256" != "$EXPECTED_BIN_SHA256" ]]; then
            rm -f "${WORKSPACE_DIR}/bin/llama-server-nemotron.part"
            echo "[!] SHA-256 mismatch on prebuilt binary (expected ${EXPECTED_BIN_SHA256}, got ${ACTUAL_BIN_SHA256}); falling back to local source build..."
            check_required_commands git cmake g++
        elif ! SMOKE_ERR="$("${WORKSPACE_DIR}/bin/llama-server-nemotron.part" --version 2>&1)"; then
            rm -f "${WORKSPACE_DIR}/bin/llama-server-nemotron.part"
            if [[ "$SMOKE_ERR" == *"libgomp.so.1"* ]]; then
                echo "[!] Missing OpenMP runtime (libgomp.so.1). Install 'sudo apt-get install -y libgomp1' (Debian/Ubuntu) or 'sudo dnf install -y libgomp' (Fedora/RHEL) to run prebuilt binaries without compiling."
            fi
            echo "[!] Prebuilt binary failed '--version' host compatibility smoke-test; falling back to local source build..."
            check_required_commands git cmake g++
        else
            mv -f "${WORKSPACE_DIR}/bin/llama-server-nemotron.part" "${WORKSPACE_DIR}/bin/llama-server-nemotron"
            ln -sf "llama-server-nemotron" "${WORKSPACE_DIR}/bin/${X86_PREBUILT_ASSET}"
            echo "[+] Installed & verified prebuilt binary: ${WORKSPACE_DIR}/bin/llama-server-nemotron (${X86_PREBUILT_ASSET}, SHA-256 + --version OK)"
            PREBUILT_OK=1
        fi
    else
        rm -f "${WORKSPACE_DIR}/bin/llama-server-nemotron.part"
        echo "[!] Prebuilt download failed; checking build toolchain for local source fallback..."
        check_required_commands git cmake g++
    fi
elif [[ "$USE_PREBUILT" == "1" ]]; then
    echo "[i] No precompiled Linux x86_64 AVX2/AVX-512 binary matches $(uname -s)/$(uname -m); compiling natively from source..."
    check_required_commands git cmake g++
fi

if [[ "$PREBUILT_OK" == "0" ]]; then
    TARGET_MODEL="$BUILD_TARGET_MODEL" bash "${REPO_ROOT}/scripts/build_zen5_llama.sh" "$LLAMA_BUILD_DIR"

    cp -f "${LLAMA_BUILD_DIR}/build/bin/llama-server" "${WORKSPACE_DIR}/bin/llama-server-nemotron"
    cp -f "${LLAMA_BUILD_DIR}/build/bin/llama-quantize" "${WORKSPACE_DIR}/bin/llama-quantize" 2>/dev/null || true
    cp -f "${LLAMA_BUILD_DIR}/build/bin/llama-perplexity" "${WORKSPACE_DIR}/bin/llama-perplexity" 2>/dev/null || true
    echo "[+] Installed binary: ${WORKSPACE_DIR}/bin/llama-server-nemotron (TARGET_MODEL=${BUILD_TARGET_MODEL})"
fi

if [[ "$PREBUILT_OK" == "0" && "$BUILD_TARGET_MODEL" == "nemotron" && -d "${LLAMA_BUILD_DIR}/include" ]]; then
    echo "[*] Compiling RouteCrack vocabulary tokenizer & table builder..."
    g++ -O3 -std=c++17 "${REPO_ROOT}/speculative/cpp/rc_tokenize_nemotron.cpp" \
        -o "${WORKSPACE_DIR}/bin/rc_tokenize_nemotron" \
        -I"${LLAMA_BUILD_DIR}/include" -I"${LLAMA_BUILD_DIR}/ggml/include" \
        -L"${LLAMA_BUILD_DIR}/build/bin" -lllama -lggml -lggml-base \
        -Wl,-rpath,"${LLAMA_BUILD_DIR}/build/bin"

    g++ -O3 -std=c++17 "${REPO_ROOT}/speculative/cpp/rc_build_tables.cpp" \
        -o "${WORKSPACE_DIR}/bin/rc_build_tables"
    echo "[+] Compiled RouteCrack table builders in ${WORKSPACE_DIR}/bin/"
fi

echo ""
echo "=============================================================================="
echo " STEP 3/5: RESOLVING / DOWNLOADING MODEL (${MODEL_LABEL})"
echo "=============================================================================="
if [[ -z "$MODEL_PATH" ]]; then
    if [[ -n "$MODEL_SUBDIR" ]]; then
        MODEL_DEST_DIR="${WORKSPACE_DIR}/models/${MODEL_SUBDIR}"
    else
        MODEL_DEST_DIR="${WORKSPACE_DIR}/models"
    fi
    if [[ -f "${MODEL_DEST_DIR}/${DEFAULT_MODEL_NAME}" ]]; then
        MODEL_PATH="${MODEL_DEST_DIR}/${DEFAULT_MODEL_NAME}"
    elif [[ -f "${WORKSPACE_DIR}/models/${DEFAULT_MODEL_NAME}" ]]; then
        MODEL_DEST_DIR="${WORKSPACE_DIR}/models"
        MODEL_PATH="${WORKSPACE_DIR}/models/${DEFAULT_MODEL_NAME}"
    else
        MODEL_PATH="${MODEL_DEST_DIR}/${DEFAULT_MODEL_NAME}"
    fi
fi

if [[ "$MODEL_EXPLICIT" == "1" ]]; then
    if [[ -f "$MODEL_PATH" ]]; then
        echo "[+] Using explicit local GGUF model: $MODEL_PATH"
    else
        echo "[!] ERROR: Explicit --model file not found: $MODEL_PATH" >&2
        exit 1
    fi
elif [[ "$SKIP_MODEL" == "1" ]]; then
    echo "[i] --skip-model specified. Skipping model weight download."
    echo "    Expected default GGUF path: $MODEL_PATH"
else
    mkdir -p "$MODEL_DEST_DIR"
    NUM_SHARDS="${#MODEL_SHARDS[@]}"
    for idx in "${!MODEL_SHARDS[@]}"; do
        SHARD_NAME="${MODEL_SHARDS[$idx]}"
        SHARD_DEST="${MODEL_DEST_DIR}/${SHARD_NAME}"
        SHARD_NUM=$((idx + 1))
        if [[ -s "$SHARD_DEST" ]]; then
            if [[ ${#MODEL_SHARD_SIZES[@]} -gt "$idx" ]]; then
                EXISTING_BYTES="$(wc -c < "$SHARD_DEST" | tr -d '[:space:]')"
                if [[ "$EXISTING_BYTES" != "${MODEL_SHARD_SIZES[$idx]}" ]]; then
                    echo "[!] Existing file ${SHARD_DEST} has unexpected byte size (${EXISTING_BYTES} != ${MODEL_SHARD_SIZES[$idx]}); re-downloading..."
                    rm -f "$SHARD_DEST"
                else
                    echo "[+] Found verified existing GGUF file (${SHARD_NUM}/${NUM_SHARDS}): ${SHARD_DEST}"
                    continue
                fi
            else
                echo "[+] Found existing GGUF file (${SHARD_NUM}/${NUM_SHARDS}): ${SHARD_DEST}"
                continue
            fi
        fi
        SHARD_URL="https://huggingface.co/${MODEL_REPO}/resolve/${MODEL_REV}/${SHARD_NAME}"
        echo "[*] Downloading (${SHARD_NUM}/${NUM_SHARDS}) ${SHARD_NAME} from ${MODEL_REPO}@${MODEL_REV:0:12}..."
        echo "    URL:  ${SHARD_URL}"
        echo "    Dest: ${SHARD_DEST}"
        curl -fL -C - --progress-bar -o "${SHARD_DEST}.part" "$SHARD_URL"

        # Verify GGUF magic header ("GGUF")
        MAGIC="$(head -c 4 "${SHARD_DEST}.part" 2>/dev/null || true)"
        if [[ "$MAGIC" != "GGUF" ]]; then
            echo "[!] ERROR: Downloaded file ${SHARD_DEST}.part does not have a valid GGUF header (got '${MAGIC}')." >&2
            rm -f "${SHARD_DEST}.part"
            exit 1
        fi

        # Verify exact byte size when pinned
        if [[ ${#MODEL_SHARD_SIZES[@]} -gt "$idx" ]]; then
            EXPECTED_BYTES="${MODEL_SHARD_SIZES[$idx]}"
            ACTUAL_BYTES="$(wc -c < "${SHARD_DEST}.part" | tr -d '[:space:]')"
            if [[ "$ACTUAL_BYTES" != "$EXPECTED_BYTES" ]]; then
                echo "[!] ERROR: Size mismatch for ${SHARD_NAME} (expected ${EXPECTED_BYTES} bytes, got ${ACTUAL_BYTES})!" >&2
                rm -f "${SHARD_DEST}.part"
                exit 1
            fi
        fi

        # Verify SHA-256 checksum for every shard when pinned
        if [[ ${#MODEL_SHARD_SHA256S[@]} -gt "$idx" && -n "${MODEL_SHARD_SHA256S[$idx]}" ]]; then
            EXPECTED_SHA256="${MODEL_SHARD_SHA256S[$idx]}"
            echo "[*] Verifying SHA-256 checksum for ${SHARD_NAME}..."
            ACTUAL_SHA256="$(sha256sum "${SHARD_DEST}.part" | awk '{print $1}')"
            if [[ "$ACTUAL_SHA256" != "$EXPECTED_SHA256" ]]; then
                echo "[!] ERROR: SHA-256 mismatch for ${SHARD_DEST}.part (removing corrupt partial file)!" >&2
                echo "    Expected: $EXPECTED_SHA256" >&2
                echo "    Got:      $ACTUAL_SHA256" >&2
                rm -f "${SHARD_DEST}.part"
                exit 1
            fi
        fi

        mv "${SHARD_DEST}.part" "$SHARD_DEST"
        echo "[+] Verified (${SHARD_NUM}/${NUM_SHARDS}): ${SHARD_DEST}"
    done
    MODEL_PATH="${MODEL_DEST_DIR}/${DEFAULT_MODEL_NAME}"
fi

echo ""
echo "=============================================================================="
echo " STEP 4/5: SPECULATIVE DECODING ENGINE CONFIGURATION"
echo "=============================================================================="
if [[ "$MODEL_FAMILY" == "nemotron" && -f "$MODEL_PATH" && -x "${WORKSPACE_DIR}/bin/rc_tokenize_nemotron" && -x "${WORKSPACE_DIR}/bin/rc_build_tables" ]]; then
    CORPUS_TXT="${WORKSPACE_DIR}/routecrack/bootstrap_corpus.txt"
    CORPUS_BIN="${WORKSPACE_DIR}/routecrack/corpus_nemotron_train.bin"
    echo "[*] Assembling coding & agentic <tool_call> training corpus..."
    {
        for _ in 1 2 3 4 5; do
            find "$REPO_ROOT" -maxdepth 3 -type f \( -name "*.cpp" -o -name "*.h" -o -name "*.py" -o -name "*.md" -o -name "*.sh" -o -name "*.jinja" \) \
                -not -path "*/.git/*" -not -path "${REPO_ROOT}/workspace/*" -not -path "${WORKSPACE_DIR}/*" -exec cat {} + 2>/dev/null || true
        done
        if [[ -n "$USER_CORPUS_DIR" && -d "$USER_CORPUS_DIR" ]]; then
            echo "[*] Including custom codebase from $USER_CORPUS_DIR..." >&2
            find "$USER_CORPUS_DIR" -maxdepth 4 -type f \( -name "*.py" -o -name "*.ts" -o -name "*.tsx" -o -name "*.js" -o -name "*.rs" -o -name "*.go" -o -name "*.cpp" -o -name "*.c" -o -name "*.h" -o -name "*.md" \) \
                -not -path "*/.git/*" -not -path "*/node_modules/*" -exec head -c 100000 {} + 2>/dev/null || true
        fi
    } > "$CORPUS_TXT"

    echo "[*] Tokenizing corpus with native Nemotron 131K vocabulary..."
    "${WORKSPACE_DIR}/bin/rc_tokenize_nemotron" "$MODEL_PATH" "$CORPUS_TXT" "$CORPUS_BIN"
    cp -f "$CORPUS_BIN" "${WORKSPACE_DIR}/routecrack/corpus_nemotron_warm.bin"

    echo "[*] Building 8-level FNV-64 Hash Cascade, 300 Hot Rules & Macro Codebook..."
    "${WORKSPACE_DIR}/bin/rc_build_tables" "$CORPUS_BIN" "${WORKSPACE_DIR}/routecrack"
    echo "[+] RouteCrack tables ready in ${WORKSPACE_DIR}/routecrack/"
else
    echo "[i] Online Stockfish-PVS v2 (192 KB L2-resident Zobrist TT) active — zero offline vocabulary build required."
fi

echo ""
echo "=============================================================================="
echo " STEP 5/5: GENERATING AUTO-TUNED LAUNCH SCRIPTS"
echo "=============================================================================="
if [[ "$TOTAL_RAM_GB" -ge 30 ]]; then
    CTX_SIZE=262144
else
    CTX_SIZE=65536
fi

mkdir -p "${WORKSPACE_DIR}/run"
printf '%s\n' "${SERVER_PORT}" > "${WORKSPACE_DIR}/run/server_port"
if [[ -n "$MODEL_PATH" ]]; then
    case "$MODEL_PATH" in
        /*) MODEL_ABS="$MODEL_PATH" ;;
        *)  MODEL_ABS="${PWD}/${MODEL_PATH}" ;;
    esac
    if [[ -f "$MODEL_ABS" || "$MODEL_EXPLICIT" == "1" ]]; then
        printf '%s\n' "${MODEL_ABS}" > "${WORKSPACE_DIR}/run/default_model"
    fi
fi

if [[ -f "${REPO_ROOT}/start_server.sh" && ! -f "${REPO_ROOT}/start_server.sh.bak" ]]; then
    cp -f "${REPO_ROOT}/start_server.sh" "${REPO_ROOT}/start_server.sh.bak"
    echo "[i] Backed up original start_server.sh -> start_server.sh.bak"
fi

cat > "${REPO_ROOT}/start_server.sh" <<EOF
#!/usr/bin/env bash
# Auto-generated by install.sh — Starts CPU LLM Server with llama-modchip Auto-Profiler & Stockfish-PVS
# Usage: ./start_server.sh [/path/to/model.gguf] [extra llama-server flags, e.g. -n 4096]
set -euo pipefail

export RC_TABLES_DIR="${WORKSPACE_DIR}/routecrack"
export LD_LIBRARY_PATH="${LLAMA_BUILD_DIR}/build/bin:\${LD_LIBRARY_PATH:-}"

MODEL="${MODEL_PATH}"
PASS_ARGS=()
if [[ \$# -gt 0 && "\$1" != -* ]]; then
    MODEL="\$1"
    shift
fi
while [[ \$# -gt 0 ]]; do
    case "\$1" in
        -m|--model)
            if [[ \$# -ge 2 ]]; then
                MODEL="\$2"
                shift 2
                continue
            fi
            ;;
        --model=*)
            MODEL="\${1#--model=}"
            shift
            continue
            ;;
    esac
    PASS_ARGS+=("\$1")
    shift
done
if [[ ! -f "\$MODEL" ]]; then
    echo "[!] Model file not found at: \$MODEL"
    echo "    Run ./install.sh (without --skip-model) or pass /path/to/model.gguf [extra flags]"
    exit 1
fi

# Run llama-modchip to inspect the GGUF header,
# engage architecture safety locks (e.g. Interleaved 3:1 SWA on Pure GQA), and write /dev/shm/llama_modchip_<uid>.bin
python3 "${REPO_ROOT}/scripts/llama_modchip.py" --model "\$MODEL" --apply-shm

PROFILE_ENV="\$(python3 - "\$MODEL" <<'PYEOF'
import sys
sys.path.insert(0, "${REPO_ROOT}/scripts")
import llama_modchip as mc
hw = mc.probe_hardware()
info = mc.inspect_gguf(sys.argv[1])
prof = mc.compute_modchip_profile(hw, info)
print(f"export MODEL_ARCH=\"{info.arch}\"")
print(f"export LLAMA_KV_L3_WINDOW=\${{LLAMA_KV_L3_WINDOW:-{prof.l3_window}}}")
print(f"export LLAMA_ATTN_WINDOW=\${{LLAMA_ATTN_WINDOW:-{prof.attn_window}}}")
print(f"export LLAMA_ATTN_SINKS=\${{LLAMA_ATTN_SINKS:-{prof.attn_sinks}}}")
print(f"export LLAMA_SWA_STRIDE=\${{LLAMA_SWA_STRIDE:-{prof.swa_stride}}}")
print(f"export LLAMA_KV_DISABLE_CHUNKING=\${{LLAMA_KV_DISABLE_CHUNKING:-{prof.disable_chunking}}}")
print(f"export LLAMA_RC_ENABLED=\${{LLAMA_RC_ENABLED:-{prof.rc_enabled}}}")
print(f"export LLAMA_RC_CERT_D0=\${{LLAMA_RC_CERT_D0:-{prof.rc_cert_d0}}}")
print(f"export LLAMA_RC_CERT_D1=\${{LLAMA_RC_CERT_D1:-{prof.rc_cert_d1}}}")
print(f"export LLAMA_RC_CERT_D2=\${{LLAMA_RC_CERT_D2:-{prof.rc_cert_d2}}}")
if prof.moe_top_k > 0 and info.arch != "deepseek4":
    print(f"export LLAMA_MOE_TOP_K=\${{LLAMA_MOE_TOP_K:-{prof.moe_top_k}}}")
print(f"export OMP_PROC_BIND=\${{OMP_PROC_BIND:-{hw.omp_proc_bind}}}")
print(f"export OMP_PLACES=\${{OMP_PLACES:-cores}}")
print(f'export TASKSET_MASK="\${{TASKSET_MASK:-{hw.taskset_mask}}}"')
print(f"export DECODE_THREADS=\${{DECODE_THREADS:-{hw.decode_threads}}}")
print(f"export BATCH_THREADS=\${{BATCH_THREADS:-{hw.batch_threads}}}")
PYEOF
)" || exit 1
eval "\$PROFILE_ENV"

CACHE_V_TYPE="q4_0"
EFFECTIVE_CTX_SIZE="\${MODCHIP_CTX_SIZE:-${CTX_SIZE}}"
UBATCH_SIZE="\${MODCHIP_UBATCH_SIZE:-2048}"
SLOT_SAVE_DIR="\${MODCHIP_SLOT_SAVE_PATH:-/dev/shm}"

# Isolate DSV4_MOE_CFG_PATH inside a 0700 per-user directory and reset to unpruned Top-6 on startup
MODCHIP_PRIV_RUN="\${XDG_RUNTIME_DIR:-${WORKSPACE_DIR}/run}/modchip-\$(id -u)"
mkdir -p "\$MODCHIP_PRIV_RUN"
chmod 700 "\$MODCHIP_PRIV_RUN" 2>/dev/null || true
export DSV4_MOE_CFG_PATH="\${DSV4_MOE_CFG_PATH:-\${MODCHIP_PRIV_RUN}/dsv4_moe_cfg.bin}"
if [[ "\${MODCHIP_KEEP_DSV4_CFG:-0}" != "1" ]]; then
    python3 - "\$DSV4_MOE_CFG_PATH" <<'CFGEOF'
import os, struct, sys
p = sys.argv[1]
fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
try:
    os.write(fd, struct.pack("<8i", 6, 6, 0, 3, 43, 1, 0, 0))
finally:
    os.close(fd)
CFGEOF
fi

if [[ "\${MODEL_ARCH:-}" == "deepseek4" ]]; then
    CACHE_V_TYPE="q8_0"
    if [[ "${TOTAL_RAM_GB}" -lt 80 ]]; then
        AVAIL_KIB="\$(awk '/MemAvailable:/ {print \$2}' /proc/meminfo 2>/dev/null || echo 62914560)"
        if [[ "\${AVAIL_KIB:-62914560}" -lt 67108864 || "${TOTAL_RAM_GB}" -le 68 ]]; then
            [[ -z "\${MODCHIP_CTX_SIZE:-}" ]] && EFFECTIVE_CTX_SIZE=32768
            [[ -z "\${MODCHIP_UBATCH_SIZE:-}" ]] && UBATCH_SIZE=512
            [[ -z "\${MODCHIP_SLOT_SAVE_PATH:-}" ]] && SLOT_SAVE_DIR="${WORKSPACE_DIR}/cache"
            echo "[i] 64GB OS Headroom Guard: using -c \${EFFECTIVE_CTX_SIZE} -ub \${UBATCH_SIZE} and disk slot cache (~58.5–59.1 GiB anonymous VmRSS)"
            echo "    so desktop/OS processes retain >=3.4–4.0 GiB free RAM. (Set MODCHIP_CTX_SIZE=262144 on a dedicated headless node.)"
        fi
    fi
elif [[ "${TOTAL_RAM_GB}" -le 18 ]]; then
    [[ -z "\${MODCHIP_CTX_SIZE:-}" ]] && EFFECTIVE_CTX_SIZE=32768
    [[ -z "\${MODCHIP_UBATCH_SIZE:-}" ]] && UBATCH_SIZE=512
    [[ -z "\${MODCHIP_SLOT_SAVE_PATH:-}" ]] && SLOT_SAVE_DIR="${WORKSPACE_DIR}/cache"
    echo "[i] 16GB RAM Swap Guard: defaulting to -c \${EFFECTIVE_CTX_SIZE} -ub \${UBATCH_SIZE} (~14.6 GiB VmRSS) to prevent swap thrashing."
fi

SERVER_HOST="\${SERVER_HOST:-${SERVER_HOST}}"
SERVER_PORT="\${SERVER_PORT:-${SERVER_PORT}}"
# Optional space-separated extra llama-server flags (no nested quotes), e.g. EXTRA_SERVER_ARGS="-ngl 99 --n-cpu-moe 44"
EXTRA_ARGS_ARR=()
if [[ -n "\${EXTRA_SERVER_ARGS:-}" ]]; then
    read -r -a EXTRA_ARGS_ARR <<< "\${EXTRA_SERVER_ARGS}"
fi
echo "[*] Starting llama-server on http://\${SERVER_HOST}:\${SERVER_PORT} (Decode: \${DECODE_THREADS}T / Prefill: \${BATCH_THREADS}T [mask: \${TASKSET_MASK}], Context: \${EFFECTIVE_CTX_SIZE}, ubatch: \${UBATCH_SIZE})..."
exec taskset -c "\${TASKSET_MASK}" "${WORKSPACE_DIR}/bin/llama-server-nemotron" \\
  --model "\$MODEL" \\
  --port "\${SERVER_PORT}" \\
  --host "\${SERVER_HOST}" \\
  --no-webui \\
  --offline \\
  -c "\${EFFECTIVE_CTX_SIZE}" \\
  -np 1 \\
  -b 2048 \\
  -ub "\${UBATCH_SIZE}" \\
  -t "\${DECODE_THREADS}" \\
  -tb "\${BATCH_THREADS}" \\
  --cache-type-k q8_0 \\
  --cache-type-v "\${CACHE_V_TYPE}" \\
  --flash-attn auto \\
  --load-mode none \\
  --repack \\
  --cache-reuse 256 \\
  --slot-save-path "\${SLOT_SAVE_DIR}" \\
  --ctx-checkpoints 64 \\
  --checkpoint-min-step 128 \\
  --spec-type ngram-simple \\
  --spec-ngram-simple-size-n 4 \\
  --spec-ngram-simple-size-m 7 \\
  --spec-ngram-simple-min-hits 1 \\
  --reasoning-format "\${REASONING_FORMAT:-deepseek}" \\
  --reasoning-budget "\${REASONING_BUDGET:-0}" \\
  \${EXTRA_ARGS_ARR[@]+"\${EXTRA_ARGS_ARR[@]}"} \\
  \${PASS_ARGS[@]+"\${PASS_ARGS[@]}"}
EOF
chmod +x "${REPO_ROOT}/start_server.sh"

cat > "${REPO_ROOT}/start_gateway.sh" <<EOF
#!/usr/bin/env bash
# Auto-generated by install.sh — Starts the Agentic SSE Gateway + Live Web UI
set -euo pipefail

export SERVER_HOST="\${SERVER_HOST:-${SERVER_HOST}}"
export LLAMA_BACKEND_URL="\${LLAMA_BACKEND_URL:-http://\${SERVER_HOST}:${SERVER_PORT}}"
export GATEWAY_HOST="\${GATEWAY_HOST:-${GATEWAY_HOST}}"
export GATEWAY_PORT="\${GATEWAY_PORT:-${GATEWAY_PORT}}"
echo "[*] Starting Agentic SSE Gateway & Live Web UI on http://\${GATEWAY_HOST}:\${GATEWAY_PORT} -> Backend \${LLAMA_BACKEND_URL}..."
echo "[i] 🌐 Open Web Chat & Telemetry HUD : http://\${GATEWAY_HOST}:\${GATEWAY_PORT}"
echo "[i] 🤖 Connect Claude Code CLI      : ANTHROPIC_BASE_URL=http://\${GATEWAY_HOST}:\${GATEWAY_PORT} claude"
exec python3 "${REPO_ROOT}/gateway/sse_gateway.py" "\${GATEWAY_PORT}"
EOF
chmod +x "${REPO_ROOT}/start_gateway.sh"

echo "=============================================================================="
echo " [+] INSTALLATION COMPLETE!"
echo "=============================================================================="
echo " Quickest way to start & chat (Unified ./modchip CLI):"
echo "      ./modchip start        # Starts server + gateway in background"
echo "      ./modchip chat         # Interactive terminal chat with live tok/s"
echo "      ./modchip bench        # Run 30s benchmark & print shareable scorecard"
echo "      ./modchip claude       # Launch Claude Code connected to local CPU engine"
echo ""
echo " Or open the Live Web Chat & Telemetry HUD in your browser:"
echo "      http://${GATEWAY_HOST}:${GATEWAY_PORT}"
echo "=============================================================================="
