#!/usr/bin/env bash
# ==============================================================================
# install.sh — Interactive Zero-to-Running Installer for llama-modchip (v2.5.0)
# ==============================================================================
# Usage:
#   One-liner (from scratch):
#     curl -sL https://raw.githubusercontent.com/synchopate/llama-modchip/main/install.sh | bash
#
#   From cloned repo:
#     ./install.sh [options]
#
# Options:
#   --workspace <dir>   Target build/model directory (default: ./workspace)
#   --preset <id>       Model preset: dsv4-ablit | dsv4-vq3 | dsv4-fractal |
#                                     nemotron-vq3 | nemotron-fractal | nemotron-q4
#   --model <path>      Use an existing local GGUF file instead of downloading
#   --skip-model        Install engine & scripts without downloading GGUF weights
#   --prebuilt          Download CPU-matched Linux x86_64 binary (-avx512 / -avx2)
#   --source            Compile patched engine locally from source (requires cmake/g++)
#   --yes, -y           Non-interactive mode: accept hardware-matched defaults
#   --corpus <dir>      Optional path to your codebase for Nemotron RouteCrack n-grams
#   --port <port>       llama-server port (default: 8789)
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
        --corpus)
            USER_CORPUS_DIR="$(realpath -m "$2")"
            shift 2
            ;;
        --port)
            SERVER_PORT="$2"
            shift 2
            ;;
        -h|--help)
            sed -n '2,23p' "$0"
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
import os
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
print(len(cores) if cores else (os.cpu_count() or 8))
')"

TOTAL_RAM_GB="$(python3 -c '
import glob
ram_gb = 32
try:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemTotal:"):
                ram_gb = int(line.split()[1]) // (1024 * 1024)
                break
except Exception:
    pass
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

IS_INTERACTIVE=0
if [[ "$NONINTERACTIVE" != "1" && -t 1 && -r /dev/tty ]]; then
    IS_INTERACTIVE=1
fi

# Hardware-aware default preset:
#   >= 60 GB RAM -> DeepSeek-V4-Flash-0731-Abliterated-Fractal (284B default DeepSeek)
#   <  60 GB RAM -> Nemotron-3.5-30B-A3B-VQ3-64 (on AVX-512 VBMI or <=18 GB RAM) else Fractal-64
if [[ "$TOTAL_RAM_GB" -ge 60 ]]; then
    DEFAULT_PRESET="dsv4-ablit"
    DEFAULT_MENU_CHOICE="1"
elif [[ "$HAS_AVX512_VBMI" == "1" || "$TOTAL_RAM_GB" -le 18 ]]; then
    DEFAULT_PRESET="nemotron-vq3"
    DEFAULT_MENU_CHOICE="4"
else
    DEFAULT_PRESET="nemotron-fractal"
    DEFAULT_MENU_CHOICE="5"
fi

if [[ "$MODEL_EXPLICIT" == "0" && -z "$MODEL_PRESET" ]]; then
    if [[ "$SKIP_MODEL" == "0" && "$IS_INTERACTIVE" == "1" ]]; then
        echo ""
        echo "------------------------------------------------------------------------------"
        echo " MODEL SELECTION (Detected Host RAM: ~${TOTAL_RAM_GB} GiB | Physical Cores: ${PHYS_CORES})"
        echo "------------------------------------------------------------------------------"
        echo "  1) DeepSeek-V4-Flash-0731-Abliterated-Fractal [284B MoE · 56.93 GiB · 3 shards]"
        echo "     Default DeepSeek (64 GB+ RAM) — Single-pass Q6_K + Fractal-64, uncensored/abliterated"
        echo "  2) DeepSeek-V4-Flash-0731-VQ3                 [284B MoE · 56.40 GiB · 3 shards]"
        echo "     Standard safety-aligned 284B — Hybrid LUT-VQ3-64 + Fractal-64 (64 GB+ RAM)"
        echo "  3) DeepSeek-V4-Flash-0731-Fractal             [284B MoE · 56.57 GiB · 3 shards]"
        echo "     Standard safety-aligned 284B — Pure Fractal-64 (AVX2 / Apple Silicon 64 GB+)"
        echo "  4) NVIDIA-Nemotron-3.5-30B-A3B-VQ3-64         [30B MoE  · 13.30 GiB · 1 file]"
        echo "     Default 30B for 16–32 GB RAM & AVX-512 VBMI CPUs (100/100 Quality Gate)"
        echo "  5) NVIDIA-Nemotron-3.5-30B-A3B-Fractal-64     [30B MoE  · 14.36 GiB · 1 file]"
        echo "     Recommended 30B for AVX2-only x86_64 & ARM64 CPUs (100/100 Quality Gate)"
        echo "  6) NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0 [30B MoE  · 17.59 GiB · 1 file]"
        echo "     Official 4.50-bpw Q4_0 baseline checkpoint (24 GB+ RAM)"
        echo "------------------------------------------------------------------------------"
        printf " Select model [1-6, Enter=%s]: " "$DEFAULT_MENU_CHOICE"
        read -r USER_MODEL_CHOICE < /dev/tty || USER_MODEL_CHOICE=""
        USER_MODEL_CHOICE="${USER_MODEL_CHOICE:-$DEFAULT_MENU_CHOICE}"
        case "$USER_MODEL_CHOICE" in
            1|dsv4-ablit|dsv4-abliterated|deepseek) MODEL_PRESET="dsv4-ablit" ;;
            2|dsv4-vq3)                             MODEL_PRESET="dsv4-vq3" ;;
            3|dsv4-fractal)                         MODEL_PRESET="dsv4-fractal" ;;
            4|nemotron-vq3)                         MODEL_PRESET="nemotron-vq3" ;;
            5|nemotron-fractal)                     MODEL_PRESET="nemotron-fractal" ;;
            6|nemotron-q4)                          MODEL_PRESET="nemotron-q4" ;;
            *)
                echo "[!] Unrecognized choice '${USER_MODEL_CHOICE}'; using hardware default (${DEFAULT_PRESET})."
                MODEL_PRESET="$DEFAULT_PRESET"
                ;;
        esac
    else
        MODEL_PRESET="$DEFAULT_PRESET"
    fi
fi

MODEL_FAMILY="nemotron"
IS_ABLITERATED=0
MODEL_SHA256=""
MODEL_SHARDS=()
MODEL_SHARD_SIZES=()
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
        1|dsv4-ablit|dsv4-abliterated|deepseek)
            MODEL_PRESET="dsv4-ablit"
            MODEL_FAMILY="dsv4"
            IS_ABLITERATED=1
            MODEL_LABEL="DeepSeek-V4-Flash-0731-Abliterated-Fractal (284B · 56.93 GiB · 3 shards)"
            MODEL_REPO="Synchopate/DeepSeek-V4-Flash-0731-Abliterated-Fractal-64GB-CPU-GGUF"
            MODEL_REV="main"
            MODEL_SUBDIR="dsv4-abliterated-fractal"
            MODEL_SHARDS=(
                "DeepSeek-V4-Flash-0731-Abliterated-Fractal-00001-of-00003.gguf"
                "DeepSeek-V4-Flash-0731-Abliterated-Fractal-00002-of-00003.gguf"
                "DeepSeek-V4-Flash-0731-Abliterated-Fractal-00003-of-00003.gguf"
            )
            MODEL_SHARD_SIZES=(5257792 36817932704 24306173504)
            ;;
        2|dsv4-vq3)
            MODEL_PRESET="dsv4-vq3"
            MODEL_FAMILY="dsv4"
            MODEL_LABEL="DeepSeek-V4-Flash-0731-VQ3 (284B · 56.40 GiB · 3 shards)"
            MODEL_REPO="Synchopate/DeepSeek-V4-Flash-0731-VQ3-64GB-CPU-GGUF"
            MODEL_REV="main"
            MODEL_SUBDIR="dsv4-vq3"
            MODEL_SHARDS=(
                "DeepSeek-V4-Flash-0731-VQ3-00001-of-00003.gguf"
                "DeepSeek-V4-Flash-0731-VQ3-00002-of-00003.gguf"
                "DeepSeek-V4-Flash-0731-VQ3-00003-of-00003.gguf"
            )
            ;;
        3|dsv4-fractal)
            MODEL_PRESET="dsv4-fractal"
            MODEL_FAMILY="dsv4"
            MODEL_LABEL="DeepSeek-V4-Flash-0731-Fractal (284B · 56.57 GiB · 3 shards)"
            MODEL_REPO="Synchopate/DeepSeek-V4-Flash-0731-Fractal-64GB-CPU-GGUF"
            MODEL_REV="main"
            MODEL_SUBDIR="dsv4-fractal"
            MODEL_SHARDS=(
                "DeepSeek-V4-Flash-0731-Fractal-00001-of-00003.gguf"
                "DeepSeek-V4-Flash-0731-Fractal-00002-of-00003.gguf"
                "DeepSeek-V4-Flash-0731-Fractal-00003-of-00003.gguf"
            )
            ;;
        4|nemotron-vq3)
            MODEL_PRESET="nemotron-vq3"
            MODEL_FAMILY="nemotron"
            MODEL_LABEL="NVIDIA-Nemotron-3.5-30B-A3B-VQ3-64 (30B · 13.30 GiB)"
            MODEL_REPO="Synchopate/NVIDIA-Nemotron-3.5-30B-A3B-VQ3-64-CPU-GGUF"
            MODEL_REV="main"
            MODEL_SHARDS=("NVIDIA-Nemotron-3.5-30B-A3B-VQ3-64.gguf")
            ;;
        5|nemotron-fractal)
            MODEL_PRESET="nemotron-fractal"
            MODEL_FAMILY="nemotron"
            MODEL_LABEL="NVIDIA-Nemotron-3.5-30B-A3B-Fractal-64 (30B · 14.36 GiB)"
            MODEL_REPO="Synchopate/NVIDIA-Nemotron-3.5-30B-A3B-Fractal-64-CPU-GGUF"
            MODEL_REV="main"
            MODEL_SHARDS=("NVIDIA-Nemotron-3.5-30B-A3B-Fractal-64.gguf")
            ;;
        6|nemotron-q4)
            MODEL_PRESET="nemotron-q4"
            MODEL_FAMILY="nemotron"
            MODEL_LABEL="NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0 (30B · 17.59 GiB)"
            MODEL_REPO="ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF"
            MODEL_REV="8cd082170ed313484be1cf44f256a5caa2cfdad3"
            MODEL_SHA256="61f87e75974e4b535dcdf9aad056541a9514f1dfa4538b463b081d19b7a00e3c"
            MODEL_SHARDS=("NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0.gguf")
            ;;
        *)
            echo "[!] ERROR: Unknown --preset '${MODEL_PRESET}'. Valid presets: dsv4-ablit, dsv4-vq3, dsv4-fractal, nemotron-vq3, nemotron-fractal, nemotron-q4" >&2
            exit 1
            ;;
    esac
    DEFAULT_MODEL_NAME="${MODEL_SHARDS[0]}"
fi

if [[ "$MODEL_FAMILY" == "dsv4" && "$TOTAL_RAM_GB" -lt 60 ]]; then
    echo "[!] WARNING: DeepSeek-V4 284B requires ~58.3–59.1 GiB VmRSS (64 GB+ host RAM recommended)."
fi

if [[ "$IS_ABLITERATED" == "1" ]]; then
    echo ""
    echo "=============================================================================="
    echo " [!] DISCLAIMER — ABLITERATED / UNCENSORED MODEL CHECKPOINT"
    echo "=============================================================================="
    echo " DeepSeek-V4-Flash-0731-Abliterated-Fractal derives from an abliterated base"
    echo " checkpoint in which refusal-mediating activation directions were removed."
    echo " Standard upstream refusal filters are disabled."
    echo "   • Scope: Authorized security research, red-teaming, CTF, code auditing,"
    echo "     and local offline development only."
    echo "   • Deployment: Do NOT expose this model to unmoderated public networks"
    echo "     without external access control and safety filtering."
    echo "   • Responsibility: You assume full legal and operational responsibility for"
    echo "     all generated outputs. For standard safety alignment, re-run with"
    echo "     './install.sh --preset dsv4-vq3' (Option 2)."
    echo "=============================================================================="
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
            echo "     Applies deepseek-v4-flash-fractal-256k.patch + lut-vq3-64-avx512-vbmi.patch."
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
    ASSET_URL="https://github.com/synchopate/llama-modchip/releases/download/v2.5.0/${X86_PREBUILT_ASSET}"
    echo "[*] CPUID selected precompiled binary: ${X86_PREBUILT_ASSET}"
    if curl -fL --progress-bar -o "${WORKSPACE_DIR}/bin/llama-server-nemotron.part" "$ASSET_URL"; then
        mv -f "${WORKSPACE_DIR}/bin/llama-server-nemotron.part" "${WORKSPACE_DIR}/bin/llama-server-nemotron"
        chmod +x "${WORKSPACE_DIR}/bin/llama-server-nemotron"
        ln -sf "llama-server-nemotron" "${WORKSPACE_DIR}/bin/${X86_PREBUILT_ASSET}"
        echo "[+] Installed prebuilt binary: ${WORKSPACE_DIR}/bin/llama-server-nemotron (${X86_PREBUILT_ASSET})"
        PREBUILT_OK=1
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
    elif [[ "$MODEL_FAMILY" == "nemotron" && -f "/opt/models/nemotron/${DEFAULT_MODEL_NAME}" ]]; then
        MODEL_DEST_DIR="/opt/models/nemotron"
        MODEL_PATH="/opt/models/nemotron/${DEFAULT_MODEL_NAME}"
    elif [[ "$MODEL_FAMILY" == "dsv4" && -f "/opt/models/deepseek-v4/lcc/${DEFAULT_MODEL_NAME}" ]]; then
        MODEL_DEST_DIR="/opt/models/deepseek-v4/lcc"
        MODEL_PATH="/opt/models/deepseek-v4/lcc/${DEFAULT_MODEL_NAME}"
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
            echo "[+] Found existing GGUF file (${SHARD_NUM}/${NUM_SHARDS}): ${SHARD_DEST}"
            continue
        fi
        SHARD_URL="https://huggingface.co/${MODEL_REPO}/resolve/${MODEL_REV}/${SHARD_NAME}"
        echo "[*] Downloading (${SHARD_NUM}/${NUM_SHARDS}) ${SHARD_NAME} from ${MODEL_REPO}..."
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

        # Verify SHA-256 checksum when pinned
        if [[ -n "$MODEL_SHA256" && "$NUM_SHARDS" -eq 1 ]]; then
            echo "[*] Verifying SHA-256 checksum..."
            ACTUAL_SHA256="$(sha256sum "${SHARD_DEST}.part" | awk '{print $1}')"
            if [[ "$ACTUAL_SHA256" != "$MODEL_SHA256" ]]; then
                echo "[!] ERROR: SHA-256 mismatch for ${SHARD_DEST}.part (removing corrupt partial file)!" >&2
                echo "    Expected: $MODEL_SHA256" >&2
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

cat > "${REPO_ROOT}/start_server.sh" <<EOF
#!/usr/bin/env bash
# Auto-generated by install.sh — Starts CPU LLM Server with llama-modchip Auto-Profiler & Stockfish-PVS
set -euo pipefail

export RC_TABLES_DIR="${WORKSPACE_DIR}/routecrack"
export LD_LIBRARY_PATH="${LLAMA_BUILD_DIR}/build/bin:\${LD_LIBRARY_PATH:-}"

MODEL="\${1:-${MODEL_PATH}}"
if [[ ! -f "\$MODEL" ]]; then
    echo "[!] Model file not found at: \$MODEL"
    echo "    Run ./install.sh (without --skip-model) or pass /path/to/model.gguf"
    exit 1
fi

# Run llama-modchip to inspect the GGUF header,
# engage architecture safety locks (e.g. Interleaved 3:1 SWA on Pure GQA), and write /dev/shm/llama_modchip.bin
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
if [[ "\${MODEL_ARCH:-}" == "deepseek4" ]]; then
    CACHE_V_TYPE="q8_0"
    if [[ -z "\${MODCHIP_CTX_SIZE:-}" && "${TOTAL_RAM_GB}" -lt 80 ]]; then
        AVAIL_KIB="\$(awk '/MemAvailable:/ {print \$2}' /proc/meminfo 2>/dev/null || echo 62914560)"
        if [[ "\${AVAIL_KIB:-62914560}" -lt 63438848 || "${TOTAL_RAM_GB}" -le 68 ]]; then
            EFFECTIVE_CTX_SIZE=32768
            echo "[i] 64GB Desktop OS Headroom Guard: defaulting to -c 32768 (~58.3–59.1 GiB VmRSS) so GUI/OS retain >=3.4–4.2 GiB free RAM."
            echo "    (On a dedicated headless 64GB+ server, pass MODCHIP_CTX_SIZE=262144 ./start_server.sh for full 256K context.)"
        fi
    fi
fi

SERVER_HOST="\${SERVER_HOST:-${SERVER_HOST}}"
SERVER_PORT="\${SERVER_PORT:-${SERVER_PORT}}"
# Optional space-separated extra llama-server flags (no nested quotes), e.g. EXTRA_SERVER_ARGS="-ngl 99 --n-cpu-moe 44"
EXTRA_ARGS_ARR=()
if [[ -n "\${EXTRA_SERVER_ARGS:-}" ]]; then
    read -r -a EXTRA_ARGS_ARR <<< "\${EXTRA_SERVER_ARGS}"
fi
echo "[*] Starting llama-server on http://\${SERVER_HOST}:\${SERVER_PORT} (Decode: \${DECODE_THREADS}T / Prefill: \${BATCH_THREADS}T [mask: \${TASKSET_MASK}], Context: \${EFFECTIVE_CTX_SIZE})..."
exec taskset -c "\${TASKSET_MASK}" "${WORKSPACE_DIR}/bin/llama-server-nemotron" \\
  --model "\$MODEL" \\
  --port "\${SERVER_PORT}" \\
  --host "\${SERVER_HOST}" \\
  --no-webui \\
  --offline \\
  -c "\${EFFECTIVE_CTX_SIZE}" \\
  -np 1 \\
  -b 2048 \\
  -ub 2048 \\
  -t "\${DECODE_THREADS}" \\
  -tb "\${BATCH_THREADS}" \\
  --cache-type-k q8_0 \\
  --cache-type-v "\${CACHE_V_TYPE}" \\
  --flash-attn auto \\
  --load-mode none \\
  --repack \\
  --cache-reuse 256 \\
  --slot-save-path /dev/shm \\
  --ctx-checkpoints 64 \\
  --checkpoint-min-step 128 \\
  --spec-type ngram-simple \\
  --spec-ngram-simple-size-n 4 \\
  --spec-ngram-simple-size-m 7 \\
  --spec-ngram-simple-min-hits 1 \\
  --reasoning-format deepseek \\
  --reasoning-budget 0 \\
  \${EXTRA_ARGS_ARR[@]+"\${EXTRA_ARGS_ARR[@]}"}
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
