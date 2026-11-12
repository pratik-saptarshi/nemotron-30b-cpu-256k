#!/usr/bin/env bash
# ==============================================================================
# install.sh — Zero-to-Running Installer for Nemotron-3.5-30B-A3B (256K CPU)
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
#   --model <path>      Use an existing GGUF file instead of downloading
#   --skip-model        Build binaries & scripts without downloading the 17.6 GiB (18.9 GB) GGUF
#   --prebuilt          Download CPU-matched prebuilt release binary (-avx512 or -avx2) on Linux x86_64
#   --corpus <dir>      Optional path to your own codebase to train RouteCrack n-grams
#   --port <port>       llama-server port (default: 8789)
# ==============================================================================
set -euo pipefail

# If piped via `curl ... | bash`, clone the repository first and re-exec
SCRIPT_SOURCE="${BASH_SOURCE[0]:-}"
if [[ -z "$SCRIPT_SOURCE" ]] || [[ ! -f "$(dirname "$SCRIPT_SOURCE")/patches/zen5-nemotron-256k-optimizations.patch" ]]; then
    echo "=============================================================================="
    echo " [*] Bootstrapping llama-modchip repository..."
    echo "=============================================================================="
    if [[ ! -d "./llama-modchip" ]]; then
        git clone https://github.com/synchopate/llama-modchip.git ./llama-modchip
    fi
    cd ./llama-modchip
    exec bash ./install.sh "$@"
fi

REPO_ROOT="$(cd "$(dirname "$SCRIPT_SOURCE")" && pwd)"
WORKSPACE_DIR="${WORKSPACE_DIR:-${REPO_ROOT}/workspace}"
MODEL_PATH="${MODEL_PATH:-}"
MODEL_EXPLICIT=0
[[ -n "$MODEL_PATH" ]] && MODEL_EXPLICIT=1
SKIP_MODEL="${SKIP_MODEL:-0}"
USE_PREBUILT="${USE_PREBUILT:-0}"
USER_CORPUS_DIR=""
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${SERVER_PORT:-8789}"
GATEWAY_HOST="${GATEWAY_HOST:-127.0.0.1}"
GATEWAY_PORT="${GATEWAY_PORT:-8787}"

MODEL_REPO="ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF"
MODEL_REV="8cd082170ed313484be1cf44f256a5caa2cfdad3"
DEFAULT_MODEL_NAME="NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0.gguf"
MODEL_SHA256="61f87e75974e4b535dcdf9aad056541a9514f1dfa4538b463b081d19b7a00e3c"
HF_MODEL_URL="https://huggingface.co/${MODEL_REPO}/resolve/${MODEL_REV}/${DEFAULT_MODEL_NAME}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --workspace)
            WORKSPACE_DIR="$(realpath -m "$2")"
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
            sed -n '2,19p' "$0"
            exit 0
            ;;
        *)
            echo "[!] Unknown option: $1 (use --help for usage)"
            exit 1
            ;;
    esac
done

mkdir -p "$WORKSPACE_DIR/bin" "$WORKSPACE_DIR/models" "$WORKSPACE_DIR/routecrack" "$WORKSPACE_DIR/cache"

echo "=============================================================================="
echo " STEP 1/5: HARDWARE READINESS AUDIT & SYSTEM DEPENDENCIES"
echo "=============================================================================="
python3 "${REPO_ROOT}/scripts/check_cpu_readiness.py" || true

# Check essential build tools
MISSING_CMDS=()
for cmd in git cmake g++ curl python3; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
        MISSING_CMDS+=("$cmd")
    fi
done

if [[ ${#MISSING_CMDS[@]} -gt 0 ]]; then
    echo "[!] Missing build tools detected: ${MISSING_CMDS[*]}"
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
        echo "      Debian/Ubuntu : sudo apt-get install -y build-essential cmake git curl python3 libomp-dev"
        echo "      Fedora/RHEL   : sudo dnf install -y gcc-c++ cmake git curl python3 libomp-devel"
        echo "      Arch Linux    : sudo pacman -S base-devel cmake git curl python openmp"
        exit 1
    fi
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
echo " STEP 2/5: BUILDING OR DOWNLOADING PATCHED LLAMA-MODCHIP ENGINE"
echo "=============================================================================="
LLAMA_BUILD_DIR="${WORKSPACE_DIR}/llama.cpp"

detect_x86_prebuilt_asset() {
    local flags=""
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

PREBUILT_OK=0
if [[ "$USE_PREBUILT" == "1" && "$(uname -s)" == "Linux" && "$(uname -m)" == "x86_64" ]]; then
    ASSET_NAME="$(detect_x86_prebuilt_asset)"
    if [[ -n "$ASSET_NAME" ]]; then
        ASSET_URL="https://github.com/synchopate/llama-modchip/releases/download/v2.5.0/${ASSET_NAME}"
        echo "[*] CPUID auto-selected prebuilt binary: ${ASSET_NAME}"
        if curl -fL --progress-bar -o "${WORKSPACE_DIR}/bin/llama-server-nemotron" "$ASSET_URL"; then
            chmod +x "${WORKSPACE_DIR}/bin/llama-server-nemotron"
            echo "[+] Installed prebuilt binary: ${WORKSPACE_DIR}/bin/llama-server-nemotron (${ASSET_NAME})"
            PREBUILT_OK=1
        else
            echo "[!] Prebuilt download failed; falling back to local source build..."
        fi
    else
        echo "[i] Legacy x86_64 CPU without AVX2 detected; compiling scalar/AVX1 engine from source..."
    fi
fi

if [[ "$PREBUILT_OK" == "0" ]]; then
    BUILD_TARGET_MODEL="${TARGET_MODEL:-nemotron}"
    if [[ "$MODEL_PATH" == *[Dd]eep[Ss]eek* || "$MODEL_PATH" == *dsv4* ]]; then
        BUILD_TARGET_MODEL="dsv4"
    fi
    TARGET_MODEL="$BUILD_TARGET_MODEL" bash "${REPO_ROOT}/scripts/build_zen5_llama.sh" "$LLAMA_BUILD_DIR"

    cp -f "${LLAMA_BUILD_DIR}/build/bin/llama-server" "${WORKSPACE_DIR}/bin/llama-server-nemotron"
    cp -f "${LLAMA_BUILD_DIR}/build/bin/llama-quantize" "${WORKSPACE_DIR}/bin/llama-quantize" 2>/dev/null || true
    cp -f "${LLAMA_BUILD_DIR}/build/bin/llama-perplexity" "${WORKSPACE_DIR}/bin/llama-perplexity" 2>/dev/null || true
    echo "[+] Installed binary: ${WORKSPACE_DIR}/bin/llama-server-nemotron"
    echo "[+] Installed binary: ${WORKSPACE_DIR}/bin/llama-quantize"
fi

if [[ "$PREBUILT_OK" == "0" && -d "${LLAMA_BUILD_DIR}/include" ]]; then
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
echo " STEP 3/5: RESOLVING / DOWNLOADING NEMOTRON-3.5-30B-A3B Q4_0 GGUF (17.6 GiB / 18.9 GB)"
echo "=============================================================================="
if [[ -z "$MODEL_PATH" ]]; then
    if [[ -f "${WORKSPACE_DIR}/models/${DEFAULT_MODEL_NAME}" ]]; then
        MODEL_PATH="${WORKSPACE_DIR}/models/${DEFAULT_MODEL_NAME}"
    elif [[ -f "/opt/models/nemotron/${DEFAULT_MODEL_NAME}" ]]; then
        MODEL_PATH="/opt/models/nemotron/${DEFAULT_MODEL_NAME}"
    else
        MODEL_PATH="${WORKSPACE_DIR}/models/${DEFAULT_MODEL_NAME}"
    fi
fi

if [[ -f "$MODEL_PATH" ]]; then
    echo "[+] Found existing GGUF model: $MODEL_PATH"
elif [[ "$MODEL_EXPLICIT" == "1" ]]; then
    echo "[!] ERROR: Explicit --model file not found: $MODEL_PATH" >&2
    exit 1
elif [[ "$SKIP_MODEL" == "1" ]]; then
    echo "[i] --skip-model specified. Skipping 18.9 GB download."
    echo "    Place your GGUF at: $MODEL_PATH"
else
    mkdir -p "$(dirname "$MODEL_PATH")"
    echo "[*] Downloading ${DEFAULT_MODEL_NAME} from Hugging Face (pinned revision ${MODEL_REV:0:12})..."
    echo "    URL:    ${HF_MODEL_URL}"
    echo "    SHA256: ${MODEL_SHA256}"
    echo "    Dest:   ${MODEL_PATH}"
    curl -fL -C - --progress-bar -o "${MODEL_PATH}.part" "$HF_MODEL_URL"
    echo "[*] Verifying SHA-256 checksum..."
    ACTUAL_SHA256="$(sha256sum "${MODEL_PATH}.part" | awk '{print $1}')"
    if [[ "$ACTUAL_SHA256" != "$MODEL_SHA256" ]]; then
        echo "[!] ERROR: SHA-256 mismatch for ${MODEL_PATH}.part (removing corrupt partial file)!" >&2
        echo "    Expected: $MODEL_SHA256" >&2
        echo "    Got:      $ACTUAL_SHA256" >&2
        rm -f "${MODEL_PATH}.part"
        exit 1
    fi
    mv "${MODEL_PATH}.part" "$MODEL_PATH"
    echo "[+] Model download & SHA-256 verification complete: $MODEL_PATH"
fi

echo ""
echo "=============================================================================="
echo " STEP 4/5: BUILDING ROUTECRACK SPECULATIVE N-GRAM & CODEBOOK TABLES"
echo "=============================================================================="
if [[ -f "$MODEL_PATH" && -x "${WORKSPACE_DIR}/bin/rc_tokenize_nemotron" && -x "${WORKSPACE_DIR}/bin/rc_build_tables" ]]; then
    CORPUS_TXT="${WORKSPACE_DIR}/routecrack/bootstrap_corpus.txt"
    CORPUS_BIN="${WORKSPACE_DIR}/routecrack/corpus_nemotron_train.bin"
    echo "[*] Assembling coding & agentic <tool_call> training corpus..."
    {
        # Repeat repository source, docs, and agent tool schemas to build strong priors
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
elif [[ -f "$MODEL_PATH" ]]; then
    echo "[i] Prebuilt binary mode active — RouteCrack will use its zero-setup online Trie/PPM/Two-Zone engine."
else
    echo "[i] GGUF model not downloaded yet — RouteCrack will use its zero-setup online Trie/PPM/Two-Zone engine until tables are generated."
fi

echo ""
echo "=============================================================================="
echo " STEP 5/5: GENERATING AUTO-TUNED LAUNCH SCRIPTS"
echo "=============================================================================="
# Detect physical CPU core count and RAM size for safe defaults
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

if [[ "$TOTAL_RAM_GB" -ge 30 ]]; then
    CTX_SIZE=262144
else
    CTX_SIZE=65536
fi

mkdir -p "${WORKSPACE_DIR}/run"
printf '%s\n' "${SERVER_PORT}" > "${WORKSPACE_DIR}/run/server_port"
if [[ "$MODEL_EXPLICIT" == "1" ]]; then
    case "$MODEL_PATH" in
        /*) MODEL_ABS="$MODEL_PATH" ;;
        *)  MODEL_ABS="${PWD}/${MODEL_PATH}" ;;
    esac
    printf '%s\n' "${MODEL_ABS}" > "${WORKSPACE_DIR}/run/default_model"
fi

cat > "${REPO_ROOT}/start_server.sh" <<EOF
#!/usr/bin/env bash
# Auto-generated by install.sh — Starts CPU LLM Server with llama-modchip Auto-Profiler & RouteCrack
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
            echo "[i] 64GB Desktop OS Headroom Guard: defaulting to -c 32768 (~58.3 GiB VmRSS) so GUI/OS retain >=4 GiB free RAM."
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
echo " 🚀 Quickest way to start & chat (Unified ./modchip CLI):"
echo "      ./modchip start        # Starts server + gateway in background"
echo "      ./modchip chat         # Interactive terminal chat with live tok/s"
echo "      ./modchip bench        # Run 30s benchmark & print shareable scorecard"
echo "      ./modchip claude       # Launch Claude Code connected to local CPU engine"
echo ""
echo " 🌐 Or open the Live Web Chat & Telemetry HUD in your browser:"
echo "      http://${GATEWAY_HOST}:${GATEWAY_PORT}"
echo "=============================================================================="
