# llama-modchip on Windows — Porting Notes & Benchmarks (`v2.5.0`)

**Project:** `llama-modchip` `v2.5.0` (`DeepSeek-V4-Flash-0731` `VQ3-64` `56.40 GiB` & `Fractal-64` `56.57 GiB`, plus `Nemotron-3.5-30B-A3B` `VQ3-64` `13.30 GiB` & `Fractal-64` `14.36 GiB`)
**Test machine:** AMD Ryzen 7 9800X3D (8C/16T, 96 MB L3) · 64 GB DDR5 · NVIDIA RTX 4070 Ti 12 GB · NVMe PCIe 5.0
**OS:** Windows 11 Pro · engine built with MinGW-w64 GCC (UCRT64) + CMake + Ninja, `-march=native` (AVX-512F/BW/VBMI/VNNI/BF16)
**Original Port & 1M-Context Verification:** [`@MR1KRAKEN`](https://github.com/MR1KRAKEN) ([PR #27](https://github.com/synchopate/llama-modchip/pull/27), [Issue #28](https://github.com/synchopate/llama-modchip/issues/28)); x64 AVX-512 ABI & `VQ3_0` update in `commit 54c89bf` ([Issue #31](https://github.com/synchopate/llama-modchip/issues/31))

---

## 1. The problem

Stock `llama.cpp` (official `b12248` binaries and any upstream build) cannot load `Fractal-64` (`GGML_TYPE_Q2_0 = 42`) or `LUT-VQ3-64` (`GGML_TYPE_VQ3_0 = 43`) GGUF checkpoints:

```text
E llama_model_load: error loading model: check_tensor_dims: tensor 'blk.0.attn_kv.weight'
  has offset 596983872, expected 596590656
```

`Fractal-64` (`type 42`) uses a custom **30 bytes / 64 weights** block layout (`2B` fp16 super-scale + `4B` 8×4-bit sub-scales + `16B` low bits + `8B` high bits) instead of an 18-byte upstream block, and `LUT-VQ3-64` (`type 43`) adds a **24 bytes / 64 weights** 3D vector-quantized block (`2B` fp16 super-scale + `3B` 8×3-bit sub-scales + `19B` 3D/2D codebook indices). Upstream loaders reject `type 43` outright and miscalculate offsets on `type 42`.

## 2. What had to change for Windows

The shipped POSIX scripts (`install.sh`, `modchip`, `scripts/build_zen5_llama.sh`) rely on Linux/macOS APIs (`taskset`, `/dev/shm/llama_modchip.bin`, `sys/mman.h`), and GCC/MinGW-w64 on Windows x64 differs from Linux System V AMD64 in both ABI register preservation and PE/COFF `.rdata` alignment.

Five source and toolchain fixes are handled by `windows\build_patched_engine.ps1` and `patches/lut-vq3-64-avx512-vbmi.patch`:

| # | File | Problem | Fix |
| :-- | :--- | :--- | :--- |
| 1 | `ggml/src/ggml-cpu/ops.cpp` | `#include <sys/mman.h>` + `open("/dev/shm/llama_modchip.bin")` with no platform guard | Wrapped in `#ifndef _WIN32`; on Windows the attention policy comes from env (`LLAMA_ATTN_WINDOW`, `LLAMA_ATTN_SINKS`, `LLAMA_KV_L3_WINDOW`) |
| 2 | `ggml/src/ggml-cpu/ggml-cpu.c` | `__thread` and `__attribute__((aligned(64)))` are GCC-only | `MC_TLS` = `__declspec(thread)`; buffer aligned with `__declspec(align(64))` under `_MSC_VER` |
| 3 | `ggml/src/ggml-cpu/arch/x86/quants.c` | `__builtin_ctz(m)` inside the AVX-512 path | MSVC path uses `_BitScanForward` (`mc_ctz32`) |
| 4 | `src/llama-graph.cpp` | `ggml_dsv4_skip_csa_comp` is defined in a C file but referenced from C++ | C linkage declared under `_MSC_VER`; MinGW behaviour unchanged |
| 5 | `patches/lut-vq3-64-avx512-vbmi.patch` & `windows/build_patched_engine.ps1` | On Windows x64, `XMM6..XMM15` are callee-saved (`128-bit`), PE/COFF `.rdata` only guarantees 16-byte section alignment (`vmovdqa64` faults on 64-byte tables), and default `1–2 MB` Windows thread stacks overflow during deep graph builds ([Issue #31](https://github.com/synchopate/llama-modchip/issues/31)) | Replaced aligned 512-bit constant loads with unaligned `_mm512_loadu_si512` (`vmovdqu64`), isolated heavy AVX-512 kernels to avoid GCC UCRT64 `XMM6..XMM15` prologue/epilogue clobbers, and set `-Wl,--stack,16777216` (`16 MB` stack) |

One runtime constraint: for `deepseek4`/MLA the KV cache requires **identical K and V types** —
`-ctk q8_0 -ctv q4_0` fails with `model does not support different K and V cache types`, so `q8_0` is used for both.

## 3. Build

```powershell
# DeepSeek-V4-Flash 284B (Fractal-64 & VQ3-64, default):
powershell -ExecutionPolicy Bypass -File windows\build_patched_engine.ps1 -Target dsv4

# NVIDIA-Nemotron-3.5-30B-A3B (Fractal-64 & VQ3-64):
powershell -ExecutionPolicy Bypass -File windows\build_patched_engine.ps1 -Target nemotron

# Optional MSVC + CUDA build (experimental):
powershell -ExecutionPolicy Bypass -File windows\build_patched_engine.ps1 -Target dsv4 -Cuda
```

The script clones `ggml-org/llama.cpp` at **`d2f83055`**, applies `deepseek-v4-flash-fractal-256k.patch` (or `zen5-nemotron-256k-optimizations.patch` for `-Target nemotron`) followed by `lut-vq3-64-avx512-vbmi.patch` + `vq3_codebook.h`, copies the RouteCrack headers (`rc_predictor.h`, `rc_stockfish.h`, `rc_trie_v2.h`) into `common/`, applies the Windows portability guards, and builds a statically linked `windows/bin-modchip/llama-server.exe`.

Key flags: `-DGGML_NATIVE=ON -DGGML_OPENMP=OFF -DBUILD_SHARED_LIBS=OFF -DCMAKE_EXE_LINKER_FLAGS="-static -Wl,--stack,16777216" -D_WIN32_WINNT=0x0A00`.

## 4. Measured performance (`Ryzen 7 9800X3D`, `DeepSeek-V4-Flash 284B`)

`llama-bench -r 3`, `-t 8`, KV `q8_0`, flash-attn on:

| Configuration | pp512 | tg128 |
| :--- | ---: | ---: |
| CPU, Top-K 6 (lossless) | 21.3 tok/s | 6.35 ± 0.29 tok/s |
| CPU, Top-K 4 (pruned) | 21.1 tok/s | 8.04 ± 0.45 tok/s |
| CPU, 16 threads (SMT) | — | 6.41 tok/s (no gain) |
| GPU offload, 5 MoE layers in VRAM | 5.2 | 2.65 tok/s |
| GPU offload, dense-only | 3.7 | 2.71 tok/s |

Context scaling: 32K → 6.7, 128K → 6.9, 256K → 6.6 tok/s. The full **1M context (1,048,576)** starts and runs cleanly (~16 GB RAM free on a 64 GB host).

## 5. Why a 12 GB GPU does not help `DeepSeek-V4 284B`

1. MoE experts occupy ~54 GB of the 56.4–56.6 GiB checkpoint — only 5–7 of 43 layers fit in 12 GB VRAM (~15% of the work).
2. Custom `Q2_0` (`type 42`) and `VQ3_0` (`type 43`) kernels are CPU-only on Windows/Linux (no CUDA/Vulkan shader). Offloading requires pinning those tensors to CPU (`-ot "blk\.\d+\.(attn_.*|indexer.*|ffn_(gate|up)_shexp)\.weight=CPU"`), forcing PCIe synchronization every layer that costs more than partial MoE offload saves.

Net result: on a 12 GB card `DeepSeek-V4 284B` is faster on CPU alone. (For 30B/14B/8B standard `Q4_0`, `MXFP4`, or `Q3_K_M` checkpoints where the dense trunk runs natively on CUDA/Vulkan, use `windows\AVVIA_MODCHIP.bat` instead.)

## 6. Operational notes & pitfalls

1. **Never set `DSV4_MOE_TOPK` / `LLAMA_MOE_TOP_K` in the environment for `DeepSeek-V4`.** During load the engine derives `ffn_gate_tid2eid` dimensions from that variable; because the GGUF stores 6 hash-routed experts, any other value aborts with `expected 4, 129280, got 6, 129280`. Drive runtime Top-K pruning exclusively through `%TEMP%\dsv4_moe_cfg.bin` (`modchip_dsv4.ps1 -Fast`).
2. **Repetition penalties and DRY sampling:** Do **not** pass `--dry-sequence-breaker none` — DRY then penalizes repeated chat-template tokens and corrupts output. If enabling DRY, keep default sequence breakers (`--repeat-penalty 1.1 --repeat-last-n 256`).
3. **Long-prompt prefill timeouts:** At ~15–22 tok/s prefill on 8 cores, a 6,000-token cold prompt takes several minutes before the first token. Raise HTTP client stream idle timeouts above 300 s, or pair `llama-server.exe` with `--slot-save-path` so recurring system/tool prefixes restore in milliseconds.
4. **One server per port:** Context size and MoE placement are fixed at startup; changing presets requires restarting `llama-server.exe`.
5. **`cache_reuse` log notice:** `cache_reuse is not supported by this context` is expected for `deepseek4`; LCP prefix slot reuse still works normally.

## 7. Summary

`DeepSeek-V4-Flash 284B` (`VQ3-64` `56.40 GiB` and `Fractal-64` `56.57 GiB`) and `Nemotron-3.5-30B-A3B` (`VQ3-64` `13.30 GiB` and `Fractal-64` `14.36 GiB`) compile and run natively on Windows 11 via `windows\build_patched_engine.ps1` and `windows\AVVIA_DSV4.bat`, delivering **6.35–8.04 tok/s pure decode and 21.3–22.6 tok/s prefill up to 1M context on an 8-core Ryzen 7 9800X3D in 64 GB RAM**.
