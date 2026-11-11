# Fractal-64 GGUF on Windows (patched engine)

**Stock Windows binaries cannot load the Fractal GGUF.** The model uses `GGML_TYPE_Q2_0` with the
patch's 30-byte / 64-weight block, while upstream `llama.cpp` (including the official
`llama-bXXX-bin-win-cuda-x64` zips downloaded by `modchip_win.ps1`) implements the old 18-byte block.
Loading fails with:

```
check_tensor_dims: tensor 'blk.0.attn_kv.weight' has offset 596983872, expected 596590656
```

`modchip_win.ps1` remains the launcher for the *other* models (Qwen3, Nemotron Q4_0, etc.) that run on
official binaries. For the Fractal GGUF use the flow below.

## 1. Build the patched engine (or download prebuilt `.exe` binaries)

> **Prebuilt Static Windows Binaries (Zero DLLs, No MinGW Required):**
> Statically linked `llama-server.exe`, `llama-cli.exe`, and `llama-bench.exe` (both **native AVX-512** for Zen 4/5 & Sapphire Rapids+ and **portable AVX2** for Haswell 2013+ / all AMD Zen) are available at:
> 👉 [**`MR1KRAKEN/llama-modchip v2.4.0-windows-cpu`**](https://github.com/MR1KRAKEN/llama-modchip/releases/tag/v2.4.0-windows-cpu) *(extract into `windows\bin-modchip\`)*.

To build from source with MinGW-w64:

```
powershell -ExecutionPolicy Bypass -File windows\build_patched_engine.ps1
```

Requirements: Git, CMake, Ninja, MinGW-w64 GCC >= 14 on `PATH`.

The script clones `ggml-org/llama.cpp` at `d2f8305` (the base commit for `patches/deepseek-v4-flash-fractal-256k.patch`
and `patches/lut-vq3-64-avx512-vbmi.patch`), applies the patches, copies `speculative/cpp/rc_*.h` into `common/`, applies the four
Windows portability fixes and writes `windows/bin-modchip/llama-server.exe` (static, no DLLs).

The fixes, in short:

| File | Problem | Fix |
| :--- | :--- | :--- |
| `ggml-cpu/ops.cpp` | `sys/mman.h` + `open("/dev/shm/...")` without a guard | `#ifndef _WIN32`; policy taken from `LLAMA_ATTN_WINDOW` / `LLAMA_ATTN_SINKS` / `LLAMA_KV_L3_WINDOW` |
| `ggml-cpu/ggml-cpu.c` | `__thread`, `__attribute__((aligned(64)))` | `__declspec(thread)` / `__declspec(align(64))` under `_MSC_VER` |
| `ggml-cpu/arch/x86/quants.c` | `__builtin_ctz` | `_BitScanForward` (`mc_ctz32`) on MSVC |
| `src/llama-graph.cpp` | C global referenced from C++ | C linkage under `_MSC_VER` |

`-Cuda` builds the same tree with MSVC + CUDA instead (slower for this model, see the report).

## 2. Run

```
windows\AVVIA_DSV4.bat
```

Put the three `DeepSeek-V4-Flash-0731-Fractal-*.gguf` shards into `windows\models\` first.
The launcher prints the detected hardware, writes `%TEMP%\dsv4_moe_cfg.bin`, starts an OpenAI-compatible
server on `http://127.0.0.1:8080` (`/v1`, model name `deepseek-v4-flash`) and does not open a browser
unless `-OpenBrowser` is passed.

Useful flags: `-Ctx 65536`, `-Fast` (Top-K 4), `-KvType q4_0`, `-NPredict N`, `-Bench`.

`make_shortcuts.ps1` creates desktop shortcuts for every context preset (32K … 1M) in fast/exact variants.

## 3. Things that bite

* `deepseek4` / MLA requires equal K and V cache types: `-ctk q8_0 -ctv q4_0` is rejected.
* Do **not** export `DSV4_MOE_TOPK` / `LLAMA_MOE_TOP_K`: during load the patch derives the
  `ffn_gate_tid2eid` shape from that value and the model aborts
  (`expected 4, 129280, got 6, 129280`). Top-K pruning goes through `%TEMP%\dsv4_moe_cfg.bin` only.
* Prefill is ~15-18 tok/s, so a 6 000-token prompt waits ~5 minutes before the first token.
  Raise client idle timeouts.
* The model degenerates into repetition loops on open-ended tasks; sampling penalties reduce but do not
  cure it. See the report.

Measured performance, GPU analysis and the full list of pitfalls: [`docs/WINDOWS_PORT.md`](../docs/WINDOWS_PORT.md).
