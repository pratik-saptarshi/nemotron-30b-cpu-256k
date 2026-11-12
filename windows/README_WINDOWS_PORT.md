# `LUT-VQ3-64` & `Fractal-64` GGUFs on Windows (`v2.5.2` Patched Engine)

**Stock Windows binaries cannot load `Fractal-64` (`type 42`) or `LUT-VQ3-64` (`type 43`) GGUFs.**
`Fractal-64` uses `GGML_TYPE_Q2_0 = 42` with a 30-byte / 64-weight block and `LUT-VQ3-64` uses `GGML_TYPE_VQ3_0 = 43` with a 24-byte / 64-weight 3D vector-quantized block, whereas upstream `llama.cpp` (`llama-bXXX-bin-win-cuda-x64` downloaded by `modchip_win.ps1`) only implements an older 18-byte `Q2_0` layout and rejects `type 43`. Loading with stock binaries fails with:

```text
check_tensor_dims: tensor 'blk.0.attn_kv.weight' has offset 596983872, expected 596590656
```

`AVVIA_MODCHIP.bat` (`modchip_win.ps1`) remains the GPU+CPU hybrid launcher for standard GGUFs (`Qwen3`, `Nemotron Q4_0` / `MXFP4`, `Nemotron-H-8B`). For `Fractal-64` and `LUT-VQ3-64` checkpoints (`DeepSeek-V4-Flash 284B` and `Nemotron-3.5-30B`), use the patched engine workflow below.

## 1. Build the `v2.5.2` patched engine

With MSYS2 MinGW-w64 UCRT64 (`gcc >= 14`), Git, CMake, and Ninja on `PATH`:

```powershell
# For DeepSeek-V4-Flash 284B (MaxQ-Fractal 56.93 GiB, VQ3-64 56.40 GiB & Fractal-64 56.57 GiB — default):
powershell -ExecutionPolicy Bypass -File windows\build_patched_engine.ps1 -Target dsv4

# For NVIDIA-Nemotron-3.5-30B-A3B (VQ3-64 13.30 GiB & Fractal-64 14.36 GiB):
powershell -ExecutionPolicy Bypass -File windows\build_patched_engine.ps1 -Target nemotron
```

> **Note on Prebuilt `v2.4.0` Binaries:** Community static `.exe` builds for `Fractal-64` (`type 42` only, prior to `v2.5.0` `VQ3_0 = 43` and the `54c89bf` UCRT64 AVX-512 ABI fix) are archived at [`MR1KRAKEN/llama-modchip v2.4.0-windows-cpu`](https://github.com/MR1KRAKEN/llama-modchip/releases/tag/v2.4.0-windows-cpu). For `v2.5.x` `MaxQ-Fractal` / `VQ3-64` (`type 43`) models or Zen 4/5 AVX-512 hosts on Windows 11, compile directly with `windows\build_patched_engine.ps1` above.

`build_patched_engine.ps1` clones `ggml-org/llama.cpp` at `d2f83055`, applies `patches/deepseek-v4-flash-fractal-256k.patch` (or `patches/zen5-nemotron-256k-optimizations.patch`) plus `patches/lut-vq3-64-avx512-vbmi.patch` and `patches/vq3_codebook.h`, copies `speculative/cpp/rc_*.h` into `common/`, applies all Windows portability fixes (including unaligned `vmovdqu64` `.rdata` loads, Windows x64 `XMM6..XMM15` ABI safety, and `16 MB` stack size `-Wl,--stack,16777216`), and outputs static `windows\bin-modchip\llama-server.exe`, `llama-cli.exe`, and `llama-bench.exe`.

## 2. Run

Place the GGUF shards (`DeepSeek-V4-Flash-0731-MaxQ-Fractal-*.gguf`, `DeepSeek-V4-Flash-0731-VQ3-*.gguf`, or `DeepSeek-V4-Flash-0731-Fractal-*.gguf`) in `windows\models\` and run:

```bat
windows\AVVIA_DSV4.bat
```

`modchip_dsv4.ps1` detects physical cores and RAM, writes `%TEMP%\dsv4_moe_cfg.bin`, and starts an OpenAI-compatible server at `http://127.0.0.1:8080/v1` (pass `-OpenBrowser` to launch the browser UI automatically).

Useful flags for `windows\modchip_dsv4.ps1`: `-Ctx 65536`, `-Fast` (`Top-4` $L_2$-quadrature pruning), `-KvType q4_0`, `-NPredict N`, `-Bench`.
`windows\make_shortcuts.ps1` generates desktop shortcuts for context presets from `32K` up to `1M`.

## 3. Key constraints

- `deepseek4` / MLA requires matching K and V KV-cache types (`-ctk q8_0 -ctv q8_0` or `-ctk q4_0 -ctv q4_0`; mixed `-ctk q8_0 -ctv q4_0` is rejected).
- Never export `DSV4_MOE_TOPK` or `LLAMA_MOE_TOP_K` in the environment when loading `DeepSeek-V4-Flash`; control Top-K pruning strictly via `%TEMP%\dsv4_moe_cfg.bin` (`-Fast`).
- Raise HTTP client stream idle timeouts above 300 s for multi-thousand-token cold prompts on 8-core CPUs.

Full engineering report and Ryzen 7 9800X3D benchmarks: [`docs/WINDOWS_PORT.md`](../docs/WINDOWS_PORT.md).
