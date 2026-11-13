# `llama-modchip v2.5.3` — Windows 10/11 Launchers & Patched Engine Builder

The `windows/` directory provides two complementary workflows on Windows 10/11:

1. **Patched CPU Engine for `LUT-VQ3-64` (`type 43`) & `Fractal-64` (`type 42`) (`DeepSeek-V4-Flash 284B` & `Nemotron-30B`):**
   - Build static AVX-512 / AVX2 binaries from source via **[`build_patched_engine.ps1`](build_patched_engine.ps1)** (`-Target dsv4` or `-Target nemotron`) and launch via **[`AVVIA_DSV4.bat`](AVVIA_DSV4.bat)** ([`modchip_dsv4.ps1`](modchip_dsv4.ps1)).
   - Verified up to **`1,048,576` (`1M`) context** on Windows 11 (`AMD Ryzen 7 9800X3D`, `64 GB` DDR5). Full instructions in **[`README_WINDOWS_PORT.md`](README_WINDOWS_PORT.md)** and **[`docs/WINDOWS_PORT.md`](../docs/WINDOWS_PORT.md)**.
2. **1-Click Hybrid GPU + CPU MoE Pack for Standard GGUFs (`RTX / Radeon 8–16 GB` + `16–128 GB RAM`):**
   - **[`AVVIA_MODCHIP.bat`](AVVIA_MODCHIP.bat)** ([`modchip_win.ps1`](modchip_win.ps1)) automates **Tensor-Role Splitting (`-ngl 99 --n-cpu-moe N --load-mode none`)**, **Asymmetric KV Quantization (`-ctk q8_0 -ctv q4_0 -fa on`)**, **Physical Core Pinning (`-t PhysCores`)**, and **Draftless N-Gram Speculative Decoding** using official `ggml-org/llama.cpp` CUDA 12.4 or Vulkan binaries.

---

## Quickstart: Hybrid GPU + CPU MoE Pack (`AVVIA_MODCHIP.bat`)

1. Open the `windows` folder and run **[`AVVIA_MODCHIP.bat`](AVVIA_MODCHIP.bat)** (or `./modchip start` from Git Bash / MSYS2).
2. On first run, [`modchip_win.ps1`](modchip_win.ps1) automatically:
   - Detects physical CPU core count, free System RAM, and live free GPU VRAM via `nvidia-smi`.
   - Downloads official `ggml-org/llama.cpp` Windows CUDA 12.4 binaries + CUDA runtime DLLs into `bin\` (no CUDA Toolkit required beyond the standard NVIDIA driver; falls back to Vulkan on AMD/Intel GPUs).
   - Offers **4 pre-configured standard GGUF profiles** (with resume-safe `curl.exe -C -` downloads into `models\`):
     - **`[1]` `Qwen3-30B-A3B` (`Q3_K_M`, `13.7 GiB`)** — *For `11 GB VRAM + 16 GB RAM`*: keeps the dense trunk + ~30 MoE layers on GPU (`~9.2 GB` VRAM) and `~4.5 GB` of sparse MoE layers in System RAM (`--load-mode none`).
     - **`[2]` `NVIDIA-Nemotron-3.5-30B-A3B` (`Q4_0`, `17.60 GiB` or `Fractal-MXFP4`, `15.95 GiB`)** — *128K–256K Hybrid Mamba-2 + MoE*: places Mamba-2 + GQA + top MoE layers in GPU VRAM and remaining MoE layers in System RAM (`75.1–80.8 tok/s` on Radeon RX 9070 XT 16 GB Vulkan with `Fractal-MXFP4`).
     - **`[3]` `Qwen3-14B` (`Q4_K_M`, `8.4 GiB`)** — *100% GPU (`0 GB` System RAM)*: fits inside an `11 GB` RTX 2080 Ti at `~50–58 tok/s`.
     - **`[4]` `NVIDIA-Nemotron-H-8B-Reasoning-128K` (`Q8_0`, `8.0 GiB`)** — *100% GPU 128K*: 8-bit precision with 128K context (`~210 MB` KV cache) in VRAM at `~100+ tok/s`.
   - Computes `--n-cpu-moe` dynamically from live free VRAM with a `550 MB` WDDM safety reserve so Windows does not spill allocations into shared GTT memory, auto-detects `--load-mode none` vs `--no-mmap` and `--spec-ngram-size-n` CLI flags, and starts `http://127.0.0.1:8080` (`/v1` OpenAI-compatible API).

3. To benchmark your Windows PC, run **[`BENCHMARK_PC.bat`](BENCHMARK_PC.bat)**. For custom `LUT-VQ3-64` / `Fractal-64` checkpoints, see **[`README_WINDOWS_PORT.md`](README_WINDOWS_PORT.md)**.
