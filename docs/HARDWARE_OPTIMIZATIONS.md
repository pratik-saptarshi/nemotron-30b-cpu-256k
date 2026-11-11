# Supported Chips & Hardware-Specific Optimizations Matrix

> **Back to:** [Main Overview (`README.md`)](../README.md) · [LUT-VQ3-64 Quality Gate (`benchmarks/QUALITY_GATE_VQ3_64.md`)](../benchmarks/QUALITY_GATE_VQ3_64.md) · [Nemotron Guide (`docs/NEMOTRON_OPTIMIZATIONS.md`)](NEMOTRON_OPTIMIZATIONS.md) · [DeepSeek-V4-Flash 284B Guide (`docs/DEEPSEEK_V4_OPTIMIZATIONS.md`)](DEEPSEEK_V4_OPTIMIZATIONS.md) · [Windows Port (`docs/WINDOWS_PORT.md`)](WINDOWS_PORT.md) · [Apple Silicon Pack (`apple-silicon/README.md`)](../apple-silicon/README.md)

---

## 1. Master Hardware & Processor Matrix

| Processor / Platform Family | Verified Chips | Active ISA & Kernel Paths | Memory & Cache Co-Design | Verified Speed (`Nemotron-30B` / `DeepSeek-V4 284B`) |
| :--- | :--- | :--- | :--- | :--- |
| **1. AMD Zen 5 (`Desktop / Mobile / Server`)** | `Ryzen 9 9955HX`, `9950X` (`@nadams`), `Ryzen 7 9800X3D` (`@MR1KRAKEN`), `EPYC 9005 Turin` | Full single-pump 512-bit **AVX-512 (`VPDPBUSD`, `VPRORVQ`, `VBMI VPERMI2B`, `BF16`)** + AVX-512BW `_1x8/_1x4/_1x3/_1x2` + `MQA8` FlashAttn | 4-Level Register+L1D(`48KB`)+L2(`1MB`)+L3(`64MB`) tiling, `_MM_HINT_NTA` victim-L3 isolation, `2MB` THP (`6.14 GB` L2 STLB) | **Nemotron**: `20.8–60.4 tok/s` (`135.6–306.8` pp512)<br/>**DeepSeek-V4**: `9.05–25.3 tok/s` (**`41.1–44.4`** pp512; `1M` ctx on 9800X3D) |
| **2. AMD Multi-CCD Threadripper & EPYC (`Zen 4 / Zen 2`)** | `Threadripper PRO 7985WX` (`64C`, `@aifartist`), `EPYC 7402P` (`24C` `AVX2`, `@AdamKlob`), `EPYC 9004` | **AVX-512 VNNI** / **AVX2 (`vpmaddwd`)** tile repacking + lockless `192 KB` L2 Zobrist TT (`Stockfish-PVS v2`) | **Deterministic Per-CCD Stride (`8 CCDs`, `128–256 MB` L3)**: saturates 8-channel DDR5 / DDR4 memory bandwidth | **7985WX (DDR5)**: **`111.62 tok/s` warm** / **`386.0` pp**<br/>**EPYC 7402P (DDR4 AVX2)**: **`27.29–31.22 tok/s`** / **`158.7` pp** |
| **3. AMD 3D V-Cache + Radeon Vulkan Hybrid (`Linux / Windows`)** | `Ryzen 9 7945HX3D` (`128 MB` L3) + `Radeon RX 9070 XT 16GB` (`RADV / Vulkan`) | Native **Vulkan `SSM_SCAN`** + **`Fractal-MXFP4` (`4.25 bpw`) GPU shaders** + AVX-512 CPU MoE overflow | **Tensor-Role Splitting (`-ngl 99 --n-cpu-moe N --load-mode none`)**: Mamba-2 + GQA + top MoE on GPU VRAM, remaining MoE in DDR5 | **Nemotron (`256K` ctx)**: **`75.10 tok/s` new code / `80.80 tok/s` refactor**, **`563.00 tok/s` prefill** |
| **4. Intel Core / Xeon + NVIDIA RTX CUDA Hybrid (`Windows 10/11 & Linux`)** | `Core i9-9900K`, `13900K/14900K`, `Core Ultra`, `Xeon` + `RTX 2080 Ti 11GB / 3090 / 4090` | **CUDA 12.4** backbone + **AVX2 (`vpsllvd + vpmaddwd`)** / **AVX-512 VNNI** CPU MoE + `_WIN32` `CreateFileMappingA` | **Live-VRAM `--n-cpu-moe` Solver (`windows/modchip_win.ps1`)**: reserves `550 MB` WDDM safety margin to prevent slow GTT spillover + `--load-mode none` | **Qwen3-30B-A3B**: fits in `11GB VRAM + 16GB RAM`<br/>**Qwen3-14B / Nemotron-8B**: **`50–100+ tok/s`** |
| **5. Apple Silicon (`M1 / M2 / M3 / M4 / M5`)** | `M5 Max` (`64 GB`), `M4 Max` (`36/64 GB`), `M4 Pro` (`24 GB`), `M3 Max`, `M1 Max` | **Register-Tiled `Fractal-64` (`Q2_0`) Metal Fast-Decode Shaders ([PR #24](https://github.com/synchopate/llama-modchip/pull/24))** + **ARM64 NEON `SDOT`** + portable `HARPCache` | Zero-copy unified memory residency (`0` swap at `128K–256K` context) + `Stockfish-PVS v2` | **Nemotron `Q2_0`**: `57.65 tok/s` M1 Max (`+86%`), `78.60` M3 Max, `72.5` M4 Max (`979.9` pp512)<br/>**DeepSeek-V4 284B (M5 Max)**: **`27.0 tok/s` `tg128`** / **`169.4` `pp512`** |
| **6. ARM64 Cloud & Edge (`AWS Graviton / Ampere`)** | `AWS Graviton3 m7g.2xlarge` (`8 vCPU Neoverse-V1`, `32 MB` L3, DDR5) | **ARMv8.4-A DotProd (`SDOT`) + 256-bit SVE** + `Stockfish-PVS v2 (M=7)` | 1-vCPU-per-physical-core (`-t 8`, zero SMT), `32 MB` System Level Cache retention | **Nemotron**: **`41.05 tok/s` C++20 / `44.97 tok/s` refactor**, **`101.67 tok/s` prefill** |
| **7. AMD Zen 4 Compact APU / Mini-PC (`Fedora / Linux`)** | `Ryzen 7 8745HS` (`GEEKOM A8`, `8C/16T`, `16 MB` L3 + `Radeon 780M`, `64 GB` DDR5) | **AVX-512 VNNI** + optional `Radeon 780M` UMA iGPU offload ([`fedora/nemotron.sh`](../fedora/nemotron.sh)) | `LLAMA_KV_L3_WINDOW=1024` tuned for `16 MB` L3 cache | Verified 256K full-stack setup on Fedora 44 (`19.48–23.38 tok/s` on 8745H) |

---

## 2. Processor-by-Processor Optimization Breakdown

### 2.1. AMD Zen 5 (`Ryzen 9 9955HX / 9950X`, `Ryzen 7 9800X3D`, `EPYC 9005 Turin`)
1. **Single-Cycle 512-Bit Datapath (`_mm512_rorv_epi64`, `_mm512_permutex2var_epi8`, & `VPDPBUSD`)**:
   - Unlike Zen 4 (which double-pumps 512-bit instructions across 256-bit execution units), Zen 5 executes 512-bit AVX-512 instructions in a single clock cycle.
   - In `ggml_vec_dot_q2_0_q8_0_generic` (`ggml/src/ggml-cpu/quants.c`), `_mm512_rorv_epi64` (`vprorvq`) extracts the 64-bit upper bit-plane (`qh`) of `Fractal-64` (`Q2_0`) blocks in one instruction.
2. **48 KB L1D Cache 3D Tiling (`TILE_I0 = 64`)**:
   - Zen 5 features a `48 KB` 12-way L1D cache per core (up from `32 KB` on Zen 4). Our 3D `MUL_MAT` tiler (`ggml-cpu.c`) tiles multi-head projections (`ne02 > 1`, such as `DeepSeek-V4`'s `wo_a`) in `64`-row chunks so the `[4096, 64]` activation tile fits inside L1D across all threads.
3. **Victim-L3 Isolation (`_MM_HINT_NTA`) & `2MB` Transparent HugePages**:
   - Zen 5's `32 MB` per-CCD L3 cache is a victim cache filled by L2 evictions. Streaming `1.4–3.2 GB/token` of MoE weights with temporal loads pollutes L3; issuing `_mm_prefetch(..., _MM_HINT_NTA)` in `q4_0`, `mxfp4`, `q2_0`, `vq3_0`, `q4_K`, and `q6_K` dot-product kernels marks cold weight lines as non-temporal, protecting the hot KV window (`LLAMA_KV_L3_WINDOW`) and `192 KB` `Stockfish-PVS` tables.
4. **Built-In `LUT-VQ3-64` (`GGML_TYPE_VQ3_0 = 43`, `24 B / 64 W = 3.00 bpw`) via AVX-512 VBMI `VPERMI2B` ([`vq3/`](../vq3/), [`patches/lut-vq3-64-avx512-vbmi.patch`](../patches/lut-vq3-64-avx512-vbmi.patch))**:
   - **Built-In GGML Tensor Type (`v2.5.0`)**: Wired directly into `ggml` (`GGML_TYPE_VQ3_0 = 43`) with multi-column GEMM/GEMV tiling (`_1x8`, `_1x4`, `_1x3`, `_1x2`, `1x1`) in `ggml/src/ggml-cpu/arch/x86/quants.c`, portable AVX2 + scalar fallbacks in `ggml/src/ggml-cpu/quants.c`, and Windows x64 PE/COFF unaligned `.rdata` loads (`_mm512_loadu_si512`).
   - **`block_vq3_0` Layout (`24 bytes` per `64` weights)**: `2B` `FP16` super-scale `d`, `3B` (`s24[3]`) storing `8 × 3-bit` unsigned sub-scales (`mult = 2 + u_b ∈ {2..9}`), and `19B` (`qs[19]`, `152 bits`) storing `20` 7-bit 3D Lloyd-Max indices (`140 bits`, weights `0..59`) + `2` 6-bit 2D tail indices (`12 bits`, weights `60..63`).
   - **Register-Resident 3D Codebook Lookup**: All 128 3D `int8` centroids (`384 bytes`) stay in 6 ZMM registers (`cb0_lo/hi`, `cb1_lo/hi`, `cb2_lo/hi`), resolving 64 byte lookups per `_mm512_permutex2var_epi8` (`VPERMI2B`) instruction with zero L1/L2 cache reads and achieving **`13.57 dB` SQNR** on `BF16` MoE weights (`+3.42 dB` over `Q2_K`).

### 2.2. Multi-CCD AMD Threadripper PRO & EPYC (`7985WX` 8-CCD / `64C`)
1. **Cross-CCD Bandwidth Contention**:
   - On an 8-CCD processor (`64` cores sharing an 8-channel DDR5-6000 controller), launching 64 threads (`8` cores/CCD) causes atomic work-stealing contention and Infinity Fabric cache-coherency traffic, reducing decode throughput from `80.14 tok/s` to `51.80 tok/s`.
2. **Deterministic Per-CCD Stride (`4C/CCD` Sweet Spot)**:
   - [`scripts/llama_modchip.py`](../scripts/llama_modchip.py) queries `/sys/devices/system/cpu/cpu*/cache/index3/shared_cpu_list` to discover exact L3 CCD domains and selects `4` physical cores per CCD (`-t 32`), achieving **`386.00 tok/s` prefill**, **`80.14 tok/s` cold decode**, and **`111.62 tok/s` warm decode**.

### 2.3. Hybrid GPU + CPU MoE Systems (`Windows 10/11 CUDA`, `Linux/Windows AMD Vulkan`, & `Apple Silicon Metal`)
1. **Why Tensor-Role Splitting (`-ngl 99 --n-cpu-moe N --load-mode none`) Outperforms Layer-Index Splitting**:
   - Standard `-ngl N` offloads layers `0..N-1` (both attention/SSM and large MoE tensors) to GPU and leaves layers `N..L` entirely on CPU, forcing KV attention and dense projections on upper layers onto CPU RAM.
   - Tensor-Role Splitting (`-ngl 99 --n-cpu-moe N`) keeps **100% of the dense backbone, GQA/MLA KV cache, and Mamba-2 `SSM_SCAN` inside GPU VRAM (`616–1000 GB/s`)**, placing only $N$ sparse MoE expert layers (`ffn_*_exps`) in System RAM.
2. **Quantization Format Selection Across Backends**:
   - **Vulkan & CUDA GPU+CPU Hybrids (`Fractal-MXFP4` `15.95 GiB` or `Q4_0` `17.59 GiB`)**: `llama.cpp` provides native Vulkan and CUDA compute shaders for `MXFP4` and `Q4_0`, delivering **`75.1 tok/s` (new code) / `80.8 tok/s` (refactor)** at `256K` context on an `RX 9070 XT 16GB` (`~15.2 GB` VRAM+RAM active) vs `69.7 / 76.2 tok/s` on `Q4_0`.
   - **Apple Silicon Metal (`Fractal-64` `Q2_0` `14.36 GiB` or `Q4_0` `17.59 GiB`)**: [`patches/apple-silicon/`](../patches/apple-silicon/) includes native register-tiled Metal shaders for `Fractal-64` (`Q2_0 = 42`, [PR #24](https://github.com/synchopate/llama-modchip/pull/24)). Because `LUT-VQ3-64` (`VQ3_0 = 43`) targets x86-64 AVX-512 VBMI, Apple Silicon users should select `Fractal-64` (`14.36 GiB` / `56.57 GiB`).
   - **x86-64 Pure CPU (`LUT-VQ3-64` `13.30 GiB` / `56.40 GiB` or `Fractal-64` `14.36 GiB` / `56.57 GiB`)**: Use `LUT-VQ3-64` on AVX-512 VBMI CPUs (Zen 4/5, Ice Lake+) for lowest RAM footprint (`16.04 GiB` / `58.31 GiB` `VmRSS`), or `Fractal-64` on AVX2-only CPUs.
3. **Windows WDDM Safety Margin (`windows/modchip_win.ps1`)**:
   - On Windows 10/11, exceeding physical VRAM causes the WDDM scheduler to page GPU allocations into shared system RAM (`GTT`) over PCIe. [`windows/modchip_win.ps1`](../windows/modchip_win.ps1) queries live free VRAM via `nvidia-smi` right before launch and reserves a `550 MB` WDDM safety margin.
