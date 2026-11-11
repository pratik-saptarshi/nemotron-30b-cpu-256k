# NVIDIA Nemotron (`30B-A3B` & `H-8B`) — Optimization Guide

> **Back to:** [Main Overview (`README.md`)](../README.md) · [LUT-VQ3-64 Quality Gate (`benchmarks/QUALITY_GATE_VQ3_64.md`)](../benchmarks/QUALITY_GATE_VQ3_64.md) · [Hardware & Chip Matrix (`docs/HARDWARE_OPTIMIZATIONS.md`)](HARDWARE_OPTIMIZATIONS.md) · [DeepSeek-V4-Flash 284B Guide (`docs/DEEPSEEK_V4_OPTIMIZATIONS.md`)](DEEPSEEK_V4_OPTIMIZATIONS.md) · [Full Paper (`ARCHITECTURE.md`)](../ARCHITECTURE.md)

---

## 1. Architectural Profile: `Nemotron-3.5-30B-A3B` at 256K Context

`NVIDIA-Nemotron-3.5-Lightning-30B-A3B` (`nemotron_h_moe`) is a **52-layer Hybrid Mamba-2 SSM + GQA Attention + Sparse MoE** model (`30.2B` total parameters, **`3.0B` active per token**):
- **23 Mamba-2 State-Space Layers (`SSM`)**: Maintain a constant-size $O(1)$ recurrent state (`18.4 MB` total across all 23 layers) whether context is `1K` or `256K` tokens.
- **6 Grouped-Query Attention Layers (`GQA`, `2` KV heads per layer = `12` KV heads total)**: Only **6 out of 52 layers** allocate a growing KV cache. At **`262,144`-token (`256K`) context**, the KV cache occupies **`384 MB` (`Q4_0`) to `432 MB` (`EAKV Q8_0/Q4_0`)** (`624 MB` with static `-ctk q8_0 -ctv q4_0`) — compared to `7.1–16.0 GB` on standard 48-layer GQA Transformers.
- **23 Sparse MoE Layers (`128` routed experts/layer, `top-6` active + `1` double-width shared expert)**: `~1.4–1.8 GB` of active weights read per token during `N=1` decode.

---

## 2. Summary of Optimizations Applied to `Nemotron-3.5-30B-A3B`

| Optimization Layer | Technique Implemented | Target Bottleneck Solved | Measured Gain (`Zen 5` / `Threadripper` / `Vulkan` / `Metal`) |
| :--- | :--- | :--- | :--- |
| **1. Quantization (`LUT-VQ3-64`)** | Block-64 3D Lloyd-Max Vector Quantizer (`GGML_TYPE_VQ3_0 = 43`, `24B / 64w = 3.00 bpw`) via AVX-512 VBMI `VPERMI2B` ([`scripts/build_nemotron_vq3.py`](../scripts/build_nemotron_vq3.py)) | Reduces MoE `up_exps` footprint by `-33.3%` vs `Q4_0` and `-20.0%` vs `Fractal-64` without scalar 2–3 bit grid loss (`+3.42 dB` SQNR over `Q2_K`) | **`13.30 GiB` shipped Hybrid (`13.08 GiB` full `up_exps` / `16.04 GiB` RSS)**, **`24.18 tok/s` E2E (`+23.7%` vs `Q4_0`)**, **`70.19%` PVS acc**, **`100/100` Gate** |
| **2. Quantization (`Fractal-64`)** | Block-64 2-Level Hierarchical Non-Linear `Q2_0 = 42` (`30B / 64w = 3.75 bpw`) from `BF16` master weights ([`scripts/build_fractal_moe_gguf.sh`](../scripts/build_fractal_moe_gguf.sh)) | `1856 × 2688` expert dimensions are not divisible by `256`, causing `Q3_K`/`IQ3_S` in `llama.cpp` to fall back to `Q4_0` | **`17.59 GiB -> 14.36 GiB` (`-6.01 GiB` idle RAM)**, **`+28.0%` CPU decode (`25.02 tok/s`)**, **`78.60 tok/s` M3 Max Metal ([PR #24](https://github.com/synchopate/llama-modchip/pull/24))**, **`100/100` Gate** |
| **3. Quantization (`Fractal-MXFP4`)** | Asymmetric Block-32 `MXFP4` (`4.25 bpw`) on routed MoE experts + `Q8_0`/`Q4_0` on GQA/SSM backbone + `_MM_HINT_NTA` | Bandwidth bottleneck on CPU & VRAM capacity on `16 GB` GPUs | **`15.95 GiB`**, **`+33.5%` CPU decode (`26.08 tok/s`)**, **`75.1–80.8 tok/s` on AMD Vulkan Hybrid** |
| **4. SIMD Kernels (`Q2_0` & `VQ3_0`)** | Single-cycle AVX-512 VBMI `VPERMI2B` (`VQ3_0`) and AVX2 variable bit-shifts (`_mm256_sllv_epi32` / `_mm256_srlv_epi32`) fused into `vpmaddwd` (`Q2_0`) | Scalar bit-unpacking overhead in sub-4-bit hierarchical blocks | **`+55%` `Q2_0` kernel speedup** (`86.5 ms -> 55.8 ms` per 10M blocks), register-resident 384-byte `VQ3_0` codebook |
| **5. Speculative Engine (`Stockfish-PVS v2`)** | `192 KB` L2-resident Zobrist Transposition Tables (`tt5/tt4/tt3` in [`rc_stockfish.h`](../speculative/cpp/rc_stockfish.h)) + 1-Ply Quiescence PVS + `RouteCrack` 8-Level FNV-64 Cascade + Macro Codebook (`M=7`) | 1B neural draft models read `3.0 GB` from DDR5 (`40 ms`) and flush L3 cache | **`48.39 tok/s` avg / `60.40 tok/s` peak** on 16C Zen 5 (`96.4%` acc, `len 6.90`); **`111.62 tok/s` warm** on Threadripper 7985WX |
| **6. Cache Isolation (`_MM_HINT_NTA`)** | Bidirectional Non-Temporal L3 bypass on cold MoE weights and KV tokens older than `LLAMA_KV_L3_WINDOW=2048` | Streaming `1.4–1.8 GB/tok` of MoE weights evicts hot Mamba-2 state and recent KV tokens from `64 MB` L3 | Keeps `18.4 MB` Mamba-2 state + `4.88 MiB` hot KV window in L3 cache |
| **7. Virtual Memory (`2MB HugePages`)** | `MADV_HUGEPAGE` on weight mmap + compute arenas + `2MB`-aligned `posix_memalign` | `17.6 GiB` in `4KB` pages = `4.5M` PTEs, thrashing the `3,072`-entry Zen 5 L2 STLB (`14 ms/tok` page-walk stall) | Expands L2 STLB coverage to **`6.14 GB`**, cutting dTLB page-walk latency from `14.0 ms` to **`0.8 ms/tok` (`-99.6%` misses)** |
| **8. Weight Tile Repacking (`q4_0_8x8`)** | In-memory AVX-512 VNNI (`VPDPBUSD`) / AVX2 interleaved `8×8` matrix tile repacking at model load | Cache-line splits and nibble-unpacking stalls during GEMV/GEMM | **`+19%` raw GEMV throughput** and **`241–306.8 tok/s` (`pp512`) prefill** on 16C Zen 5 |
| **9. KV Cache (`EAKV + VPI`)** | Entropy-Guided Asymmetric KV (`Q8_0` Keys for RoPE phase precision, `Q4_0` Values) + Virtual Positional Indexing | FP16 KV cache growth and RoPE degradation at `128K–256K` context | Full **`256K` context in `432 MB`**, `100%` passkey needle retrieval |
| **10. Agentic Loop (`HARP-Cache` Slot-0)** | `/dev/shm` memory-mapped KV + Mamba-2 recurrent state snapshot + SSE Gateway auto-restore | Re-evaluating Claude Code's `22.2K`-token 40-tool system prompt takes `130.8 s` on CPU | **`18.9 ms` warm turn restore** |
| **11. Dynamic MoE Pruning (`Top-K 6->4`)** | Opt-in norm-preserving dynamic MoE router truncation (`LLAMA_MOE_TOP_K=4` via `/dev/shm/llama_modchip.bin`) | Reading 6 routed experts per layer on bandwidth-constrained DDR4/DDR5 | **`-33%` routed MoE DRAM traffic** (switchable live via TUI) |

---

## 3. Verified Quality & Speed Gate (`100 / 100` Score)

All four quantization profiles (`Baseline-Q4_0`, `Fractal-MXFP4`, `Fractal-64-BF16`, and `LUT-VQ3-64`) pass the automated grading suite [`benchmarks/eval_quality_and_tools.py`](../benchmarks/eval_quality_and_tools.py) (full reports in [`benchmarks/QUALITY_GATE_VQ3_64.md`](../benchmarks/QUALITY_GATE_VQ3_64.md) and [`benchmarks/FRACTAL_MOE_QUALITY_GATE.md`](../benchmarks/FRACTAL_MOE_QUALITY_GATE.md)):

| Metric / Suite ([`benchmarks/eval_quality_and_tools.py`](../benchmarks/eval_quality_and_tools.py)) | `Baseline-Q4_0` (`17.59 GiB`) | `Fractal-MXFP4` (`15.95 GiB`) | `Fractal-64-BF16` (`14.36 GiB`) | **`LUT-VQ3-64` (`13.30 / 13.08 GiB`)** |
| :--- | :---: | :---: | :---: | :---: |
| **1. Agentic Tool-Calling (`30` pts — 5 multi-tool JSON calls)** | `30.0 / 30.0` | `30.0 / 30.0` | `30.0 / 30.0` | **`30.0 / 30.0`** |
| **2. Executable Python Code Gen (`30` pts — sandbox-tested)** | `30.0 / 30.0` | `30.0 / 30.0` | `30.0 / 30.0` | **`30.0 / 30.0`** |
| **3. Long-Context Needle (`8.5K`) & Symbolic Math (`20` pts)** | `20.0 / 20.0` | `20.0 / 20.0` | `20.0 / 20.0` | **`20.0 / 20.0`** |
| **Single-Turn Decode Throughput (`Stockfish-PVS` E2E)** | `19.54 tok/s` | **`26.08 tok/s`** | **`25.02 tok/s`** | **`24.18 tok/s` (`+23.7%` vs `Q4_0`)** |
| **Multi-Turn In-Context Refactoring (`Stockfish-PVS M=7`)** | `40.91 tok/s` | **`48.39–60.40 tok/s`** | **`48.39–60.40 tok/s`** | **`48.39–60.40 tok/s`** |
| **Peak Resident RAM (`VmRSS` @ `256K` Context)** | `21.71 GiB` | `18.86 GiB` | `17.27 GiB` | **`16.04 GiB` (`-5.67 GiB` vs `Q4_0`)** |

### 3.1 Empirical Paired KL-Divergence & Logit Fidelity (`Q4_0` Reference Baseline)

Independent paired evaluation conducted by `@khairullinr` on an Apple M4 Max (`36 GB` LPDDR5X-8533, Metal) using `llama-perplexity --kl-divergence` on a Python source-code corpus (`12` chunks × `2048` ctx, `~12.3k` scored tokens, `-ngl 99 -fa 1 -b 256 -ub 256`, [Issue #16](https://github.com/synchopate/llama-modchip/issues/16#issuecomment-6752457578)):

- **Reference Model**: `NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0.gguf` (`17.59 GiB`, nominal `4.50 bpw`). Note this measures fidelity relative to `Q4_0`, not unquantized `BF16`.
- **Target Quantization**: `Fractal-64-BF16` (`14.36 GiB`, nominal `3.75 bpw`, `-18.42%` file footprint).
- **Top-1 (Argmax) Token Agreement**: **`94.347 ± 0.208%`** (reported as `Same top p` by `llama-perplexity`). In ~19 out of 20 token positions, `Fractal-64` selects the exact same greedy token as the `Q4_0` baseline.
- **Kullback-Leibler Divergence (KLD)**:
  - **Mean KLD**: **`0.056606 ± 0.001964 nats`**
  - **90th Percentile KLD**: **`0.139480 nats`**
  - **95th Percentile KLD**: **`0.280968 nats`**
  - **99th Percentile KLD**: **`0.845352 nats`**
  - **Median KLD**: **`0.000829 nats`**
  - *Distribution profile*: On the vast majority of tokens, predicted logit distributions are nearly identical (median KLD `< 0.001 nats`), with divergence concentrated in a tail of roughly 5–10% of positions.
- **Per-Token Log-Likelihood Correlation**: **`97.28%`** (`Cor(ln PPL(Q), ln PPL(base))`).
- **Paired Perplexity & Probability Shifts**:
  - **Mean PPL(base)**: `2.082848 ± 0.037092`
  - **Mean PPL(Q)**: `2.123941 ± 0.039133`
  - **Mean Paired $\ln(\text{PPL}(Q)/\text{PPL}(\text{base}))$**: `0.019537 ± 0.004266` (**`+1.97%`**)
  - **Mean Paired $\text{PPL}(Q) - \text{PPL}(\text{base})$**: `0.041092 ± 0.009110`
  - **Median $\Delta p$**: **`-0.000%`** (no systematic probability bias)
  - **RMS $\Delta p$**: **`8.133 ± 0.197%`**
- *Methodological Caveats*: Single Python code corpus (low entropy compared to open prose); tokens within context windows are correlated, so asymptotic variance may exceed i.i.d. estimates.

---

## 4. Quick Commands for Nemotron

```bash
# 1. Default 1-command launch (Linux / macOS / Windows Git Bash)
./modchip chat

# 2. Build LUT-VQ3-64 (13.30 GiB Hybrid from Fractal-64 + BF16), Fractal-64 (14.36 GiB), or Fractal-MXFP4 (15.95 GiB)
./modchip quantize --vq3-nemotron  --input /path/to/Fractal-64.gguf --bf16 /path/to/bf16.gguf --output /path/to/VQ3-64.gguf
./modchip quantize --profile fractal64 --input /path/to/bf16.gguf --output /path/to/Fractal-64.gguf
./modchip quantize --profile mxfp4     --input /path/to/q4_0.gguf --output /path/to/Fractal-MXFP4.gguf

# 3. Run the 100-Point Quality & Tool-Calling Gate
./modchip eval
```
