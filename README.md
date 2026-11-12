<div align="center">

# llama-modchip (v2.5.0)

### 3.00–3.75 bpw Vector & Dyadic CPU Inference for 284B & 30B MoE Models
**Continuous PPL/KLD + 100-Pt Functional Gate · Zero GPUs Required · Linux, macOS Metal & Windows 11**

[![Release](https://img.shields.io/badge/release-v2.5.0-3fb950?style=flat-square)](RELEASE_NOTES.md)
[![License](https://img.shields.io/badge/license-Apache%202.0-58a6ff?style=flat-square)](LICENSE)
[![Quality Gate VQ3](https://img.shields.io/badge/quality%20gate-100%2F100%20PASSED-2ea043?style=flat-square)](benchmarks/QUALITY_GATE_VQ3_64.md)
[![Technical Paper PDF](https://img.shields.io/badge/PDF-llama--modchip--paper.pdf-b31b1b?style=flat-square)](paper/llama-modchip-paper.pdf)
[![Architecture](https://img.shields.io/badge/docs-ARCHITECTURE.md-bc8cff?style=flat-square)](ARCHITECTURE.md)
[![Drafter](https://img.shields.io/badge/drafter-Stockfish--PVS%20v2%20(192KB%20L2)-d29922?style=flat-square)](speculative/ROUTECRACK_TECHNIQUES.md)
[![Submit Benchmark](https://img.shields.io/badge/leaderboard-Submit%20Your%20CPU-2ea043?style=flat-square)](https://github.com/synchopate/llama-modchip/issues/new?template=cpu_benchmark_submission.yml)

[**Quickstart**](#quickstart) · [**Signal Fidelity & Benchmarks**](#signal-fidelity-and-end-to-end-benchmarks) · [**How 3D VQ & Fractal-64 Work**](#how-lut-vq3-64-and-fractal-64-work) · [**Quantization Specs**](#quantization-and-memory-breakdown) · [**Supported Models**](#supported-models-and-model-specific-optimizations) · [**Leaderboard**](#community-cpu-and-hybrid-leaderboard) · [**Technical Paper (`paper/`)**](paper/llama-modchip-paper.pdf) · [**FAQ**](FAQ.md)

<table>
<tr>
<td align="center"><b>100 / 100</b><br><sub>Quality Gate (parity w/ BF16)</sub></td>
<td align="center"><b>99.15% / 97.78%</b><br><sub>CosSim vs BF16 (Fractal / VQ3)</sub></td>
<td align="center"><b>56.93 / 56.40 GiB</b><br><sub>DeepSeek-V4 284B (Default / VQ3)</sub></td>
<td align="center"><b>13.30 GiB</b><br><sub>Nemotron-30B VQ3 GGUF</sub></td>
<td align="center"><b>24.2–111.6 tok/s</b><br><sub>30B MoE CPU decode</sub></td>
<td align="center"><b>10.4 ms</b><br><sub>/dev/shm Claude 40-tool restore</sub></td>
<td align="center"><b>0</b><br><sub>GPUs required</sub></td>
</tr>
</table>

<table>
<tr>
<th align="left">Hardware Class</th>
<th align="right">System RAM</th>
<th align="left">Target Checkpoint</th>
<th align="right">Throughput (0 GPUs)</th>
<th align="left">Architecture & Memory Notes</th>
</tr>
<tr>
<td align="left">Developer laptop / Mini-PC</td>
<td align="right">16–32 GB</td>
<td align="left"><b>Nemotron-3.5-30B-A3B-VQ3</b> (<code>13.30 GiB</code>, <code>3.00/3.75 bpw</code>)</td>
<td align="right"><b>24.2–60.4 tok/s</b></td>
<td>16.0–16.3 GiB VmRSS; 70.19% Stockfish-PVS acceptance; 100/100 Quality Gate</td>
</tr>
<tr>
<td align="left">64 GB Desktop / Server</td>
<td align="right">64 GB</td>
<td align="left"><b>DeepSeek-V4-Flash 284B Abliterated-Fractal</b> (<code>56.93 GiB</code>, Default) / <b>VQ3</b> (<code>56.40 GiB</code>)</td>
<td align="right"><b>41.1 cold / 44.4 warm pp · 8.8–9.5 N=1 / 19.4–25.3 spec tg</b></td>
<td>58.2–59.1 GiB VmRSS in ~62.5 GiB usable RAM (<code>-c 32768</code> desktop safe default; <code>-c 262144</code> headless 64GB / 96GB+)</td>
</tr>
<tr>
<td align="left">Hybrid Desktop (CPU + 16 GB GPU)</td>
<td align="right">32–64 GB</td>
<td align="left"><b>Nemotron-30B / Qwen3-30B</b> (<code>3.75–4.25 bpw</code>)</td>
<td align="right"><b>80.80 tok/s</b></td>
<td>Hot attention/SSM trunk on GPU (<code>-ngl 99</code>), cold routed MoE experts in CPU RAM</td>
</tr>
<tr>
<td align="left">8-Channel Workstation (Threadripper)</td>
<td align="right">128 GB+</td>
<td align="left"><b>DeepSeek-284B / Nemotron-30B</b></td>
<td align="right"><b>111.62 tok/s</b> (30B)</td>
<td>8-CCD AVX-512BW <code>_1x8</code> register tiling saturates 8-channel DDR5 bandwidth</td>
</tr>
</table>

</div>

## Overview

Mixture-of-Experts (MoE) inference on CPU is bounded primarily by DRAM bandwidth per generated token. Standard 4.50-bpw scalar quantization (`Q4_0`) leaves significant memory-bandwidth headroom unused, while conventional 2–3 bit scalar K-quants (`Q2_K`, `Q3_K`) require multi-shift bit unpacking and fall back to `Q4_0` whenever expert matrix dimensions are not multiples of 256 (such as `1856 × 2688` in `Nemotron-3.5-30B-A3B`).

`llama-modchip v2.5.0` introduces two block-64 quantization formats engineered for single-instruction SIMD register lookup—**`Fractal-64` (`GGML_TYPE_Q2_0 = 42`, `30 bytes / 64 weights = 3.75 bpw`)** and **`LUT-VQ3-64` (`GGML_TYPE_VQ3_0 = 43`, `24 bytes / 64 weights = 3.00 bpw`)**—paired with cache-aware KV management (`HARP-Cache`), `/dev/shm` Slot-0 prefix snapshots, and `RouteCrack + Stockfish-PVS v2` speculative decoding:

1. **Domain-Aware `LUT-VQ3-64 + Fractal-64` Hybrid Quantization (`100/100` Quality Gate):** On uncompressed `BF16` MoE weights (`blk.1.ffn_up_exps.weight`), `Fractal-64` (`3.75 bpw`) reaches **`17.75 dB` SQNR (`99.15%` Cosine Similarity, `+1.45 dB` over `Q3_K`)**, while `LUT-VQ3-64` (`3.00 bpw`) reaches **`13.57 dB` SQNR (`97.78%` Cosine Similarity, `+3.42 dB` over `Q2_K`)** via 128 3D Lloyd-Max centroids decoded in registers with AVX-512 VBMI `VPERMI2B`. Keeping boundary layers and residual output projections (`ffn_down_exps`, MLA attention) in `Fractal-64` while encoding interior expansion projections in `LUT-VQ3-64` cuts `up_exps` footprint by `33.3%` vs. `Q4_0` with zero regression on [`benchmarks/eval_quality_and_tools.py`](benchmarks/eval_quality_and_tools.py) (`100.0 / 100.0`).
2. **`DeepSeek-V4-Flash 284B` (`13.17B` Active) Inside `64 GB` RAM:** Three 3-shard checkpoints fit cleanly inside `64 GB` DDR5 RAM without swap: the **default DeepSeek checkpoint** [`DeepSeek-V4-Flash-0731-Abliterated-Fractal`](https://huggingface.co/Synchopate/DeepSeek-V4-Flash-0731-Abliterated-Fractal-64GB-CPU-GGUF) (`56.93 GiB` disk / `59.13 GiB` `VmRSS`, single-pass `BF16/FP8 -> Q6_K` on all 130 dense bottlenecks + `Fractal-64` MLA, achieving **`1.7597` `PPL`**, **`70.0 / 80` Functional Gate**, and **`50.0 / 60` Hard-60**), alongside standard safety-aligned [`DeepSeek-V4-Flash-0731-VQ3`](https://huggingface.co/Synchopate/DeepSeek-V4-Flash-0731-VQ3-64GB-CPU-GGUF) (`56.40 GiB` Hybrid; `56.07 GiB` / `58.31 GiB` `VmRSS` in full-VQ3 ablation at **`9.05 tok/s` `N=1`**, **`41.11 tok/s` cold prefill**, **`19.4–25.3 tok/s` agentic bursts**) and [`DeepSeek-V4-Flash-0731-Fractal`](https://huggingface.co/Synchopate/DeepSeek-V4-Flash-0731-Fractal-64GB-CPU-GGUF) (`56.57 GiB`).
3. **`Nemotron-3.5-30B-A3B` (`3.0B` Active) in `16 GiB` RAM:** Shipped `NVIDIA-Nemotron-3.5-30B-A3B-VQ3-64` (`13.30 GiB` Hybrid; `13.08 GiB` / `16.04 GiB` `VmRSS` in full 23-layer `up_exps` VQ3) saves up to **`-5.67 GiB` RAM vs. `Q4_0`**, runs at **`24.18 tok/s` E2E (`+23.7%` vs. `Q4_0`)**, achieves **`70.19%` `Stockfish-PVS` draft acceptance**, and scores **`100.0 / 100.0`** on the Quality & Tool-Calling Gate.
4. **Sub-20ms `/dev/shm` Slot-0 Restore + Cross-Platform Native Support:** Restores the 40-tool Claude Code prefix (`2,776` tokens) from `/dev/shm` in **`10.4 ms` (`0.16s` TTFT)** on Linux, runs on Apple Silicon Metal (`57.65 tok/s` M1 Max, `78.60 tok/s` M3 Max, `27.0 tok/s` M5 Max `284B`), and compiles natively on Windows 11 (`MinGW-w64 UCRT64`, verified up to `1M` context).

<div align="center">

<img src="assets/launch_284b_64gb.gif" alt="llama-modchip v2.5.0 — 284.33B DeepSeek-V4-Flash & Nemotron-30B on Pure CPU" width="100%" />

</div>

---

## Signal Fidelity and End-to-End Benchmarks

### 1. Direct Signal Fidelity on Uncompressed `BF16` MoE Weights
Measured via `quantize_row_*_ref` $\to$ `dequantize_row_*` on the uncompressed `59.0 GiB BF16` master checkpoint (`blk.1.ffn_up_exps.weight`, $N = 4,194,304$ weights, kurtosis $\kappa = 3.2–4.8$):

| Quantization Format | Block Layout & Size | Effective `bpw` | Direct `BF16` SQNR | Cosine Sim vs. `BF16` | Relative $L_2$ Error | SIMD Decode Path |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`BF16` (Reference)** | `128 B / 64 weights` | `16.000 bpw` | **$\infty\text{ dB}$** | **`100.00%`** | **`0.00%`** | Native F32/BF16 Dot |
| **`Baseline-Q4_0`** | `36 B / 64 weights` | `4.500 bpw` | **`21.09 dB`** | **`99.61%`** | **`8.82%`** | Scalar Uniform AVX2/AVX-512 |
| **`Fractal-64` (`GGML_TYPE_Q2_0 = 42`)** | **`30 B / 64 weights`** | **`3.750 bpw`** | **`17.75 dB`** *(+1.45 dB vs Q3_K)* | **`99.15%`** | **`12.96%`** | **Single-Cycle Dyadic `VPSHUFB` / Metal** |
| **Standard `Q3_K`** | `27.5 B / 64 weights` | `3.438 bpw` | `16.30 dB` | `98.84%` | `15.32%` | Multi-Shift `Q3_K` Unpack |
| **`LUT-VQ3-64` (`GGML_TYPE_VQ3_0 = 43`)** | **`24 B / 64 weights`** | **`3.000 bpw`** | **`13.57 dB`** *(+3.42 dB vs Q2_K)* | **`97.78%`** | **`20.97%`** | **Single-Cycle 3D VQ `VPERMI2B`** |
| **Standard `Q2_K`** | `21 B / 64 weights` | `2.625 bpw` | `10.15 dB` | `95.37%` | `31.08%` | Multi-Shift `Q2_K` Unpack |

---

### 2. End-to-End Quality & Speed Ablations ([`benchmarks/eval_quality_and_tools.py`](benchmarks/eval_quality_and_tools.py) + `llama-bench`)
*Measured on 16-Core AMD Zen 5 (`--cpu-range 0-15 --cpu-strict 1`, Transparent HugePages enabled; full methodology and shipped-hybrid vs. full-VQ3 ablation breakdown in [`benchmarks/QUALITY_GATE_VQ3_64.md`](benchmarks/QUALITY_GATE_VQ3_64.md)).*

#### A. `NVIDIA-Nemotron-3.5-30B-A3B` (23-Layer `up_exps` VQ3 Ablation vs. Baselines)
| Metric / Suite | `Reference BF16` | `Standard Q4_0` | `Fractal-64` (`v2.4.0`) | **`LUT-VQ3-64` (`v2.5.0`)** | Delta vs. `Q4_0` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **GGUF File Size on Disk** | `59.00 GiB` | `19.86 GiB` *(`17.59` pure Q4)* | `14.36 GiB` | **`13.08 GiB`** *(`13.30 GiB` shipped Hybrid)* | **`-6.78 GiB` (`-34.1%`)** |
| **Resident RAM (`VmRSS`)** | `59.40 GiB` | `21.71 GiB` | `17.27 GiB` | **`16.04 GiB`** | **`-5.67 GiB` (`-26.1%`)** |
| **Raw Pure Decode (`tg128`, $N=1$)** | `7.82 tok/s` | `19.54 tok/s` | `22.25 tok/s` | **`22.44 tok/s`** | **`+14.8%`** |
| **E2E Decode (`Stockfish-PVS`)** | `7.82 tok/s` | `19.54 tok/s` | `25.02 tok/s` | **`24.18 tok/s`** | **`+23.7%`** |
| **Stockfish-PVS Spec Acceptance** | `62.50%` | `66.67%` | `59.09%` | **`70.19%`** | **`+3.52%` (`+11.10%` vs Fractal)** |
| **1. Agentic Tool-Calling (`30` pts)** | `30.0 / 30.0` | `30.0 / 30.0` | `30.0 / 30.0` | **`30.0 / 30.0`** | **`100.0%` Schema Match** |
| **2. Executable Code Gen (`30` pts)** | `30.0 / 30.0` | `30.0 / 30.0` | `30.0 / 30.0` | **`30.0 / 30.0`** | **`100.0%` Unit-Test Pass** |
| **3. Long-Context Needle & Math (`20` pts)** | `20.0 / 20.0` | `20.0 / 20.0` | `20.0 / 20.0` | **`20.0 / 20.0`** | **`100.0%` Exact Retrieval** |
| **Total Quality Gate Score** | **`100.0 / 100.0`** | **`100.0 / 100.0`** | **`100.0 / 100.0`** | **`100.0 / 100.0`** | **Verified Parity** |

#### B. `DeepSeek-V4-Flash-0731 284B` (`284.33B` Total / `13.17B` Active, Unpruned Top-6 Routing)
| Metric / Suite | `Reference BF16` | `Fractal-64` (`Aligned`) | `LUT-VQ3-64` (`Aligned`) | **`Abliterated-Fractal` (Default DeepSeek)** | Notes |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **GGUF File Size on Disk** | `568.6 GiB` | `56.57 GiB` | `56.40 GiB` *(`56.07` abl.)* | **`56.93 GiB`** (`61.13 GB`) | **3 shards; fits 64 GB RAM** |
| **Peak Resident RAM (`VmRSS`, `-c 32768`)** | *OOM on 64GB* | `58.20 GiB` | `58.31 GiB` | **`59.13 GiB`** | **Leaves ~3.4–4.2 GiB OS Headroom on 64 GB** |
| **Cold Prompt Prefill (`pp128`, 0 cached)** | *N/A* | `37.79 tok/s` | `41.11 tok/s` *(`44.36` warm)* | **`38.0–43.2 tok/s`** *(`44.36` warm)* | **4-level AVX-512 `{8,4,3,2}` tiling** |
| **Raw Pure Decode (`tg64`, $N=1$, Unpruned Top-6)** | *N/A* | `7.74–9.55 tok/s` | `9.05 tok/s` | **`8.75–9.50 tok/s`** | **`10.1–10.7 tok/s` w/ optional `Top-4 L2`** |
| **Speculative / Agentic Burst (`Stockfish-PVS`)** | *N/A* | `11.5–21.2 tok/s` | `19.4–25.3 tok/s` | **`19.0–25.0 tok/s`** | **Sub-20ms `/dev/shm` Slot-0 restore** |
| **Continuous `llama-perplexity` (`4,096` tok)** | *N/A* | `1.8284 ± 0.0704` | `1.8258 ± 0.0698` | **`1.7597 ± 0.0653`** *(`-3.76%`)* | **Direct single-pass `BF16/FP8 -> Q6_K`** |
| **Functional Gate (`80` pts) & `Hard-60` (`60` pts)** | `100.0%` | `64/80` · `20/60` | `66/80` · `40/60` | **`70.0 / 80` (`30/30` Tools) · `50.0 / 60`** | **Reproducible via [`eval_perplexity_and_kld.py`](benchmarks/eval_perplexity_and_kld.py)** |

---

## How `LUT-VQ3-64` and `Fractal-64` Work

### 1. 3D Sphere Packing (`LUT-VQ3-64`, `GGML_TYPE_VQ3_0 = 43`, `24 B / 64 W`)
Scalar uniform quantizers quantize each coordinate $w_i \in [-M, +M]$ independently across the 3D Cartesian bounding cube $[-M, +M]^3$ of volume $8M^3$. Heavy-tailed MoE weights ($\kappa = 3.2–4.8$) have a radial joint density $p(\mathbf{w}) \propto \exp(-\|\mathbf{w}\|_p^\alpha / \sigma^\alpha)$ concentrated inside the inscribed $L_2$ sphere ($\text{Vol} \approx 4.189 M^3$, leaving $47.64\%$ of the cube empty) or $L_1$ cross-polytope ($\text{Vol} \approx 1.333 M^3$, leaving $83.33\%$ empty).

[`LUT-VQ3-64`](vq3/lut_vq3_quantizer.py) places `128` 3D Lloyd-Max centroids ($7\text{ bits} / 3\text{ weights} = 2.333\text{ bpw}$, antipodally symmetric $\mathbf{c}_{127-k} = -\mathbf{c}_k$) directly on the populated spherical/cross-polytope shell inside a `24-byte` `block_vq3_0` struct ([`patches/vq3_codebook.h`](patches/vq3_codebook.h)):
- **`2 bytes` (`d`)**: `FP16` super-block scale (scaled by `GGML_VQ3_G = 8.6373016238e-3f`).
- **`22 bytes` (`qs[22]`, `176 bits`)**:
  - `qs[0..19]` bits `0..6` (`140 bits`): `20` 7-bit 3D indices covering `60` weights at stride 22 (`j, j+22, j+44`).
  - `qs[20..21]` bits `0..5` (`12 bits`): `2` 6-bit 2D tail indices covering `4` weights (`20, 42` and `21, 43`).
  - `s24` (`24 bits` packed in bit 7 of `qs[0..21]` and bit 6 of `qs[20..21]`): Eight **3-bit unsigned sub-scales** ($m_b = 2 + u_b \in \{2..9\}$) across the eight 8-weight sub-blocks (**`24 bytes / 64 weights = 3.00 bpw`**).

### 2. Register-Resident AVX-512 VBMI `VPERMI2B` Lookup + Domain-Aware Layer Split
The 128-entry 3D codebook occupies $128 \times 3 = 384\text{ bytes}$, fitting inside **6 AVX-512 ZMM registers** (`cb0_lo/hi`, `cb1_lo/hi`, `cb2_lo/hi`). AVX-512 VBMI `VPERMI2B` (`_mm512_permutex2var_epi8`) resolves 64 byte lookups per instruction directly in registers with zero L1/L2 cache reads. In the shipped Hugging Face checkpoints, interior pre-activation expansion tensors (`ffn_up_exps` in Nemotron, interior `ffn_gate_shexp` & `ffn_up_shexp` in DeepSeek-V4) use `LUT-VQ3-64` (`3.00 bpw`), while boundary layers, `ffn_down_*` residual projections, and MLA attention tensors remain in `Fractal-64` (`GGML_TYPE_Q2_0 = 42`, `3.75 bpw`, `30 B / 64 W`: `2B FP16 d` + `4B` `8×4-bit` sub-scales `{4..19}` + `24B` 3-bit indices into `{-14,-8,-4,-1,+1,+4,+8,+14}`).

---

## Quantization and Memory Breakdown

Exact bit-widths (`bpw`), base checkpoints, and tensor allocations for each published GGUF:

| Checkpoint | Total / Active Params | Base Source | Disk Size (`GiB` / `GB`) | Effective `bpw` | Layer-by-Layer Tensor Allocation |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`Nemotron-3.5-30B-A3B-Q4_0`** | `30.2B` / `3.0B` | BF16 Official | `17.59 GiB` (`18.89 GB`) | **`4.50 bpw`** | Standard `Q4_0` across MoE + attention; `F32` norms & SSM conv/decay |
| **`Nemotron-3.5-30B-A3B-MXFP4`** | `30.2B` / `3.0B` | `Q4_0` / BF16 | `15.95 GiB` (`17.13 GB`) | **`4.25 bpw`** | `GGML_TYPE_MXFP4` (`E2M1` + shared exponent) on routed MoE `up_exps`; `Q4_0`/`Q8_0` trunk |
| **`Nemotron-3.5-30B-A3B-Fractal-64`** | `30.2B` / `3.0B` | BF16 Official | `14.36 GiB` (`15.42 GB`) | **`3.75 bpw`** | Attention, Mamba-2 SSM & shared experts in `Q4_0`/`Q8_0`/`F32`; all 23 routed MoE layers in `Fractal-64` (`Q2_0 = 42`, 30B/64W). Scores `100/100` on [`benchmarks/eval_quality_and_tools.py`](benchmarks/eval_quality_and_tools.py). |
| **`Nemotron-3.5-30B-A3B-VQ3-64`** | `30.2B` / `3.0B` | BF16 Official | **`13.30 GiB` (`14.28 GB`)** | **`3.00 / 3.75 bpw`** | **Shipped Hybrid ([`scripts/build_nemotron_vq3.py`](scripts/build_nemotron_vq3.py)):** Boundary MoE layers (`blk.1, 3, 49, 51`) + all 23 `blk.*.ffn_down_exps.weight` in `Fractal-64` (`3.75 bpw`); 19 interior `blk.6..47.ffn_up_exps.weight` encoded from `BF16` into `LUT-VQ3-64` (`VQ3_0 = 43`, `3.00 bpw`). Pass `--all-up-exps` for the `13.08 GiB` 23-layer ablation (`100/100` Gate). |
| **`DeepSeek-V4-Flash-0731-Fractal`** | `284.33B` / `13.17B` | `DeepSeek-V4-Flash` | `56.57 GiB` (`60.74 GB`) | **`1.70 bpw`** | **Standard Aligned:** 100% MLA attention & compressor projections in `Fractal-64` (`Q2_0 = 42`, `3.75 bpw`); shared experts in `Q5_K`/`Q6_K`; 256 routed MoE experts in `SwiGLU-H2` (`IQ1_S`/`IQ1_M`/`IQ2_XXS`). |
| **`DeepSeek-V4-Flash-0731-VQ3`** | `284.33B` / `13.17B` | `DeepSeek-V4-Flash` | `56.40 GiB` (`60.56 GB`) | **`1.70 bpw`** | **Standard Aligned Hybrid ([`scripts/build_dsv4_vq3.py`](scripts/build_dsv4_vq3.py)):** 100% of MLA attention preserved in `Fractal-64` (`3.75 bpw`); boundary shared experts (`blk.0..4, 39..42`) & all `ffn_down_shexp` in `Q5_K`/`Q6_K`; 68 interior `blk.5..38` `ffn_gate_shexp` & `ffn_up_shexp` in `LUT-VQ3-64` (`3.00 bpw`); 256 routed experts in `SwiGLU-H2`. Pass `--aggressive` for the `56.07 GiB` 255-tensor ablation (`+16.9%` raw decode). |
| **[`DeepSeek-V4-Flash-0731-Abliterated-Fractal`](https://huggingface.co/Synchopate/DeepSeek-V4-Flash-0731-Abliterated-Fractal-64GB-CPU-GGUF)** | `284.33B` / `13.17B` | `DeepSeek-V4-Flash-Abliterated` | **`56.93 GiB` (`61.13 GB`)** | **`1.71 bpw`** | **Default DeepSeek Checkpoint (Single-Pass Max-Quality, Uncensored):** All `130` dense bottleneck tensors (`token_embd`, `output`, `43× attn_q_a`, `43× ffn_{gate,up,down}_shexp`) quantized single-pass `BF16/FP8 -> Q6_K`; 100% MLA attention in `Fractal-64` (`3.75 bpw`); 256 routed experts in `SwiGLU-H2`. Reaches **`1.7597` `PPL`**, **`70.0 / 80` Gate**, and **`50.0 / 60` Hard-60**. |

> **Note on MoE Expert Routing (`Unpruned Top-6` vs. Runtime Tail Pruning):** By default (`Unpruned Top-6`), `DeepSeek-V4-Flash` evaluates **all 6 routed experts + 1 shared expert (`Top-6`)** per layer with zero runtime expert dropping relative to the loaded GGUF (`1.00000` logit cosine similarity vs. unpruned `Top-6` execution on the same quantized checkpoint; weight quantization fidelity itself is measured separately via `BF16` SQNR and continuous `llama-perplexity` / Logit `KLD` in [`benchmarks/QUALITY_GATE_VQ3_64.md`](benchmarks/QUALITY_GATE_VQ3_64.md)). Runtime $L_2$-quadrature `Top-5` / `Top-4` pruning via `/tmp/dsv4_moe_cfg.bin` (`%TEMP%\dsv4_moe_cfg.bin` on Windows) is optional.
>
> **Why `DeepSeek-V4-Flash-0731` (and not `V4.1-Flash-1017`) for 64 GB Resident CPU Inference:** `DeepSeek-V4.1-Flash-1017` (`DeepseekV41ForCausalLM`, `510.3 GB` source across `384` routed experts/layer) adds **`188.83 GiB` of native `FP8` Engram lookup tables** on top of **`151.77 GiB`** of `Q2` model weights (**`340.60 GiB` total even at 2-bit quantization**), requiring `128 GB+` RAM with NVMe SSD table streaming or `512 GB` RAM for resident execution. `DeepSeek-V4-Flash-0731` (`284.33B` total / `13.17B` active, zero Engram tables) is the largest DeepSeek-V4 Flash checkpoint that runs **100% resident inside 64 GB RAM** with zero disk streaming.

---

## Quickstart

### Linux & macOS (Automated Setup & Chat)
```bash
git clone https://github.com/synchopate/llama-modchip.git
cd llama-modchip
./modchip chat
```

### Direct Run (Pre-Compiled Linux `x86_64` `-avx512` / `-avx2` Binaries + Hugging Face `.gguf`)
> **Compatibility Note (`GGML_TYPE_Q2_0 = 42` & `GGML_TYPE_VQ3_0 = 43`):** `Fractal-64` (`type 42`) and `LUT-VQ3-64` (`type 43`) checkpoints use custom block-64 tensor types and `deepseek4` / `Stockfish-PVS` kernels. Unpatched `llama.cpp`, Ollama, vLLM, or LM Studio builds will fail to load them with `unknown tensor type 42` or `43`.
> - **ISA-Matched Precompiled Linux `x86_64` Binaries ([GitHub Release `v2.5.0`](https://github.com/synchopate/llama-modchip/releases/tag/v2.5.0)):** Run `./modchip download-bin` (or `./install.sh --prebuilt`) to auto-detect CPUID flags, or pick:
>   - `llama-server-modchip-linux-x86_64-avx512`: AMD Zen 4/5 (`Ryzen 7000/8000/9000`, `EPYC Genoa/Turin`, `Threadripper 7000`) & Intel Ice Lake / Sapphire Rapids (`avx512f` + `avx512_vbmi`).
>   - `llama-server-modchip-linux-x86_64-avx2`: Pre-AVX512 `x86_64` CPUs (`x86-64-v3`: Intel Haswell/Broadwell/Skylake/Alder/Raptor Lake, AMD Zen 1/2/3 — compiled with `-mno-avx512f`, `0` `%zmm` instructions, preventing `SIGILL`). The unsuffixed `llama-server-modchip-linux-x86_64` asset also points to this safe `x86-64-v3` build.

```bash
# Auto-select -avx512 or -avx2 binary for your Linux x86_64 CPU:
BIN_ISA=$(grep -q "avx512_vbmi" /proc/cpuinfo && echo "avx512" || echo "avx2")
curl -fLO "https://github.com/synchopate/llama-modchip/releases/download/v2.5.0/llama-server-modchip-linux-x86_64-${BIN_ISA}"
chmod +x "llama-server-modchip-linux-x86_64-${BIN_ISA}"

# 1A. 30B Nemotron-3.5-A3B Hybrid VQ3-64 + Fractal-64 (13.30 GiB — 16 GB+ RAM at 22.4 N=1 / 24.2–60.4 spec tok/s):
hf download Synchopate/NVIDIA-Nemotron-3.5-30B-A3B-VQ3-64-CPU-GGUF \
  NVIDIA-Nemotron-3.5-30B-A3B-VQ3-64.gguf --local-dir .
./llama-server-modchip-linux-x86_64-${BIN_ISA} -m NVIDIA-Nemotron-3.5-30B-A3B-VQ3-64.gguf -c 262144 -t 16 --port 8080

# 1B. 30B Nemotron-3.5-A3B Fractal-64 (14.36 GiB — recommended on AVX2-only CPUs & Apple Silicon):
hf download Synchopate/NVIDIA-Nemotron-3.5-30B-A3B-Fractal-64-CPU-GGUF \
  NVIDIA-Nemotron-3.5-30B-A3B-Fractal-64.gguf --local-dir .

# 2A. Default 284B DeepSeek-V4-Flash-0731-Abliterated-Fractal (56.93 GiB — 64 GB RAM, Uncensored / Single-Pass Q6_K + Fractal-64):
hf download Synchopate/DeepSeek-V4-Flash-0731-Abliterated-Fractal-64GB-CPU-GGUF \
  --include "DeepSeek-V4-Flash-0731-Abliterated-Fractal-*.gguf" --local-dir ./dsv4-ablit
# Safe 64 GB Desktop default (-c 32768 -> ~59.1 GiB VmRSS, preserving ~3.4-4.2 GiB OS/GUI headroom):
./llama-server-modchip-linux-x86_64-${BIN_ISA} -m ./dsv4-ablit/DeepSeek-V4-Flash-0731-Abliterated-Fractal-00001-of-00003.gguf -c 32768 -t 16 -fa on --port 8080
# Dedicated Headless 64 GB Server / 96 GB+ Workstation (full 256K MLA context):
# ./llama-server-modchip-linux-x86_64-${BIN_ISA} -m ./dsv4-ablit/DeepSeek-V4-Flash-0731-Abliterated-Fractal-00001-of-00003.gguf -c 262144 -t 16 -ctk q8_0 -ctv q8_0 -fa on --port 8080

# 2B. Standard Safety-Aligned 284B DeepSeek-V4-Flash VQ3 (56.40 GiB) or Fractal-64 (56.57 GiB):
hf download Synchopate/DeepSeek-V4-Flash-0731-VQ3-64GB-CPU-GGUF \
  --include "DeepSeek-V4-Flash-0731-VQ3-*.gguf" --local-dir ./dsv4-vq3
hf download Synchopate/DeepSeek-V4-Flash-0731-Fractal-64GB-CPU-GGUF \
  --include "DeepSeek-V4-Flash-0731-Fractal-*.gguf" --local-dir ./dsv4
```

> [!CAUTION]
> **Responsible-Use Disclaimer (`DeepSeek-V4-Flash-0731-Abliterated-Fractal`):** The default 284B checkpoint derives from an abliterated base model in which refusal-mediating activation directions were removed via directional orthogonalization. Standard upstream refusal filters are disabled.
> - **Intended Scope:** Authorized security research, red-teaming, penetration testing, CTF competitions, vulnerability auditing, and local offline software engineering.
> - **Deployment:** Do **not** expose an unmoderated endpoint of the abliterated checkpoint to public or untrusted networks without an external safety filter and access control.
> - **Liability & Safety-Aligned Alternatives:** Users assume sole responsibility for compliance with applicable laws and upstream model license terms. For deployments requiring standard upstream safety alignment, select Option `2` (`--preset dsv4-vq3`, [`Synchopate/DeepSeek-V4-Flash-0731-VQ3-64GB-CPU-GGUF`](https://huggingface.co/Synchopate/DeepSeek-V4-Flash-0731-VQ3-64GB-CPU-GGUF)) or Option `3` (`--preset dsv4-fractal`, [`Synchopate/DeepSeek-V4-Flash-0731-Fractal-64GB-CPU-GGUF`](https://huggingface.co/Synchopate/DeepSeek-V4-Flash-0731-Fractal-64GB-CPU-GGUF)) during `./install.sh`.

> **64 GB Host RAM Headroom (`-c 32768` Desktop vs. `-c 262144` Headless):** On a 64 GB Linux box (`~62.5 GiB` kernel-usable RAM after firmware reservation), the 284B GGUF (`56.40–56.57 GiB`) reaches `58.2–58.4 GiB` `VmRSS` at `-c 32768`, leaving `~4.1–4.3 GiB` for the OS and desktop GUI. Using `-c 262144` (`256K` tokens; single-head MLA compressed KV cache adds `~0.8–1.2 GiB` in `q8_0`) fits on a **dedicated headless 64 GB node** (`zram`/swap recommended against transient spikes) or **96 GB+ host**, whereas active GUI/browser desktops on 64 GB should keep `-c 32768` (`start_server.sh` enforces this automatically unless `MODCHIP_CTX_SIZE=262144` is set). Measured long-context prefill/decode scaling up to `3,038` tokens (`30.7–38.4 tok/s` prefill, `9.16 tok/s` decode) is reported in [`docs/DEEPSEEK_V4_OPTIMIZATIONS.md` §6](docs/DEEPSEEK_V4_OPTIMIZATIONS.md).

### Windows 10/11 (Patched `VQ3-64` / `Fractal-64` Engine & Hybrid GPU+CPU Pack)
- **Standard GGUFs (`Q4_0`, `MXFP4`, `Q3_K_M`, `Q8_0`) with Automatic GPU+CPU Offload:** Run **[`windows/AVVIA_MODCHIP.bat`](windows/AVVIA_MODCHIP.bat)** ([`windows/README.md`](windows/README.md)). It detects physical CPU cores, free RAM, and GPU VRAM (`nvidia-smi`), downloads official `llama.cpp` CUDA 12.4 / Vulkan binaries, computes `--n-cpu-moe`, and starts `http://127.0.0.1:8080`.
- **Patched `DeepSeek-V4 284B` & `Nemotron-30B` (`Fractal-64` `type 42` & `LUT-VQ3-64` `type 43`):** Run `powershell -ExecutionPolicy Bypass -File windows\build_patched_engine.ps1 -Target dsv4` (or `-Target nemotron`) to compile the native MinGW-w64 UCRT64 AVX-512 engine, then launch via **[`windows/AVVIA_DSV4.bat`](windows/AVVIA_DSV4.bat)** (details in **[`docs/WINDOWS_PORT.md`](docs/WINDOWS_PORT.md)**).

| `./modchip` Command | Description |
| :--- | :--- |
| **`./modchip download-bin`** | Auto-detects `avx512_vbmi` vs `avx2` on Linux `x86_64` and installs the matching release binary |
| **`./modchip chat`** | Interactive terminal chat with live `tok/s`, TTFT, and `Stockfish-PVS` acceptance telemetry |
| **`./modchip web`** | Opens the Dark-Mode Web Chat & Live Telemetry HUD (`http://127.0.0.1:8787`) |
| **`./modchip bench`** | Runs a 30-second 2-stage speed test and prints a shareable CPU Benchmark Scorecard |
| **`./modchip claude`** | Launches Claude Code CLI (`40 tools`) wired to the local 256K engine (`10.4–32.6 ms` warm restore) |
| **`./modchip eval`** | Runs the reproducible 100-Point Quality, Tool-Calling, Coding & 8.5K Needle Gate |
| **`./modchip quantize`** | Transcodes GGUFs into `LUT-VQ3-64` (`--vq3-nemotron` / `--vq3-dsv4`), `Fractal-64`, or `Fractal-MXFP4` |
| **`./modchip status` / `stop`** | Inspects CPU readiness, `2MB HugePages`, and `/dev/shm` IPC, or stops background services |

### Hardware Readiness Check (Zero Model Download Required)

```bash
curl -sL https://raw.githubusercontent.com/synchopate/llama-modchip/main/scripts/check_cpu_readiness.py | python3
```

Submit your terminal scorecard via a [GitHub Benchmark Issue](https://github.com/synchopate/llama-modchip/issues/new?template=cpu_benchmark_submission.yml) or [GitHub Discussions](https://github.com/synchopate/llama-modchip/discussions) to add your CPU to the leaderboard below.

---

## Supported Models and Model-Specific Optimizations

Each model family has a dedicated technical guide with mathematical derivations, operator-level latency profiles, and quality gate benchmarks:

| Model Family | Total / Active Params | Disk (`GGUF`) & `256K` RAM (`VmRSS`) | Key Optimizations Implemented | Verified Speed (`16C Zen 5` / Hybrid) | Detailed Guide |
| :--- | :---: | :---: | :--- | :---: | :---: |
| **`NVIDIA-Nemotron-3.5-30B-A3B`**<br/>*(`VQ3-64` / `Fractal-64` / `MXFP4` / `Q4_0`)* | `30.2B` / **`3.0B`**<br/>*(`23 SSM + 23 MoE + 6 GQA`)* | **`13.30 – 17.59 GiB`**<br/>*(`15.4 – 21.7 GiB` RSS)* | • **`LUT-VQ3-64` (`3.00 bpw`) & `Fractal-64` (`3.75 bpw`)** (`100/100` Quality Gate)<br/>• **`Stockfish-PVS v2 (M=7)` + `RouteCrack`** (`192 KB` L2-resident Zobrist TT)<br/>• **`432 MB` 256K EAKV (`Q8_0/Q4_0`)** + `_MM_HINT_NTA` L3 isolation<br/>• **`18.9 ms` `/dev/shm` Slot-0 Restore** | **`24.2 – 60.4 tok/s`** (`0 GPU`)<br/>**`80.8 tok/s`** (`Vulkan Hybrid`)<br/>**`111.6 tok/s`** (`Threadripper`) | **[Nemotron Guide](docs/NEMOTRON_OPTIMIZATIONS.md)** |
| **`DeepSeek-V4-Flash-0731`**<br/>*(`Abliterated-Fractal` Default / `VQ3-64` / `Fractal-64`)* | **`284.33B`** / **`13.17B`**<br/>*(`43L MLA + 256+1s MoE`)* | **`56.40 – 56.93 GiB`**<br/>*(**`58.2 – 59.1 GiB` RSS** — `32K` desktop / `256K` headless)* | • **Single-Pass `BF16/FP8 -> Q6_K` (`1.76` PPL, `70/80` Gate, `50/60` Hard-60) + `LUT-VQ3-64` + `SwiGLU-H2` MoE + `Fractal-64` MLA**<br/>• **4-Level Register (`ZMM` AVX-512BW/VBMI) + L1D + L2 + L3 Multi-Column (`{8,4,3,2}`) Prefill Tiling (`25.0 -> 43.98 cold / 44.36 warm tok/s`)**<br/>• **Register-Stationary AVX-512 `MQA8` FlashAttention (`2.0x`) + AVX-512 `quantize_row_q8_K`**<br/>• **Optional $L_2$-Quadrature MoE Pruning + Gate-First SwiGLU (`10.74 tok/s`)** & $O(1)$ Rollback Fix | **`41.1–44.0 tok/s`** *(Cold pp)*<br/>**`44.36 tok/s`** *(Warm prefix)*<br/>**`8.75–9.55 tok/s`** *(Unpruned Top-6)*<br/>**`9.87–10.74 tok/s`** *(`Top-5/4 L2`)*<br/>**`11.5–25.3 tok/s`** *(`PVS` / Tools)* | **[DeepSeek-V4 Guide](docs/DEEPSEEK_V4_OPTIMIZATIONS.md)** |
| **`Qwen3-30B-A3B` & `Qwen3-14B`**<br/>*(`Q3_K_M` / `Q4_K_M`)* | `30.5B` / **`3.3B -> 1.7B`**<br/>*(`48L GQA`, `128` exp)* | **`13.70 – 17.28 GiB`**<br/>*(`14.5 – 19.4 GiB` RSS)* | • **Norm-Preserving Dynamic MoE Top-K (`8 -> 4`, `+38.5%` decode)**<br/>• **3:1 Interleaved SWA Safety Lock (`LLAMA_SWA_STRIDE=4`)**<br/>• **Windows `11 GB VRAM + 16 GB RAM` Hybrid Split (`--n-cpu-moe N`)** | **`34.80 tok/s`** (`0 GPU` Top-4)<br/>**`50–58 tok/s`** (`14B` RTX 2080 Ti) | **[Architecture §3.9](ARCHITECTURE.md)** |
| **`NVIDIA-Nemotron-H-8B-128K`**<br/>*(`Q8_0` / `Q4_K_M`)* | `8.0B` Dense<br/>*(`48 SSM + 4 GQA`)* | **`4.60 – 8.00 GiB`**<br/>*(`210 MB` KV @ `128K`)* | • **48-Layer Mamba-2 SSM Constant-Memory Backbone**<br/>• **100% GPU VRAM Residency on `8–11 GB` Cards (`0 GB` System RAM)** | **`100+ tok/s`** (`RTX 2080 Ti`)<br/>**`147.6 tok/s`** (`CPU pp512`) | **[Nemotron Guide](docs/NEMOTRON_OPTIMIZATIONS.md)** |

---

## Supported Chips and Processor-Specific Optimizations

Microarchitectural details and tuning flags for each processor and GPU hybrid are documented in **[`docs/HARDWARE_OPTIMIZATIONS.md`](docs/HARDWARE_OPTIMIZATIONS.md)**:

| Processor / Platform | Verified Hardware | Hardware-Specific Optimizations Active | Platform Guide & Scripts |
| :--- | :--- | :--- | :---: |
| **AMD Zen 5 (`Desktop / Mobile / Server`)** | `Ryzen 9 9955HX`, `9950X`, `Ryzen 7 9800X3D`, `EPYC 9005 Turin` | • Single-cycle 512-bit **AVX-512 (`VPDPBUSD`, `VPRORVQ`, `VBMI VPERMI2B`, `BF16`)**<br/>• **`48 KB` L1D 3D Tiling (`TILE_I0=64`)** on multi-head `wo_a` projections<br/>• **`_MM_HINT_NTA` Victim-L3 Isolation (`64 MB`)** + **`2MB` THP (`6.14 GB` L2 STLB)** | **[Hardware Guide §2.1](docs/HARDWARE_OPTIMIZATIONS.md)**<br/>[`install.sh`](install.sh) |
| **AMD Zen 4 Multi-CCD (`Threadripper / EPYC`)** | `Threadripper PRO 7985WX` (`64C/128T`, `8 CCDs`, `256 MB` L3) | • **Dynamic sysfs L3 CCD Topology Discovery** ([`scripts/llama_modchip.py`](scripts/llama_modchip.py))<br/>• **Deterministic `4C/CCD` Stride (`-t 32`)**: avoids Infinity Fabric saturation (`+54%` faster than `-t 64`) | **[Hardware Guide §2.2](docs/HARDWARE_OPTIMIZATIONS.md)**<br/>[Issue #1](https://github.com/synchopate/llama-modchip/issues/1) |
| **AMD 3D V-Cache + Radeon Vulkan Hybrid** | `Ryzen 9 7945HX3D` (`128 MB` L3) + `RX 9070 XT 16GB` (`RADV`) | • **Tensor-Role Splitting (`-ngl 99 --n-cpu-moe N --load-mode none`)**<br/>• Native **Vulkan `SSM_SCAN` + `Fractal-MXFP4` (`15.2 GB`) GPU Shaders** (`80.8 tok/s` at 256K) | **[Hardware Guide §2.3](docs/HARDWARE_OPTIMIZATIONS.md)**<br/>[`FAQ.md` Q8](FAQ.md) |
| **Intel / AMD + NVIDIA RTX (`Windows 10/11`)** | `Ryzen 7 9800X3D` (`1M` ctx), `Core i9-9900K / 14900K` + `RTX 2080 Ti / 3090 / 4090` | • **MinGW-w64 UCRT64 AVX-512 VBMI/BW Patched Engine (`build_patched_engine.ps1`)**<br/>• **Live-VRAM `--n-cpu-moe` Solver (`550 MB` WDDM Reserve)** + `_WIN32` `CreateFileMappingA` | **[Windows Port](docs/WINDOWS_PORT.md)**<br/>[`windows/`](windows/README.md) |
| **Apple Silicon (`M1 / M2 / M3 / M4 / M5`)** | `M5 Max` (`64 GB` / `48 GB`), `M4 Max` (`64 GB` / `36 GB`), `M4 Pro` (`24 GB`), `M3 Max`, `M1 Max` | • **Register-Tiled `Fractal-64` (`Q2_0`) Fast-Decode Metal Shaders ([PR #24](https://github.com/synchopate/llama-modchip/pull/24))**: `57.65 tok/s` on M1 Max (`+86%`), `78.60 tok/s` on M3 Max, `72.5 tok/s` `tg128` / `979.9 tok/s` `pp512` on M4 Max ([#16](https://github.com/synchopate/llama-modchip/issues/16))<br/>• **Zero-Copy Unified Memory + NEON `SDOT`**: `27.0 tok/s` `DeepSeek-V4 284B` on M5 Max (`286.5 GB/s` DRAM BW, [#21](https://github.com/synchopate/llama-modchip/issues/21)) | **[Apple Silicon Guide](apple-silicon/README.md)**<br/>[#8](https://github.com/synchopate/llama-modchip/issues/8) · [#16](https://github.com/synchopate/llama-modchip/issues/16) · [#24](https://github.com/synchopate/llama-modchip/pull/24) |
| **ARM64 Cloud & Edge (`AWS Graviton / Ampere`)** | `AWS Graviton3 m7g.2xlarge` (`8 vCPU Neoverse-V1`, `32 MB` L3) | • **ARMv8.4-A DotProd (`SDOT`) + 256-Bit SVE** + 1-vCPU-per-core pinning (`-t 8`)<br/>• `32 MB` System Level Cache (`SLC`) retention (`44.97 tok/s` decode) | **[Hardware Guide §1](docs/HARDWARE_OPTIMIZATIONS.md)**<br/>[Issue #2](https://github.com/synchopate/llama-modchip/issues/2) |
| **AMD Zen 4 Compact Mini-PC (`Linux`)** | `Ryzen 7 8745H / 8745HS` (`8C/16T`, `16 MB` L3 + `Radeon 780M`) | • **AVX-512 VNNI** + `16 MB` L3 window auto-tuning (`19.48–23.38 tok/s` on `96 GB` DDR5-5600)<br/>• Optional `Radeon 780M` UMA iGPU offload | **[Fedora Guide](fedora/README.md)**<br/>[Issue #9](https://github.com/synchopate/llama-modchip/issues/9) |

---

## Headline Benchmarks: Pure CPU & Hybrid Summary

### 1. `DeepSeek-V4-Flash 284B` (`13.17B` Active) on a `64 GB` Zen 5 CPU (`AMD Ryzen 9 9955HX`, `0 GPUs`)
Operator-level profiling and mathematical derivations in **[`docs/DEEPSEEK_V4_OPTIMIZATIONS.md`](docs/DEEPSEEK_V4_OPTIMIZATIONS.md)** (probabilities and KL-divergence measured relative to unpatched `Top-6` execution on the `56.57 GiB` Fractal checkpoint):

| Runtime Configuration (`DeepSeek-V4-Flash`, `32K` Desktop / `256K` Headless) | Cold / Warm Prefill | Pure Decode (`N=1`) | With `Stockfish-PVS` | **Logit Cosine Sim** | **Routing KLD vs. `Top-6`** | **Top-1 / Top-3 Match** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Unpatched `UD-IQ1_S` (`76.87 GiB` / `82.5 GB`)** | *OOM / Swap on 64 GB* | `5.91 tok/s` (`169.3 ms`) | N/A *(Rollback Bug)* | `1.00000` | `0.0000` (`0.00%`) | `100% / 100%` |
| **1. `v2.5.0` Unpruned (`Top-6` All Layers, Default, `56.40–56.57 GiB`)** | **`41.11–43.98` cold / `44.36` warm** | **`9.05 – 9.55 tok/s`** | **`11.5 – 25.3 tok/s`** | **`1.00000`** *(Ref)* | **`0.0000` (`0.00%` Ref)** | **`100% / 100%`** |
| **2. `v2.5.0` Sweet-Spot (`Top-5` on `il >= 3`, $L_2$ Quadrature)** | **`41.11–43.98` cold / `44.36` warm** | **`9.68 – 9.87 tok/s`** | **`12.2 – 25.5 tok/s`** | **`0.99714`** | **`0.0042 nats` (`+0.42%` PPL)** | **`95.8% / 100%`** |
| **3. `v2.5.0` `Top-5 L2` + Gate-First SwiGLU (`80/128` $L_2$)** | **`41.11–43.98` cold / `44.36` warm** | **`10.15 tok/s` (`98.5 ms`)** | **`12.8 – 26.0 tok/s`** | **`0.99385`** | **`0.0531 nats`** | **`100% / 100%`** |
| **4. `v2.5.0` Turbo `10+` (`Top-4 L2` + Gate-First SwiGLU)** | **`41.11–43.98` cold / `44.36` warm** | **`10.12 – 10.74 tok/s` (`93.2 ms`)** | **`13.0 – 26.4 tok/s`** | **`0.99126`** | **`0.0281 – 0.0694 nats`** | **`100% / 100%`** |

### 2. `NVIDIA-Nemotron-3.5-30B-A3B` (`3.0B` Active) at `256K` Context (`AMD Ryzen 9 9955HX`, `0 GPUs`)
Full ablation tables and `100/100` Quality Gate reports in **[`docs/NEMOTRON_OPTIMIZATIONS.md`](docs/NEMOTRON_OPTIMIZATIONS.md)** and **[`benchmarks/QUALITY_GATE_VQ3_64.md`](benchmarks/QUALITY_GATE_VQ3_64.md)**:

| Quantization Profile | Disk Size | Peak `VmRSS` (`256K`) | Quality Gate (`100` pts) | Prefill (`pp512`) | Base Decode (`N=1`) | `Stockfish-PVS v2 (M=7)` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Baseline-Q4_0` (`4.50 bpw` Official GGUF)** | `17.59 GiB` | `21.71 GiB` | `100.0 / 100.0` | `241.0 – 306.8 tok/s` | `19.54 tok/s` | `40.91 avg / 45.43 peak tok/s` |
| **`Fractal-MXFP4` (`4.25 bpw`)** | `15.95 GiB` | `18.86 GiB` | **`100.0 / 100.0`** | `241.0 – 306.8 tok/s` | **`26.08 tok/s` (`+33.5%`)** | **`48.39 avg / 60.40 peak tok/s`** *(`80.8` Vulkan)* |
| **`Fractal-64-BF16` (`3.75 bpw`)** | `14.36 GiB` | `17.27 GiB` | **`100.0 / 100.0`** | `241.0 – 306.8 tok/s` | **`25.02 tok/s` (`+28.0%`)** | **`48.39 avg / 60.40 peak tok/s`** *(`96.4%` acc)* |
| **`LUT-VQ3-64` (`3.00 / 3.75 bpw`)** | **`13.30 GiB`** *(`13.08` full)* | **`16.04 GiB` (`-5.67 GiB`)** | **`100.0 / 100.0`** | `241.0 – 306.8 tok/s` | **`22.44 tok/s` (`+14.8%`)** | **`24.18 E2E / 60.40 peak tok/s`** *(`70.2%` gate acc)* |

### 3. `DeepSeek-V4-Flash 284B` (`13.17B` Active) on Apple Silicon (`Apple M5 Max`, `64 GB Unified Memory`)
- **Prefill Throughput (`pp512`)**: **`169.4 ± 29.6 tok/s`** (Apple Metal accelerated)
- **Decode Throughput (`tg128`)**: **`27.0 ± 0.4 tok/s`** in `llama-bench`
- **Real Served Generation (`llama-server`)**: **`20.7 – 23.6 tok/s`** across coding and prose (`code_edit`: `20.7 tok/s`, `code_new`: `23.3 tok/s`, `free_text`: `23.6 tok/s`) with **`0` swapouts** (`iogpu.wired_limit_mb=57344`)
- **Architecture & Placement**: Apple M5 Max (18-Core CPU + 40-Core Metal GPU), 64 GB LPDDR5X. 5 pure-`IQ1_S` MoE layers offloaded to CPU (`blk.(10|12|14|16|18).ffn_.*_exps.=CPU`), remainder on Metal.
- **Driver & Patches**: `patches/apple-silicon/0001` (Fractal-64 Q2_0 NEON/Metal), `0002` (Metal tensor run mapping), `0003` (Metal register unpack). Built via [`apple-silicon/build-dsv4-engine.sh`](apple-silicon/build-dsv4-engine.sh) and launched with [`apple-silicon/run-dsv4.sh`](apple-silicon/run-dsv4.sh). Contributed by Bobby Grayson ([`@notactuallytreyanastasio`](https://github.com/notactuallytreyanastasio)). Full report: [`apple-silicon/results/dsv4-m5max.md`](apple-silicon/results/dsv4-m5max.md).

---

## Community CPU and Hybrid Leaderboard

Run **`./modchip bench`** (or **`windows\BENCHMARK_PC.bat`** on Windows) and **[Submit Your Scorecard (GitHub Issue)](https://github.com/synchopate/llama-modchip/issues/new?template=cpu_benchmark_submission.yml)** *(pure-CPU entries sorted by Refactor decode)*:

| Hardware Platform (`CPU / SoC / Hybrid`) | Cores / Cache | Memory | Prefill (`tok/s`) | Decode — Prose / C++20 | Decode — `Stockfish M=7` Burst | Config & Contributor |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **AMD Threadripper PRO 7985WX (`Zen 4`)** | `64C/128T` (`8 CCDs`, `256 MB` L3) | `256 GB` 8-Ch DDR5-6000 (`0 GPU`) | **`386.00`** *(`-t 32`)* | **`73.84 – 80.14 tok/s`** *(`70.0%` acc)* | **`111.62 tok/s` warm** / **`80.14` cold** *(`-t 32` 4C/CCD)* | `@aifartist` ([Issue #1](https://github.com/synchopate/llama-modchip/issues/1#issuecomment-6595559921)) |
| **AMD Ryzen 9 7945HX3D + RX 9070 XT (`Hybrid`)** | `16C/32T` (`128 MB` 3D V-Cache) + `16 GB` Vulkan | `94 GB` DDR5 + `16 GB` VRAM | **`563.00`** *(256K ctx)* | **`75.10 tok/s`** *(`MXFP4` 15.2GB)* | **`80.80 tok/s`** *(Vulkan `SSM_SCAN` + `MXFP4` MoE)* | `Community (RADV/Vulkan)` ([`FAQ.md`](FAQ.md)) |
| **AMD Ryzen 9 9955HX (`Zen 5`)** | `16C/32T` (`2 CCDs`, `64 MB` L3) | `64 GB` DDR5 (`0 GPU`) | **`241.0`** *(Peak **`306.8`**)* | **`25.02 – 34.5 tok/s`** *(`Fractal-64` / `MXFP4`)* | **`48.39 avg / 60.40 peak`** *(`len 6.90`)* | `@synchopate` ([`docs/NEMOTRON_OPTIMIZATIONS.md`](docs/NEMOTRON_OPTIMIZATIONS.md)) |
| **Apple M3 Max (`Metal` · `Fractal-64` Fast-Decode)** | `ARM64 NEON` + `Metal` | `Unified` (`Q2_0` Metal Kernel) | *Metal* | **`78.60 tok/s`** *(`tg128` fast-decode `Q2_0`)* | **`78.60+ tok/s`** *(Register-tiled Metal `Q2_0`)* | `@austindixson` ([PR #24](https://github.com/synchopate/llama-modchip/pull/24)) |
| **Apple M4 Max (`14C+32C Metal` · `Q4_0`)** | `ARM64 NEON` + `Metal` | `36 GB` LPDDR5X-8533 (`Unified`) | **`860.13`** *(`1,040.2` `pp512`)* | **`77.23 tok/s`** *(C++20, `203 ms` TTFT; `80.34` `tg128`)* | **`77.95 tok/s`** *(`81.0` short; Nemotron 30B PPL `2.086 ± 0.037`, Gate `100/100`)* | `@khairullinr` ([Issue #16](https://github.com/synchopate/llama-modchip/issues/16#issuecomment-6751407859)) |
| **Apple M4 Max (`14C+32C Metal` · `Fractal-64`)** | `ARM64 NEON` + `Metal` | `36 GB` LPDDR5X-8533 (`Unified`) | **`615.89`** *(`979.9` `pp512`)* | **`71.60 tok/s`** *(C++20, `47 ms` TTFT; `72.47` `tg128`)* | **`71.31 tok/s`** *(vs `Q4_0` ref, code: Top-1 `94.35%`, KLD `0.0566` nats, Gate `100/100`)* | `@khairullinr` ([Issue #16](https://github.com/synchopate/llama-modchip/issues/16#issuecomment-6752457578)) |
| **Apple M5 Max (`DeepSeek-V4 284B` · `56.57 GiB`)** | `18C CPU` + `40C Metal` | `64 GB` LPDDR5X (`286.5 GB/s` BW) | **`169.40`** *(`pp512`)* | **`27.00 tok/s`** *(`llama-bench` `tg128`)* | **`20.70 – 23.60 tok/s`** *(`llama-server`, `0` swapouts)* | `@notactuallytreyanastasio` ([`dsv4-m5max.md`](apple-silicon/results/dsv4-m5max.md)) |
| **Apple M4 Max (`16C CPU + 40C GPU`)** | `16C` (`ARM64 NEON` + `40C Metal`) | `64 GB` LPDDR5X-8533 (`238.9 GB/s` BW) | **`~451.0`** *(Audit)* | **`49.2 – 57.7 tok/s`** *(Prose / Code)* | **`70.9 – 120.0 tok/s`** *(Agent / Refactor)* | `@jbeckman1974` ([Issue #8](https://github.com/synchopate/llama-modchip/issues/8)) |
| **Apple M1 Max (`Metal` · `Fractal-64` Fast-Decode)** | `10C CPU` + `32C Metal` | `Unified` (`Q2_0` Metal Kernel) | *Metal* | **`57.65 tok/s`** *(`+86%` vs prior `Q2_0` Metal)* | **`57.65+ tok/s`** *(Register-tiled Metal `Q2_0`)* | `@austindixson` ([PR #24](https://github.com/synchopate/llama-modchip/pull/24)) |
| **Apple M4 Pro (`14-Core SoC`)** | `14C CPU + 20C Metal` | `24 GB` Unified (`0 Swap`) | **`256.0 – 549.0`** | **`47.0 – 48.0 tok/s`** *(128K)* | **`49.0+ tok/s`** | `@jarorosFork` ([`apple-silicon/`](apple-silicon/README.md)) |
| **AWS Graviton3 `m7g.2xlarge` (`ARM64 Neoverse-V1`)** | `8 vCPU` (`32 MB` L3, `SVE-256`) | `31 GB` DDR5 (`0 GPU`) | **`101.67`** *(`-t 8`)* | **`41.05 tok/s`** *(C++20)* | **`44.97 tok/s`** *(Refactoring)* | `@gerchik` ([Issue #2](https://github.com/synchopate/llama-modchip/issues/2)) |
| **AMD EPYC 7402P (`Zen 2 Rome` · `AVX2-only`)** | `24C/48T` (`8 CCDs`, `128 MB` L3) | `128 GB` 8-Ch DDR4 (`0 GPU`) | **`158.73`** *(`-t 24`)* | **`27.29 tok/s`** *(C++20, `567 ms` TTFT)* | **`31.22 tok/s`** *(Cold Run 1, `66.7%` acc)* | `@AdamKlob` ([Issue #6](https://github.com/synchopate/llama-modchip/issues/6)) |
| **AMD Ryzen 9 9950X (`Zen 5 Desktop`)** | `16C/32T` (`2 CCDs`, `64 MB` L3) | `64 GB` DDR5-6800 (`0 GPU`) | **`135.64`** *(v2.1 `Q4_0`)* | **`20.78 tok/s`** *(C++20, `526 ms` TTFT)* | **`24.67 tok/s`** *(Cold Run 1, `66.5%` acc)* | `@nadams` ([Issue #5](https://github.com/synchopate/llama-modchip/issues/5)) |
| **AMD Ryzen 7 8745H w/ Radeon 780M (`Zen 4`)** | `8C/16T` (`16 MB` L3, `AVX-512`) | `96 GB` DDR5-5600 (`0 GPU`, Arch) | **`128.81`** *(`-t 8`)* | **`19.48 tok/s`** *(C++20, `686 ms` TTFT)* | **`23.38 tok/s`** *(`69.8%` PVS acc)* | `@arch-btw` ([Issue #9](https://github.com/synchopate/llama-modchip/issues/9)) |
| **AMD Ryzen 5 5600X (`Zen 3 Desktop` · `Fractal-64`)** | `6C/12T` (`32 MB` L3, `AVX2`) | `32 GB` DDR4-3200 (`XMP/DOCP`, `0 GPU`) | **`63.90`** *(`2MB THP`)* | **`19.04 tok/s`** *(C++20, `1070 ms` TTFT)* | **`21.58 tok/s`** *(`65.3%` PVS acc)* | `@alexander-jung` ([Issue #17](https://github.com/synchopate/llama-modchip/issues/17#issuecomment-6748843016)) |
| **AMD Ryzen 7 5700X (`Zen 3 Desktop` · `Q4_0`)** | `8C/16T` (`32 MB` L3, `AVX2`) | `64 GB` DDR4-3600 (`0 GPU`, Ubuntu 26.04) | **`101.93`** *(`-t 8`)* | **`16.04 tok/s`** *(C++20, `856 ms` TTFT)* | **`19.66 tok/s`** *(`67.9%` PVS acc)* | `@germanpsycho` ([Issue #23](https://github.com/synchopate/llama-modchip/issues/23)) |
| **AMD Ryzen 5 5600X (`Zen 3 Desktop` · `Q4_0`)** | `6C/12T` (`32 MB` L3, `AVX2`) | `32 GB` DDR4-3200 (`XMP/DOCP`, `0 GPU`) | **`86.96`** *(`88.77` w/ THP)* | **`14.95 tok/s`** *(C++20, `968 ms` TTFT)* | **`18.07 tok/s`** *(`64.5%` PVS acc)* | `@alexander-jung` ([Issue #17](https://github.com/synchopate/llama-modchip/issues/17)) |
| **Intel Core Ultra 5 125H (`Meteor Lake-H`)** | `4P+8E+2LP-E` (`18 MB` L3, `AVX-VNNI`, LP-E excl.) | `32 GB` DDR5 (`50.2 GB/s`, `0 GPU`) | **`55.19`** *(`-t 12`; `60.53` on `4P`)* | **`14.63 tok/s`** *(C++20, `1313 ms` TTFT)* | **`16.61 tok/s`** *(`67.9%` PVS acc)* | `@xma` ([Issue #18](https://github.com/synchopate/llama-modchip/issues/18)) |
| **AMD Ryzen 5 7430U (`Zen 3 Mobile 15W`)** | `6C/12T` (`16 MB` L3, `AVX2`) | `32 GB` DDR4-3200 (`0 GPU`, Pop!_OS) | **`59.03`** *(`-t 6`)* | **`13.03 tok/s`** *(C++20, `2072 ms` TTFT)* | **`15.57 tok/s`** *(`68.3%` PVS acc)* | `@armadillu` ([Issue #12](https://github.com/synchopate/llama-modchip/issues/12)) |
| **AMD Ryzen 7 PRO 6850U (`ThinkPad T14s Gen 3`)** | `8C/16T` (`16 MB` L3, `AVX2`) | `32 GB` LPDDR5-4800 (`0 GPU`, Debian 13) | **`62.74`** *(`-t 8`)* | **`12.61 tok/s`** *(C++20, `1204 ms` TTFT)* | **`14.67 tok/s`** *(`66.0%` PVS acc)* | `@mkesper` ([Issue #15](https://github.com/synchopate/llama-modchip/issues/15)) |
| **AMD Ryzen 7 5700U (`Zen 2 Mobile 15W` · `Q4_0`)** | `8C/16T` (`8 MB` L3, `AVX2`) | `64 GB` DDR4 (`0 GPU`, Debian) | **`43.93`** *(v2.4 `-t 8`)* | **`13.04 tok/s`** *(C++20, `1722 ms` TTFT)* | **`14.48 tok/s`** *(`66.5%` PVS acc)* | `@JamesStallings` ([#10](https://github.com/synchopate/llama-modchip/issues/10) → [#20](https://github.com/synchopate/llama-modchip/issues/20)) |
| **AMD Ryzen 7 5700U (`Zen 2 Mobile 15W` · `Fractal-64`)** | `8C/16T` (`8 MB` L3, `AVX2`) | `64 GB` DDR4 (`0 GPU`, Debian) | **`32.48`** *(v2.1 `Fractal-64`)* | **`13.55 tok/s`** *(C++20, `2233 ms` TTFT)* | **`13.52 tok/s`** *(`64.9%` PVS acc)* | `@JamesStallings` ([Issue #19](https://github.com/synchopate/llama-modchip/issues/19)) |
| **Intel Core i9-11950H (`Tiger Lake-H`)** | `8C/16T` (`24 MB` L3, `AVX-512`) | `128 GB` DDR4-3200 (`0 GPU`, Fedora) | **`85.39`** *(`-t 8`)* | **`10.76 tok/s`** *(C++20, `1106 ms` TTFT)* | **`13.37 tok/s`** *(`69.8%` PVS acc)* | `@ELO-I` ([Issue #11](https://github.com/synchopate/llama-modchip/issues/11)) |
| **Dual Intel Xeon E5-2670 (`Sandy Bridge-EP 2012`)** | `16C/32T` (`2×20 MB` L3, `AVX`-only) | `94 GB` RAM (`0 GPU`, Ubuntu 24.04) | **`47.31`** *(`-t 16`)* | **`10.69 tok/s`** *(C++20, `2014 ms` TTFT)* | **`11.47 tok/s`** *(`65.0%` PVS acc, `3.0x` floor)* | `@Mammux` ([Issue #14](https://github.com/synchopate/llama-modchip/issues/14)) |
| **AMD Ryzen 5 2600X (`Zen+ Desktop` · `Q4_0`)** | `6C/12T` (`16 MB` L3, `AVX2`) | `32 GB` DDR4-3200 (`29.4 GB/s`, Win11+WSL2) | **`21.06`** *(`-t 6`)* | **`7.60 tok/s`** *(C++20)* | **`10.42 tok/s`** *(`68.9%` PVS acc)* | `@lawsochi` ([Issue #26](https://github.com/synchopate/llama-modchip/issues/26#issuecomment-6780282141)) |
| **Windows 11 Hybrid (`i9-9900K + RTX 2080 Ti 11GB`)** | `8C/16T` (`16 MB` L3) + `11 GB` CUDA | `16 GB` DDR4 + `11 GB` GDDR6 | *Auto CUDA 12.4* | **`50–58 tok/s`** *(`14B`) / 30B Hybrid* | **`100+ tok/s`** *(`Nemotron-H-8B` 128K)* | [`windows/`](windows/README.md) |
| **AMD Ryzen 7 8745HS (`GEEKOM A8`)** | `8C/16T` (`16 MB` L3) | `64 GB` DDR5-5600 | *Verified Fedora 44* | *Verified (`ngram`)* | *Verified (`CPU` & `780M`)* | `@jarorosFork` ([`fedora/`](fedora/README.md)) |
| **`DeepSeek-V4-Flash 284B` on Ryzen 9 9955HX** | `16C/32T` (`2 CCDs`, `64 MB` L3) | `64 GB` DDR5 (`58.2–58.9 GiB` RSS, `32K`/`256K`) | **`41.11` cold / `44.36` warm** *(`10.4 ms` Slot-0)* | **`9.05 – 9.55 tok/s`** *(`N=1` Top-6; `10.74` Top-4)* | **`11.50 – 25.30 tok/s`** *(`PVS` / Tools)* | `@synchopate` ([`docs/DEEPSEEK_V4_OPTIMIZATIONS.md`](docs/DEEPSEEK_V4_OPTIMIZATIONS.md)) |
| **`DeepSeek-V4-Flash 284B` on Ryzen 7 9800X3D (`Win11`)** | `8C/16T` (`96 MB` 3D V-Cache, `AVX-512`) | `64 GB` DDR5 (`56.40–56.57 GiB`, `1M` ctx) | **`21.30 – 22.60 tok/s`** *(`pp512`)* | **`6.35 – 8.04 tok/s`** *(`Top-6` / `Top-4` `tg128`)* | **`8.04+ tok/s`** *(Native Win11, `1M` ctx verified)* | `@MR1KRAKEN` ([PR #27](https://github.com/synchopate/llama-modchip/pull/27), [Issue #28](https://github.com/synchopate/llama-modchip/issues/28)) |

<details>
<summary><b>Repository Structure & Manual CMake Build Instructions</b></summary>
<br/>

```text
llama-modchip/
├── modchip                                    # Unified CLI (setup | start | chat | bench | eval | quantize | web | claude | stop)
├── install.sh                                 # Automated Installer (Linux)
├── README.md / ARCHITECTURE.md / FAQ.md       # Overview, Mathematical Architecture & Engineering FAQ
├── RELEASE_NOTES.md                           # v2.5.0 Release Notes
├── docs/
│   ├── NEMOTRON_OPTIMIZATIONS.md              # Nemotron-3.5-30B-A3B & Nemotron-H-8B optimizations
│   ├── DEEPSEEK_V4_OPTIMIZATIONS.md           # DeepSeek-V4-Flash 284B profiling, SVD & L2-quadrature MoE
│   ├── HARDWARE_OPTIMIZATIONS.md              # Chip-by-chip optimizations (Zen 5, Threadripper, Vulkan, CUDA, Metal, ARM64)
│   └── WINDOWS_PORT.md                        # Native Windows 11 patched-engine build & Ryzen 7 9800X3D 1M-context report
├── vq3/                                       # LUT-VQ3-64 3D Lloyd-Max codebook, self-contained C encoder & kernel tests
├── patches/
│   ├── apple-silicon/                         # Apple Silicon engine patches (Fractal-64 Q2_0 NEON/Metal, register unpack)
│   ├── deepseek-v4-flash-fractal-256k.patch   # Base engine patch (DeepSeek-V4-Flash 284B + Fractal-64 Q2_0 = 42)
│   ├── lut-vq3-64-avx512-vbmi.patch           # Incremental LUT-VQ3-64 (VQ3_0 = 43) AVX-512 VBMI + AVX2 + scalar patch
│   ├── vq3_codebook.h                         # 128-entry 3D + 64-entry 2D LUT-VQ3-64 codebook header
│   └── zen5-nemotron-256k-optimizations.patch # Standalone ggml/llama core patch for Nemotron-3.5
├── windows/                                   # Windows 10/11 Patched Engine Builder (build_patched_engine.ps1, AVVIA_DSV4.bat) & Hybrid Pack
├── apple-silicon/                             # Apple Silicon M1–M5 setup scripts (build-dsv4-engine.sh, run-dsv4.sh, M5 Max report)
├── fedora/                                    # Fedora 44 / AMD Ryzen 7 8745HS (GEEKOM A8) all-in-one script
├── speculative/                               # RouteCrack + Stockfish-PVS v2 (rc_stockfish.h, ngram-map.cpp)
├── benchmarks/                                # eval_quality_and_tools.py (100-pt Quality Gate), QUALITY_GATE_VQ3_64.md & results/
├── gateway/sse_gateway.py                     # Agentic SSE Proxy + Compact Tool Table + Dark-Mode Web Chat HUD (:8787)
└── scripts/                                   # build_zen5_llama.sh, build_{nemotron,dsv4}_vq3.py, build_fractal_moe_gguf.sh, chat.py
```

</details>

---

## Acknowledgments

- **Salvatore Sanfilippo ([`@antirez`](https://github.com/antirez))** and **Prem Research ([`prem-research`](https://huggingface.co/prem-research))** for directional orthogonalization research and the [`DeepSeek-V4-Flash-0731-abliterated`](https://huggingface.co/prem-research/DeepSeek-V4-Flash-0731-abliterated) base checkpoint (`BF16` + native `FP8_E4M3`) from which our default single-pass 284B GGUF is built.
- **DeepSeek AI ([`deepseek-ai`](https://huggingface.co/deepseek-ai))** and **NVIDIA** for the `DeepSeek-V4-Flash` and `Nemotron-3.5-30B-A3B` base architectures, and **`ggml-org/llama.cpp`** for the upstream C/C++ inference runtime.

---

<div align="center">

**[Nemotron Deep-Dive (`docs/NEMOTRON_OPTIMIZATIONS.md`)](docs/NEMOTRON_OPTIMIZATIONS.md)** · **[DeepSeek-V4 284B Deep-Dive (`docs/DEEPSEEK_V4_OPTIMIZATIONS.md`)](docs/DEEPSEEK_V4_OPTIMIZATIONS.md)** · **[Hardware & Chip Matrix (`docs/HARDWARE_OPTIMIZATIONS.md`)](docs/HARDWARE_OPTIMIZATIONS.md)** · **[Full Paper (`ARCHITECTURE.md`)](ARCHITECTURE.md)** · **[FAQ (`FAQ.md`)](FAQ.md)**

</div>
