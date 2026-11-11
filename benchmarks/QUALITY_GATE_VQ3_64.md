# Quality, Tool-Use, and Signal Precision Report: `LUT-VQ3-64` (3.00 bpw) & `Fractal-64` (3.75 bpw)

- **Status:** Passed (`v2.5.0`)
- **Target Models:**
  - `NVIDIA-Nemotron-3.5-Lightning-30B-A3B` (`52` blocks: `23` Mamba-2 SSM + `23` Sparse MoE + `6` GQA Attention)
  - `DeepSeek-V4-Flash-0731` (`284.33B` total / `13.17B` active, `43` MLA + `256+1s` MoE layers, `Top-6` routing)
- **Testbed:** AMD Ryzen 9 9955HX (`16` Zen 5 cores, `64 MB` dual-CCD L3 cache), DDR5-5600, Linux 6.8, Transparent HugePages (`madvise`).
- **Evaluation Harness:** [`benchmarks/eval_quality_and_tools.py`](eval_quality_and_tools.py) (`30` pts Agentic Tool-Calling, `30` pts Executable Python Code Gen, `20` pts Long-Context Needle & Arithmetic, `20` pts Throughput & Memory Telemetry).

---

## 1. Format Specification & Summary

`llama-modchip v2.5.0` pairs two block-64 CPU quantization formats:

- **`Fractal-64` (`GGML_TYPE_Q2_0 = 42`, `30 bytes / 64 weights = 3.75 bpw`):**
  - `2 bytes` `FP16` super-block scale $d$, `4 bytes` storing eight 4-bit unsigned sub-scales ($m_b = 4 + u_b \in \{4..19\}$ for eight 8-weight sub-blocks), and `24 bytes` (`qs[16]` + `qh[8]`) of 3-bit indices into the symmetric non-linear dyadic codebook `{-14, -8, -4, -1, +1, +4, +8, +14}`.
  - Achieves **`17.75 dB` SQNR (`99.15%` cosine similarity)** directly on uncompressed `BF16` MoE weights (`+1.45 dB` over `ggml` `Q3_K`), decoded via single-cycle `VPSHUFB` lookup and sub-scale fusion in `VPMADDWD`.
- **`LUT-VQ3-64` (`GGML_TYPE_VQ3_0 = 43`, `24 bytes / 64 weights = 3.00 bpw`):**
  - `2 bytes` `FP16` super-block scale $d$ (scaled by `GGML_VQ3_G = 8.6373016238e-3f`) and `22 bytes` (`qs[22]`):
    - `qs[0..19]` bits `0..6`: `20` 7-bit 3D Lloyd-Max indices (`60` weights at stride 22: `j, j+22, j+44`).
    - `qs[20..21]` bits `0..5`: `2` 6-bit 2D tail indices (`4` weights at `20, 42` and `21, 43`, half-table selected by bit 6).
    - `s24` (`24 bits` packed in bit 7 of `qs[0..21]` and bit 6 of `qs[20..21]`): **eight 3-bit unsigned sub-scales** ($m_b = 2 + u_b \in \{2..9\}$ for eight 8-weight sub-blocks).
  - Achieves **`13.57 dB` SQNR (`97.78%` cosine similarity)** directly on `BF16` weights (`+3.42 dB` over `ggml` `Q2_K`), decoded in registers via three AVX-512 VBMI `VPERMI2B` (`_mm512_mask2_permutex2var_epi8`) lookups against a 384-byte codebook held in 6 ZMM registers.

### Shipped Hybrid Topology vs. Full-Expansion Ablation

1. **`NVIDIA-Nemotron-3.5-30B-A3B-VQ3-64` (`13.30 GiB` shipped hybrid / `13.08 GiB` `--all-up-exps` ablation):**
   - **Shipped Hybrid (`13.30 GiB`, `14,282,559,040` B):** Preserves boundary MoE layers (`blk.1, 3, 49, 51`) and all 23 residual-stream `ffn_down_exps` tensors in `Fractal-64` (`3.75 bpw`), while encoding the 19 interior `blk.6..47.ffn_up_exps` (pre-`ReLU²`) tensors from `BF16` into `LUT-VQ3-64` (`3.00 bpw`).
   - **All-23 `ffn_up_exps` Ablation (`13.08 GiB`, [`results/eval_nemotron_vq3_64.json`](results/eval_nemotron_vq3_64.json)):** Even with all 23 `ffn_up_exps` tensors in `VQ3_0`, the checkpoint achieves **`100.0 / 100.0`** (`80.0 / 80.0` quality + `20.0 / 20.0` speed/RAM) at `24.18 tok/s` decode (`70.19%` Stockfish-PVS acceptance) in `16.04 GiB` `VmRSS`.
2. **`DeepSeek-V4-Flash-0731-VQ3` (`56.40 GiB` shipped hybrid / `56.07 GiB` `--aggressive` ablation):**
   - **Shipped Hybrid (`56.40 GiB`, `60,558,856,608` B across 3 shards):** Keeps 100% of MLA attention (`attn_q_b`, `attn_output_a`, `attn_output_b`, `attn_kv`, `compressor_*`) untouched in `Fractal-64` (`3.75 bpw`, zero double-quantization), keeps boundary shared experts (`blk.0..4, 39..42`) and all `ffn_down_shexp` in `Q5_K`/`Q6_K`, and encodes the 68 interior shared-expert pre-`SiLU` projections (`blk.5..38` `ffn_gate_shexp` & `ffn_up_shexp`) into `LUT-VQ3-64` (`3.00 bpw`).
   - **255-Tensor Aggressive Ablation (`56.07 GiB`, [`results/eval_dsv4_vq3_64.json`](results/eval_dsv4_vq3_64.json)):** Also transcodes MLA expansion matrices (`3.94B` active weights/token) into `VQ3_0`, raising uncached `tg64` decode from `7.74 tok/s` to `9.05 tok/s` (`+16.9%`) and `pp128` prefill from `37.79 tok/s` to `41.11 tok/s` (`+8.8%`) in `58.31 GiB` `VmRSS`, while scoring `30/30` on Tool-Calling and `20/20` on Needle/Math. The shipped `56.40 GiB` release retains `Fractal-64` on MLA attention to avoid second-generation re-quantization loss.

| Evaluation Dimension | Reference `BF16` (`16.00 bpw`) | Standard `Q4_0` (`4.50 bpw`) | `Fractal-64` (`3.75 bpw`) | `LUT-VQ3-64` Hybrid (`3.00 / 3.75 bpw`) | Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Raw `BF16` Weight SQNR (`libggml`)** | $\infty\text{ dB}$ | `21.09 dB` | `17.75 dB` | `13.57 dB` (`up`) / `17.75 dB` (`down`) | `+3.42 dB` over `Q2_K` (`10.15 dB`) |
| **Weight Cosine Similarity to `BF16`** | `100.00%` | `99.61%` | `99.15%` | `97.78%` (`up`) / `99.15%` (`down`) | `-33.3%` bytes vs `Q4_0` on `VQ3_0` tensors |
| **Relative $L_2$ Reconstruction Error** | `0.00%` | `8.82%` | `12.96%` | `20.97%` (`up`) / `12.96%` (`down`) | `-10.11` pp lower error than `Q2_K` |
| **1. Agentic Tool-Calling (`30` pts)** | `30.0 / 30.0` | `30.0 / 30.0` | `30.0 / 30.0` | `30.0 / 30.0` | Exact JSON schema compliance |
| **2. Executable Code Gen (`30` pts)** | `30.0 / 30.0` | `30.0 / 30.0` | `30.0 / 30.0` | `30.0 / 30.0` | Nemotron unit-test pass rate |
| **3. Long-Context Needle & Math (`20` pts)** | `20.0 / 20.0` | `20.0 / 20.0` | `20.0 / 20.0` | `20.0 / 20.0` | Exact canary retrieval & KV math |
| **Nemotron Stockfish-PVS Acceptance** | `62.50%` | `66.67%` | `59.09%` | `70.19%` | Single-turn refactor prompt |
| **Nemotron-30B Decode (`tok/s`)** | `7.82 tok/s` | `19.54 tok/s` | `25.02 tok/s` | `24.18 tok/s` (`22.44` `tg128`) | `+23.7%` E2E vs `Q4_0` |
| **Nemotron-30B Disk / `VmRSS`** | `59.00` / `59.40 GiB` | `17.59` / `21.71 GiB` | `14.36` / `17.27 GiB` | `13.30 GiB` (`13.08` abl. / `16.04 GiB` RSS) | `-5.67 GiB` RSS vs `Q4_0` |
| **DeepSeek-V4 284B Disk / Peak `VmRSS`** | `568.6 GiB` *(OOM)* | `162.4 GiB` *(OOM)* | `56.57` / `58.80 GiB` | `56.40 GiB` (`56.07` abl. / `58.31 GiB` RSS) | Fits in 64 GB host RAM (`0` swap) |
| **Nemotron Total Quality Gate Score** | `100.0 / 100.0` | `100.0 / 100.0` | `100.0 / 100.0` | `100.0 / 100.0` | Passed |

---

## 2. Signal-Level Reconstruction Fidelity on Uncompressed `BF16` Weights

Measured via `quantize_row_*_ref` $\to$ `dequantize_row_*` on `NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16.gguf` (`blk.1.ffn_up_exps.weight`, $N = 4,194,304$ weights, heavy-tailed kurtosis $\kappa = 3.2\text{--}4.8$):

| Quantization Format | Effective `bpw` | Index Bits / Weight | Bytes / 64 Weights | Direct `BF16` SQNR | Cosine Sim vs. `BF16` | Relative $L_2$ Error | SIMD Inner Loop |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Reference `BF16`** | `16.000` | `16.00` | `128.0 B` | $\infty$ | `100.00%` | `0.00%` | Native F32/BF16 |
| **Standard `Q4_0`** | `4.500` | `4.00` (16 uniform) | `36.0 B` | `21.09 dB` | `99.61%` | `8.82%` | Uniform 4-bit scalar |
| **`Fractal-64` (`GGML_TYPE_Q2_0 = 42`)** | `3.750` | `3.00` (8 dyadic) | `30.0 B` | `17.75 dB` | `99.15%` | `12.96%` | `VPSHUFB` + 8×4b sub-scales |
| **Standard `Q3_K`** | `3.438` | `3.00` (uniform) | `27.5 B` | `16.30 dB` | `98.84%` | `15.32%` | Multi-shift `Q3_K` unpack |
| **`LUT-VQ3-64` (`GGML_TYPE_VQ3_0 = 43`)** | `3.000` | `2.33` (128 3D VQ) | `24.0 B` | `13.57 dB` | `97.78%` | `20.97%` | `VPERMI2B` in 6 ZMMs + 8×3b sub-scales |
| **Standard `Q2_K`** | `2.625` | `2.00` (uniform) | `21.0 B` | `10.15 dB` | `95.37%` | `31.08%` | Multi-shift `Q2_K` unpack |

### Key Observations
1. **`Fractal-64` (`3.75 bpw`) exceeds `Q3_K` (`3.44 bpw`) by `+1.45 dB` (`99.15%` vs `98.84%` cosine similarity):** Its symmetric non-linear codebook `{-14, -8, -4, -1, +1, +4, +8, +14}` paired with 8-weight 4-bit sub-scales tracks heavy-tailed MoE distributions more closely than uniform 3-bit grids while avoiding 256-element block divisibility constraints (`1856 % 256 != 0` on Nemotron).
2. **`LUT-VQ3-64` (`3.00 bpw`) exceeds `Q2_K` (`2.625 bpw`) by `+3.42 dB` (`97.78%` vs `95.37%` cosine similarity):** 3D Lloyd-Max vector quantization concentrates all 128 antipodally symmetric centroids onto the populated spherical/ellipsoidal density shell in $\mathbb{R}^3$ instead of sparse hypercube corners.
3. **Asymmetric expansion/output split:** Placing `LUT-VQ3-64` (`3.00 bpw`) on pre-activation expansion matrices (`ffn_up_exps` before `ReLU²`, interior `ffn_gate_shexp` / `ffn_up_shexp` before `SiLU`) and keeping residual-stream output projections (`ffn_down_exps`, `ffn_down_shexp`) and MLA attention in higher-precision formats (`Fractal-64` `3.75 bpw` or `Q5_K`/`Q6_K`) preserves residual variance and tool-calling stability.

---

## 3. Reproducibility Commands

```bash
# 1. Run Python pack/unpack and AVX-512 golden-vector unit tests
python3 -m unittest discover -s vq3 -p "test_*.py" -v

# 2. Compile and run the standalone AVX-512 VBMI kernel accuracy test
gcc -O3 -march=native -mavx512f -mavx512bw -mavx512vbmi -mf16c -mbmi2 \
  vq3/test_vq3_ggml_kernels.c -lm -o vq3/test_vq3_ggml_kernels && ./vq3/test_vq3_ggml_kernels

# 3. Build the shipped hybrid GGUFs (pass --all-up-exps or --aggressive for full-VQ3 ablations)
python3 scripts/build_nemotron_vq3.py --fractal-gguf <Fractal-64.gguf> --bf16-gguf <BF16.gguf> --output NVIDIA-Nemotron-3.5-30B-A3B-VQ3-64.gguf
python3 scripts/build_dsv4_vq3.py --input-dir <Fractal-Dir> --output-dir ./dsv4-vq3

# 4. Run the 100-Point Quality & Tool-Calling Gate against a live endpoint
python3 benchmarks/eval_quality_and_tools.py \
  --url http://127.0.0.1:8787 \
  --label Nemotron-VQ3-64-13.30GiB \
  --compare benchmarks/results/eval_fractal_64_bf16.json \
  --save-json benchmarks/results/eval_nemotron_vq3_64.json
```
