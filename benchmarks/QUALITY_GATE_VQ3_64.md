# Quality, Tool-Use, and Signal Precision Report: `LUT-VQ3-64` (3.00 bpw), `Fractal-64` (3.75 bpw), & Single-Pass `MaxQ-Fractal`

- **Release:** `v2.5.1`
- **Target Models:**
  - `NVIDIA-Nemotron-3.5-Lightning-30B-A3B` (`52` blocks: `23` Mamba-2 SSM + `23` Sparse MoE + `6` GQA Attention, unpruned `Top-6` active)
  - `DeepSeek-V4-Flash-0731` (`284.33B` total / `13.17B` active, `43` MLA + `256+1s` MoE layers, unpruned `Top-6` active)
- **Testbed:** AMD Ryzen 9 9955HX (`16` Zen 5 cores, `64 MB` dual-CCD L3 cache), DDR5-5600, Linux, Transparent HugePages (`madvise`).
- **Evaluation Suites:**
  1. [`benchmarks/eval_perplexity_and_kld.py`](eval_perplexity_and_kld.py): Teacher-forced Perplexity on uncompressed **`WikiText-2` (`wiki.test.raw`)** (`12,288` tokens on `30B`, `8,192` tokens on `284B`) + 2-pass single-host Top-64 Logit KL-Divergence (`nats/tok` at `T=1.0`).
  2. [`benchmarks/eval_quality_and_tools.py`](eval_quality_and_tools.py): `30` pts Agentic Tool-Calling, `30` pts Executable Python Code Gen, `20` pts Long-Context Needle & Arithmetic (`80` quality pts) + `20` pts Throughput & Memory Telemetry.
  3. [`benchmarks/eval_hard60_and_refusal.py`](eval_hard60_and_refusal.py): `60`-point extended algorithmic/math suite (`Hard-60`) + `8`-prompt dual-use technical security refusal audit.

---

## 1. Format Specification & Checkpoint Topology

`llama-modchip v2.5.1` pairs two block-64 CPU quantization formats:

- **`Fractal-64` (`GGML_TYPE_Q2_0 = 42`, `30 bytes / 64 weights = 3.75 bpw`):**
  - `2 bytes` `FP16` super-block scale $d$, `4 bytes` storing eight 4-bit unsigned sub-scales ($m_b = 4 + u_b \in \{4..19\}$ for eight 8-weight sub-blocks), and `24 bytes` (`qs[16]` + `qh[8]`) of 3-bit indices into the symmetric non-linear dyadic codebook `{-14, -8, -4, -1, +1, +4, +8, +14}`.
  - Achieves **`17.75 dB` SQNR (`99.15%` cosine similarity)** directly on uncompressed `BF16` MoE weights (`+1.45 dB` over `ggml` `Q3_K`), decoded via single-cycle `VPSHUFB` lookup and sub-scale fusion in `VPMADDWD` (note: enum slot `42` reuses deprecated `GGML_TYPE_IQ4_NL_4_4` in-place).
- **`LUT-VQ3-64` (`GGML_TYPE_VQ3_0 = 43`, `24 bytes / 64 weights = 3.00 bpw`):**
  - `2 bytes` `FP16` super-block scale $d$ (scaled by `GGML_VQ3_G = 8.6373016238e-3f`) and `22 bytes` (`qs[22]`):
    - `qs[0..19]` bits `0..6`: `20` 7-bit 3D Lloyd-Max indices (`60` weights at stride 22: `j, j+22, j+44`).
    - `qs[20..21]` bits `0..5`: `2` 6-bit 2D tail indices (`4` weights at `20, 42` and `21, 43`, half-table selected by bit 6).
    - `s24` (`24 bits` packed in bit 7 of `qs[0..21]` and bit 6 of `qs[20..21]`): **eight 3-bit unsigned sub-scales** ($m_b = 2 + u_b \in \{2..9\}$ for eight 8-weight sub-blocks).
  - Achieves **`13.57 dB` SQNR (`97.78%` cosine similarity)** directly on `BF16` weights (`+3.42 dB` over `ggml` `Q2_K`), decoded in registers via three AVX-512 VBMI `VPERMI2B` (`_mm512_mask2_permutex2var_epi8`) lookups against a 384-byte codebook held in 6 ZMM registers (with portable `AVX2` fallback).

### Published Checkpoint Variants (`v2.5.1`)

1. **`NVIDIA-Nemotron-3.5-30B-A3B` (`16–24 GB+` RAM, NVIDIA Open Model License):**
   - **Aligned Default `VQ3-64` (`13.30 GiB`, `14,282,559,040` B):** Boundary MoE layers (`blk.1, 3, 49, 51`) and all 23 `ffn_down_exps` in `Fractal-64` (`3.75 bpw`); 19 interior `blk.6..47.ffn_up_exps` (pre-`ReLU²`) in `LUT-VQ3-64` (`3.00 bpw`). Passing `--all-up-exps` produces the `13.08 GiB` 23-layer ablation ([`results/eval_nemotron_vq3_64.json`](results/eval_nemotron_vq3_64.json), `100.0 / 100.0` at `24.18 tok/s`, `70.19%` `Stockfish-PVS` acceptance, `16.04 GiB` `VmRSS`).
   - **Aligned `Fractal-64` (`14.36 GiB`, `15,420,034,624` B):** All 23 routed MoE layers in `Fractal-64` (`3.75 bpw`), recommended for non-VBMI `AVX2` CPUs and Apple Silicon Metal (`M1–M5`).
   - **Opt-In Uncensored `Abliterated-VQ3-64` (`13.30 GiB`, `14,282,559,488` B):** Single-pass `VQ3-64 + Fractal-64` quantization of `HangGlidersRule/Darkstar-Nemotron-3.5-Lightning-30B-A3B-Abliterated-BF16` ([`scripts/build_nemotron_abliterated_vq3.py`](../scripts/build_nemotron_abliterated_vq3.py)).
2. **`DeepSeek-V4-Flash-0731 284B` (`64 GB+` RAM, MIT License):**
   - **Aligned Default [`MaxQ-Fractal`](https://huggingface.co/Synchopate/DeepSeek-V4-Flash-0731-MaxQ-Fractal-64GB-CPU-GGUF) (`56.93 GiB`, `61,129,363,968` B):** Built single-pass from `deepseek-ai/DeepSeek-V4-Flash` ([`scripts/build_dsv4_maxq_singlepass.py`](../scripts/build_dsv4_maxq_singlepass.py)) with **174 dense 100%-active bottleneck tensors** (`173 × Q6_K + 1 × Q8_0`: `token_embd`, `output`, `43× attn_q_a`, `43× ffn_{gate,up,down}_shexp`), **317 MLA & Lightning Indexer tensors** in `Fractal-64` (`3.75 bpw`), and all **129 3D routed MoE tensors** quantized single-pass from `MXFP4` with `@antirez`'s `1.5M`-token `imatrix`.
   - **Opt-In Uncensored [`Abliterated-Fractal`](https://huggingface.co/Synchopate/DeepSeek-V4-Flash-0731-Abliterated-Fractal-64GB-CPU-GGUF) (`56.93 GiB`, `61,129,364,000` B):** Identical single-pass `174 × Q6_K/Q8_0` + `Fractal-64` recipe applied to `prem-research/DeepSeek-V4-Flash-0731-abliterated` ([`results/gate_Abliterated-Fractal-MaxQ-56.93GiB.json`](results/gate_Abliterated-Fractal-MaxQ-56.93GiB.json)).
   - **Compact Aligned `VQ3` (`56.40 GiB` shipped / `56.07 GiB` `--aggressive`):** 100% MLA in `Fractal-64` + 68 interior `shexp` in `LUT-VQ3-64` ([`results/gate_VQ3-DoubleQuant-Baseline.json`](results/gate_VQ3-DoubleQuant-Baseline.json): `64.0 / 80` Gate, `30.0 / 60` `Hard-60`, `9.15 tok/s` served / `9.05 tok/s` raw `N=1`). Note that [`results/eval_dsv4_vq3_64.json`](results/eval_dsv4_vq3_64.json) (`70.0 / 80` Gate, `10.74 tok/s`) records the `--aggressive` `56.07 GiB` ablation **with `Stockfish-PVS` speculation active** (`67.95%` draft acceptance across `156` drafted tokens), whereas raw `N=1` unpruned `Top-6` decode without speculation is `9.05 tok/s`.
   - **Cross-Platform Aligned `Fractal` (`56.57 GiB`):** Pure `Fractal-64` MLA + `Q5_K`/`Q6_K` `shexp` ([`results/gate_Fractal-Reference-56.57GiB.json`](results/gate_Fractal-Reference-56.57GiB.json): `64.0 / 80` Gate, `20.0 / 60` `Hard-60`, `9.55 tok/s` served).

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

---

## 3. Continuous `WikiText-2` Perplexity (`PPL`) & 2-Pass Logit KL-Divergence (`KLD`)

Discrete functional gates (`80/80` across JSON tool-calling, unit-tested code generation, and long-context needle/math) verify structural non-collapse, but saturate when multiple quantization profiles pass every assertion—especially on `DeepSeek-V4-Flash 284B` where routed MoE experts sit at `IQ1_S` / `IQ1_M` / `IQ2_XXS` (`1.56–2.06 bpw`).

[`benchmarks/eval_perplexity_and_kld.py`](eval_perplexity_and_kld.py) measures two continuous, non-saturating end-to-end metrics across the full model graph:
1. **Standard Uncompressed `WikiText-2` (`wiki.test.raw`) Teacher-Forced Perplexity (`PPL ± σ`)**: Auto-downloaded from `cosmo.zip` (`wiki.test.raw`, `1,290,590` bytes, zero `<unk>` pre-masking), evaluated at `--ctx-size 512` across `24` chunks (`12,288` tokens) on `Nemotron-30B` and `16` chunks (`8,192` tokens) on `DeepSeek-V4 284B`.
2. **Two-Pass Single-Host Top-64 Logit KL-Divergence (`nats/tok` at `T=1.0`)**: Compares output distributions (`n_probs = 64`) across 12 deterministic C++20, Python, SQL/HTTP, number-theory, and MoE/MLA probes against committed reference logit snapshots ([`results/dsv4_maxq_ref_logits.json`](results/dsv4_maxq_ref_logits.json) and [`results/nemotron_fractal_ref_logits.json`](results/nemotron_fractal_ref_logits.json)).

### A. `NVIDIA-Nemotron-3.5-30B-A3B` — `WikiText-2` (`12,288` tok) Perplexity & Logit KLD

| `Nemotron-3.5-30B-A3B` Checkpoint | Effective MoE `bpw` | Disk Size | **`WikiText-2` `PPL ± σ` (`12,288` tok)** | **2-Pass Logit KLD vs. `Fractal-64`** | Functional Gate (`100` pts) | Security Audit (`8` prompts) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Fractal-MXFP4` (`15.95 GiB`, Aligned)** | `4.25 bpw` | `15.95 GiB` | **`7.2927 ± 0.25382`** | — | `100.0 / 100.0` | — |
| **`Fractal-64` (`14.36 GiB`, Aligned)** | `3.75 bpw` | `14.36 GiB` | **`7.3610 ± 0.26017`** *(`+0.94%` vs MXFP4)* | **`0.0000 nats/tok` (`100%` Ref)** | `100.0 / 100.0` | — |
| **`VQ3-64` (`13.30 GiB`, Aligned Default)** | `3.00 / 3.75 bpw` | `13.30 GiB` | **`7.4400 ± 0.26263`** *(`+1.07%` vs Fractal)* | **`0.0561 nats/tok (83.3% Top-1)`** | `100.0 / 100.0` | `8 / 8` |
| **`Abliterated-VQ3-64` (`13.30 GiB`, Opt-In)** | `3.00 / 3.75 bpw` | `13.30 GiB` | **`7.7721 ± 0.28352`** *(`+4.46%` vs Aligned VQ3)* | **`0.1357 nats/tok (75.0% Top-1)`** | `100.0 / 100.0` | `8 / 8` |

*(Note: Independent community audit `@khairullinr` in [Issue #16](https://github.com/synchopate/llama-modchip/issues/16) additionally measured `PPL = 2.086 ± 0.037` on `Q4_0` and `Logit KLD = 0.0566 nats/tok` (`94.35%` Top-1 agreement) between `Fractal-64` and `Q4_0` on a deterministic Python source-code corpus.)*

### B. `DeepSeek-V4-Flash-0731 284B` — Unconfounded `MaxQ-Fractal` vs. `Abliterated-Fractal` & `VQ3`

Because `MaxQ-Fractal` and `Abliterated-Fractal` use the **exact same single-pass quantization recipe** (`173 × Q6_K + 1 × Q8_0` dense bottlenecks, `317 × Fractal-64` MLA/Indexer tensors, and `@antirez`'s `1.5M`-token `imatrix` routed MoE), comparing them isolates the effect of upstream refusal abliteration from single-pass `Q6_K` quantization:

| `DeepSeek-V4-Flash 284B` Checkpoint (`64 GB` Host) | Disk Size | Anonymous `VmRSS` | **`WikiText-2` `PPL ± σ` (`8,192` tok)** | **2-Pass Logit KLD vs. `MaxQ`** | Functional Gate (`80` pts) | `Hard-60` (`60` pts) | Security Audit (`8` prompts) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **[`MaxQ-Fractal`](https://huggingface.co/Synchopate/DeepSeek-V4-Flash-0731-MaxQ-Fractal-64GB-CPU-GGUF) (`56.93 GiB`, Aligned Default)** | `56.93 GiB` | `59.13 GiB` | **`9.3896 ± 0.40764`** | **`0.0000 nats/tok` (`100%` Ref)** | `70.0 / 80.0` *(layout)* | **`40.0 / 60.0`** | `7 / 8` |
| **[`Abliterated-Fractal`](https://huggingface.co/Synchopate/DeepSeek-V4-Flash-0731-Abliterated-Fractal-64GB-CPU-GGUF) (`56.93 GiB`, Opt-In)** | `56.93 GiB` | `59.13 GiB` | **`11.3944 ± 0.56500`** | **`1.3444 nats/tok (50.0% Top-1)`** | **`70.0 / 80.0`** *(`30/30` Tools)* | **`50.0 / 60.0`** | `8 / 8` |
| **`LUT-VQ3-64` (`56.40 GiB` Shipped Hybrid)** | `56.40 GiB` | `58.31–58.61 GiB` | **`12.2345 ± 0.59759`** | **`0.8302 nats/tok (75.0% Top-1)`** | `64.0 / 80.0` *(`70/80` `--aggressive`)* | `30.0 / 60.0` | `8 / 8` |
| **`Fractal-64` (`56.57 GiB` Cross-Platform)** | `56.57 GiB` | `58.20 GiB` | *(transcoded `Q5_K` shexp)* | — | `64.0 / 80.0` | `20.0 / 60.0` | `8 / 8` |

---

## 4. Reproducibility Commands

```bash
# 1. Run Python pack/unpack and AVX-512 golden-vector unit tests
python3 -m unittest discover -s vq3 -p "test_*.py" -v

# 2. Compile and run the standalone AVX-512 VBMI kernel accuracy test
gcc -O3 -march=native -mavx512f -mavx512bw -mavx512vbmi -mf16c -mbmi2 \
  vq3/test_vq3_ggml_kernels.c -lm -o vq3/test_vq3_ggml_kernels && ./vq3/test_vq3_ggml_kernels

# 3. Build single-pass MaxQ-Fractal (aligned default or opt-in abliterated) & Nemotron VQ3-64 GGUFs
python3 scripts/build_dsv4_maxq_singlepass.py --base-model deepseek-ai/DeepSeek-V4-Flash --output-prefix ./DeepSeek-V4-Flash-0731-MaxQ-Fractal
python3 scripts/build_nemotron_vq3.py --fractal-gguf <Fractal-64.gguf> --bf16-gguf <BF16.gguf> --output NVIDIA-Nemotron-3.5-30B-A3B-VQ3-64.gguf
python3 scripts/build_nemotron_abliterated_vq3.py --output NVIDIA-Nemotron-3.5-30B-A3B-Abliterated-VQ3-64.gguf

# 4. Run WikiText-2 Perplexity + 2-Pass Logit KLD against committed reference logit snapshots
python3 benchmarks/eval_perplexity_and_kld.py \
  --llama-perplexity ./workspace/bin/llama-perplexity \
  --model ./workspace/models/dsv4-maxq-fractal/DeepSeek-V4-Flash-0731-MaxQ-Fractal-00001-of-00003.gguf \
  --ctx-size 512 --chunks 16 \
  --url http://127.0.0.1:8080 \
  --ref-logits benchmarks/results/dsv4_maxq_ref_logits.json \
  --save-json benchmarks/results/cont_dsv4_candidate.json

python3 benchmarks/eval_perplexity_and_kld.py \
  --llama-perplexity ./workspace/bin/llama-perplexity \
  --model ./workspace/models/NVIDIA-Nemotron-3.5-30B-A3B-VQ3-64.gguf \
  --ctx-size 512 --chunks 24 \
  --url http://127.0.0.1:8080 \
  --ref-logits benchmarks/results/nemotron_fractal_ref_logits.json \
  --save-json benchmarks/results/kld_nemotron_vq3_vs_fractal.json

# 5. Run the 100-Point Functional Gate + Extended Hard-60 & Dual-Use Refusal Audit
python3 benchmarks/eval_quality_and_tools.py \
  --url http://127.0.0.1:8787 \
  --label Nemotron-VQ3-64-13.30GiB \
  --compare benchmarks/results/eval_fractal_64_bf16.json \
  --save-json benchmarks/results/eval_nemotron_vq3_64.json

python3 benchmarks/eval_hard60_and_refusal.py \
  --url http://127.0.0.1:8080 \
  --label DeepSeek-V4-MaxQ-Fractal-Aligned \
  --save-json benchmarks/results/hard60_dsv4_maxq_fractal.json
```
