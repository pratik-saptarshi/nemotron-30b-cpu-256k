# Quality, Tool-Use, and Signal Precision Report: `LUT-VQ3-64` (3.00 bpw), `Fractal-64` (3.75 bpw), & Donor-Free Single-Pass `MaxQ-Fractal`

- **Release:** `v2.5.3`
- **Target Models:**
  - `NVIDIA-Nemotron-3.5-Lightning-30B-A3B` (`52` blocks: `23` Mamba-2 SSM + `23` Sparse MoE + `6` GQA Attention, unpruned `Top-6` active)
  - `DeepSeek-V4-Flash-0731` (`284.33B` total / `13.17B` active, `43` MLA + `256+1s` MoE layers, unpruned `Top-6` active)
- **Testbed:** AMD Ryzen 9 9955HX (`16` Zen 5 cores, `64 MB` dual-CCD L3 cache), DDR5-5600, Linux, Transparent HugePages (`madvise`).
- **Evaluation Suites:**
  1. [`benchmarks/eval_perplexity_and_kld.py`](eval_perplexity_and_kld.py): Teacher-forced Perplexity on uncompressed **`WikiText-2` (`wiki.test.raw`)** (`12,288` tokens on `30B` including the `58.84 GiB` True `BF16` master; `8,192` tokens across all four `284B` checkpoints) + Cross-Vendor **`AtomicChat/dsv4-eval-artifacts` (`wiki-alt.txt`, `-c 5632 -b 5632`)** across **all `51` chunks (`143,565` scored positions)** and **chunks `1..12` (`33,780` scored positions)** + **32-probe step-0 and prefix-matched multi-step rollout Top-64 Logit KL-Divergence** (`nats/tok` at `T=1.0`).
  2. [`benchmarks/eval_quality_and_tools.py`](eval_quality_and_tools.py): `30` pts Agentic Tool-Calling, `30` pts Executable Python Code Gen, `20` pts Long-Context Needle & Arithmetic (`80` quality pts) + `20` pts Throughput & Memory Telemetry, supplemented by isolated pure `N=1` unpruned `Top-6` telemetry ([`results/telemetry_n1_dsv4_maxq_v253.json`](results/telemetry_n1_dsv4_maxq_v253.json)).
  3. [`benchmarks/eval_hard60_and_refusal.py`](eval_hard60_and_refusal.py): `8`-prompt dual-use technical security refusal audit (`--skip-hard60`; small `n=6` `Hard-60` retired from headline comparisons due to discrete binomial variance).

---

## 1. Format Specification & Checkpoint Topology

`llama-modchip v2.5.3` pairs two block-64 CPU quantization formats:

- **`Fractal-64` (`GGML_TYPE_Q2_0 = 42`, `30 bytes / 64 weights = 3.75 bpw`):**
  - `2 bytes` `FP16` super-block scale $d$, `4 bytes` storing eight 4-bit unsigned sub-scales ($m_b = 4 + u_b \in \{4..19\}$ for eight 8-weight sub-blocks), and `24 bytes` (`qs[16]` + `qh[8]`) of 3-bit indices into the symmetric non-linear dyadic codebook `{-14, -8, -4, -1, +1, +4, +8, +14}`.
  - Achieves **`17.75 dB` SQNR (`99.15%` cosine similarity)** directly on uncompressed `BF16` MoE weights (`+1.45 dB` over `ggml` `Q3_K`), decoded via single-cycle `VPSHUFB` lookup and sub-scale fusion in `VPMADDWD` (note: enum slot `42` repurposes upstream `GGML_TYPE_Q2_0 = 42` [`18 B / 32 W`] in-place).
- **`LUT-VQ3-64` (`GGML_TYPE_VQ3_0 = 43`, `24 bytes / 64 weights = 3.00 bpw`):**
  - `2 bytes` `FP16` super-block scale $d$ (scaled by `GGML_VQ3_G = 8.6373016238e-3f`) and `22 bytes` (`qs[22]`):
    - `qs[0..19]` bits `0..6`: `20` 7-bit 3D Lloyd-Max indices (`60` weights at stride 22: `j, j+22, j+44`).
    - `qs[20..21]` bits `0..5`: `2` 6-bit 2D tail indices (`4` weights at `20, 42` and `21, 43`, half-table selected by bit 6).
    - `s24` (`24 bits` packed in bit 7 of `qs[0..21]` and bit 6 of `qs[20..21]`): **eight 3-bit unsigned sub-scales** ($m_b = 2 + u_b \in \{2..9\}$ for eight 8-weight sub-blocks).
  - Achieves **`13.57 dB` SQNR (`97.78%` cosine similarity)** directly on `BF16` weights (`+3.42 dB` over `ggml` `Q2_K`), decoded in registers via three AVX-512 VBMI `VPERMI2B` (`_mm512_mask2_permutex2var_epi8`) lookups against a 384-byte codebook held in 6 ZMM registers (with portable `AVX2` fallback).

### Published Checkpoint Variants (`v2.5.3`)

1. **`NVIDIA-Nemotron-3.5-30B-A3B` (`16–24 GB+` RAM, [`OpenMDW-1.1`](https://openmdw.ai/license/1-1/)):**
   - **Aligned Default `VQ3-64` (`13.30 GiB`, `14,282,559,040` B):** Boundary MoE layers (`blk.1, 3, 49, 51`) and all 23 `ffn_down_exps` in `Fractal-64` (`3.75 bpw`); 19 interior `blk.6..47.ffn_up_exps` (pre-`ReLU²`) in `LUT-VQ3-64` (`3.00 bpw`). Passing `--all-up-exps` produces the `13.08 GiB` 23-layer ablation ([`results/eval_nemotron_vq3_64.json`](results/eval_nemotron_vq3_64.json), `100.0 / 100.0` at `24.18 tok/s`, `70.19%` `Stockfish-PVS` acceptance, `16.04 GiB` `VmRSS`).
   - **Aligned `Fractal-64` (`14.36 GiB`, `15,420,034,624` B):** All 23 routed MoE layers in `Fractal-64` (`3.75 bpw`), recommended for non-VBMI `AVX2` CPUs and Apple Silicon Metal (`M1–M5`).
   - **Opt-In Uncensored `Abliterated-VQ3-64` (`13.30 GiB`, `14,282,559,488` B):** Built via [`scripts/build_nemotron_abliterated_vq3.py`](../scripts/build_nemotron_abliterated_vq3.py), which re-quantizes all `76` residual-writing GGUF tensors modified by `HangGlidersRule/Darkstar-Nemotron-3.5-Lightning-30B-A3B-Abliterated-BF16` (`2,997` raw matrices: `token_embd`, `23× ssm_out`, `6× attn_output`, `23× ffn_down_shexp`, `23× 128-expert ffn_down_exps`) in a single pass from `BF16` into `Q4_0`/`Q8_0`/`Fractal-64`, while copying the `257` unedited tensors (including all `19` `LUT-VQ3-64` `ffn_up_exps`, SSM states, routers, norms, and `output`) byte-for-byte from aligned `VQ3-64`.
2. **`DeepSeek-V4-Flash-0731 284B` (`64 GB+` RAM, MIT License; Shard-1 Metadata Audit in [`results/gguf_shard1_metadata_audit.txt`](results/gguf_shard1_metadata_audit.txt)):**
   - **Aligned Default [`MaxQ-Fractal`](https://huggingface.co/Synchopate/DeepSeek-V4-Flash-0731-MaxQ-Fractal-64GB-CPU-GGUF) (`56.93 GiB`, `61,129,364,160` B, `1.72 bpw`; Shards `1..3` SHA256 `0ae64076...` / `155fc6c2...` / `15f2e256...`):** Donor-free single-pass build across all `1,328 / 1,328` weight tensors directly from `deepseek-ai/DeepSeek-V4-Flash-0731` (`7872f01b1d1fe23eabc4c98b48bffcef5a386062`, [`scripts/build_dsv4_maxq_singlepass.py`](../scripts/build_dsv4_maxq_singlepass.py) `--mode aligned`) with **174 dense 100%-active bottleneck tensors** (`173 × Q6_K + 1 × Q8_0`: `token_embd`, `output`, `43× attn_q_a`, `43× ffn_{gate,up,down}_shexp`), **317 MLA & Lightning Indexer tensors** in `Fractal-64` (`3.75 bpw`), and all **129 3D routed-MoE tensors** quantized single-pass from `0731` `MXFP4` weighted by `@antirez`'s `1.5M`-token `DeepSeek-V4-Flash-chat-v2-routed-moe-ds4-1p5m.dat` (`antirez/deepseek-v4-gguf @ f71f23d552d6`, committed `2026-05-12` from April `DeepSeek-V4-Flash` `chat-v2` activations; Shard-1 records `entries_count = 129` and nominal `chunks_count = 1`). Eliminating `v2.5.2`'s donor `ffn_down_exps` transcoding lifts the 80-point Functional Gate from `64.0 / 80` to **`70.0 / 80` (`30/30` Tools; `89.56 / 100` under `--target-tps 9.0`)** and reduces `wiki-alt.txt` Chunk-12 PPL from `7.5133` to **`7.3180`** (`51`-chunk `Final estimate: PPL = 7.2977 ± 0.0504`; [`results/gate_MaxQ-Fractal-0731-56.93GiB.json`](results/gate_MaxQ-Fractal-0731-56.93GiB.json), [`results/ppl_wikitext2_dsv4_maxq_fractal.json`](results/ppl_wikitext2_dsv4_maxq_fractal.json), [`results/atomicchat_wikialt_ctx5632_51c_dsv4_maxq_fractal.log`](results/atomicchat_wikialt_ctx5632_51c_dsv4_maxq_fractal.log)).
   - **Opt-In Uncensored [`Abliterated-Fractal`](https://huggingface.co/Synchopate/DeepSeek-V4-Flash-0731-Abliterated-Fractal-64GB-CPU-GGUF) (`56.93 GiB`, `61,129,364,224` B, `1.72 bpw`):** Identical single-pass `1,328`-tensor recipe applied to `prem-research/DeepSeek-V4-Flash-0731-abliterated` (`457210de3a9d`, ARA orthogonal projection across `o_b_proj`, shared `down_proj`, and top-8 routed `down_proj` experts in layers `17..36`; [`results/gate_Abliterated-Fractal-MaxQ-56.93GiB.json`](results/gate_Abliterated-Fractal-MaxQ-56.93GiB.json)). Shards 2 & 3 (`6e3f499c...`, `0063f3a9...`) are the `v2.5.1` single-pass build (with Shard-1 metadata cleaned in `v2.5.3`), verified **byte-identical** to `HEAD`'s `--mode aligned` build on unedited control experts (e.g., `blk.20.ffn_down_exps` experts `0, 1`) and differing on `prem-research`'s abliterated experts (`{53, 61, 70, 149, 156, 187, 210, 219}` in layer `20`). Note that `gate_Abliterated-Fractal-MaxQ-56.93GiB.json` (`70.0 / 80` Functional, `8.75 tok/s`, `59.13 GiB` `VmRSS`) was recorded in `v2.5.1` with default `--target-tps 15.0` (`81.67/100`), corresponding to **`89.44 / 100`** on the `--target-tps 9.0` rubric.
   - **Compact Aligned `VQ3` (`56.40 GiB` shipped [`60,558,856,960` B] / `56.07 GiB` `--aggressive`):** 100% MLA in `Fractal-64` + 68 interior `shexp` in `LUT-VQ3-64`, retaining calibrated `gate_exps`/`up_exps` donor tensors from `Unsloth` (`UD-IQ1_S @ fbbb5b93fb787c21338159b0af3318bb3f4d9768`) paired with `SwiGLU-H2` on `ffn_down_exps` ([`results/gate_VQ3-DoubleQuant-Baseline.json`](results/gate_VQ3-DoubleQuant-Baseline.json): `64.0 / 80` Functional Gate [`84.0/100` under `--target-tps 9.0`], `9.15 tok/s` served / `9.05 tok/s` raw `N=1`; [`results/atomicchat_wikialt_ctx5632_12c_dsv4_vq3.log`](results/atomicchat_wikialt_ctx5632_12c_dsv4_vq3.log): `7.6579 ± 0.1077` `wiki-alt.txt` `PPL`). Note that [`results/eval_dsv4_vq3_64.json`](results/eval_dsv4_vq3_64.json) (`70.0 / 80` Functional, `10.74 tok/s`) records the `--aggressive` `56.07 GiB` ablation **with `Stockfish-PVS` speculation active** (`67.95%` draft acceptance across `156` drafted tokens).
   - **Cross-Platform Aligned `Fractal` (`56.57 GiB`, `60,739,343,104` B):** Pure `Fractal-64` MLA + `Q5_K`/`Q6_K` `shexp` + `Unsloth UD-IQ1_S @ fbbb5b93fb78` / `SwiGLU-H2` routed MoE ([`results/gate_Fractal-Reference-56.57GiB.json`](results/gate_Fractal-Reference-56.57GiB.json): `64.0 / 80` Functional Gate [`84.0/100` under `--target-tps 9.0`], `9.55 tok/s` served; [`results/ppl_wikitext2_dsv4_fractal.json`](results/ppl_wikitext2_dsv4_fractal.json): `11.5288 ± 0.5563` `WikiText-2` `PPL`).

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

## 3. Continuous `WikiText-2` & Cross-Vendor `AtomicChat` `wiki-alt.txt` Perplexity + 32-Probe / Multi-Step Logit KLD

Discrete functional gates (`80` quality pts across JSON tool-calling, unit-tested code generation, and long-context needle/math) verify structural non-collapse, while continuous teacher-forced perplexity and Top-64 Logit KL-divergence resolve fine quantization differences across the full model graph:
1. **Standard Uncompressed `WikiText-2` (`wiki.test.raw`) Teacher-Forced Perplexity (`PPL ± σ`)**: Auto-downloaded from `cosmo.zip` (`wiki.test.raw`, `1,290,590` bytes, zero `<unk>` pre-masking), evaluated at `--ctx-size 512` across `24` chunks (`12,288` tokens) on `Nemotron-30B` (anchored directly against the `58.84 GiB` True `BF16` master) and `16` chunks (`8,192` tokens) across all four `DeepSeek-V4 284B` checkpoints.
2. **Cross-Vendor `AtomicChat/dsv4-eval-artifacts` (`wiki-alt.txt`, `-c 5632 -b 5632`, Full `51` Chunks = `143,565` Scored Positions & `12`-Chunk Prefix = `33,780` Scored Positions)**: Evaluated with identical `llama-perplexity -f wiki-alt.txt -c 5632 -b 5632 --ppl-output-type 1` parameters (`first = n_ctx/2 = 2816`) against both the full 51-chunk `Final estimate: PPL` and cumulative chunk-12 lines of the vendor logs in [`AtomicChat/dsv4-eval-artifacts`](https://huggingface.co/datasets/AtomicChat/dsv4-eval-artifacts).
3. **32-Probe Step-0 & Prefix-Matched Multi-Step Rollout Top-64 Logit KL-Divergence (`nats/tok` at `T=1.0`)**: Compares output distributions (`n_probs = 64`) across 32 deterministic coding, SQL/HTTP, math, systems, physics, and MoE/MLA probes both at `step=0` (`32` unconditional probes) and across prefix-matched multi-step rollouts (`101–185` token positions where reference and candidate greedy tokens remain identical) against committed reference logit snapshots ([`results/nemotron_bf16_ref_logits.json`](results/nemotron_bf16_ref_logits.json) [`SHA256 = fd513c91...`], [`results/nemotron_fractal_ref_logits.json`](results/nemotron_fractal_ref_logits.json), [`results/nemotron_vq3_ref_logits.json`](results/nemotron_vq3_ref_logits.json), and [`results/dsv4_maxq_ref_logits.json`](results/dsv4_maxq_ref_logits.json)). **Statistical caveats:** (a) because prefix-matched rollout KLD truncates each trajectory at the first greedy token disagreement, it conditions on higher-agreement positions and biases downward relative to unconditional teacher-forced KLD—always treat 32-probe `step=0` KLD as primary; and (b) while `Nemotron-30B` KLD is anchored directly to the uncompressed `58.84 GiB` `BF16` master, `DeepSeek-V4 284B` KLD is measured relative to quantized `MaxQ-Fractal v2.5.3` (`150.75 GiB` `BF16` OOMs on `64 GB` RAM) and therefore measures **quantization-lineage divergence** rather than ground-truth quality error.

### A. `NVIDIA-Nemotron-3.5-30B-A3B` — True-`BF16` `WikiText-2` (`12,288` tok) Perplexity & Logit KLD Ladder

| `Nemotron-3.5-30B-A3B` Checkpoint | Effective MoE `bpw` | Disk Size | **`WikiText-2` `PPL ± σ` (`12,288` tok)** | **KLD vs. True `BF16` (`32`-Probe / Matched)** | **Same-Family KLD (`Step-0` / Matched)** | Functional Gate (`100` pts) | Security Audit (`8`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **True `BF16` Master** *(`fd513c91...`)* | `16.00 bpw` | `58.84 GiB` | **`6.9582 ± 0.24151`** *(Ref)* | **`0.0000 nats` (`100%` Ref)** | — | `100.0 / 100` | — |
| **`Fractal-MXFP4` *(from `Q4_0`)*** | `4.25 bpw` | `15.95 GiB` | `7.2927 ± 0.25382` *(`+4.81%` vs BF16)* | `0.1623 / 0.1204 nats` *(`80.0% / 88.4%` Top-1)* | — | `100.0 / 100` | — |
| **`Fractal-64` *(direct from `BF16`)*** | `3.75 bpw` | `14.36 GiB` | **`7.3610 ± 0.26017`** *(`+5.79%` vs BF16)* | **`0.1254 / 0.0929 nats`** *(`81.2% / 89.8%` Top-1)* | `0.0000` *(Ref)* | `100.0 / 100` | — |
| **`VQ3-64` *(Aligned Default)*** | `3.00 / 3.75 bpw` | `13.30 GiB` | **`7.4400 ± 0.26263`** *(`+6.92%` vs BF16, `+1.07%` vs Fractal)* | **`0.1495 / 0.0944 nats`** *(`78.1% / 85.2%` Top-1)* | **`0.0523 / 0.0889 nats`** vs. `Fractal-64` | `100.0 / 100` | `8 / 8` |
| **`Abliterated-VQ3-64` *(Opt-In)*** | `3.00 / 3.75 bpw` | `13.30 GiB` | `7.7721 ± 0.28352` *(`+11.70%` vs BF16, `+4.46%` vs VQ3)* | `0.1431 / 0.1109 nats` *(`75.0% / 84.4%` Top-1)* | **`0.0308 / 0.0312 nats`** vs. `VQ3-64` *(`92.4%` Top-1, `100%` Top-5)* | *(`100.0` via `VQ3` layout)* | `8 / 8` |

*(Notes: [1] Because `Abliterated-VQ3-64` re-quantizes only the `76` `Darkstar`-edited residual-writing tensors from `BF16` and shares the remaining `257` tensors byte-for-byte with `VQ3-64`, comparing them isolates the `76`-tensor abliteration delta at **`0.0308 nats/tok` step-0 / `0.0312 nats/tok` prefix-matched (`185` pos, `92.4%` Top-1, `100.0%` Top-5)**; `WikiText-2` PPL, `BF16` KLD, same-family KLD, and the `8/8` security audit were measured directly on `Abliterated-VQ3-64.gguf`, while throughput/VmRSS/Gate reflect the identical-layout `VQ3-64` architecture. [2] `Fractal-MXFP4` shows slightly higher `32`-probe step-0 KLD vs `BF16` (`0.1623 nats`) than `BF16`-direct `VQ3-64` (`0.1495 nats`) despite lower `WikiText-2` PPL (`7.2927` vs `7.4400`) because `Fractal-MXFP4` is transcoded from an intermediate `Q4_0` trunk rather than directly from `BF16`. [3] Independent community audit `@khairullinr` in [Issue #16](https://github.com/synchopate/llama-modchip/issues/16) additionally measured `PPL = 2.083 ± 0.037` on `Q4_0` [`17.60 GiB`] and `Logit KLD = 0.0566 nats/tok` [`94.35%` Top-1] between `Fractal-64` and `Q4_0` on a Python source-code corpus.)*

### B. `DeepSeek-V4-Flash-0731 284B` — Same-Revision (`0731`) `MaxQ-Fractal v2.5.3`, `Abliterated-Fractal`, `VQ3`, & `Fractal-64`

| `DeepSeek-V4-Flash-0731 284B` Checkpoint (`64 GB` Host) | Disk Size | Anonymous `VmRSS` | **`wiki-alt.txt` `PPL ± σ` (`51`c / `12`c)** | **`WikiText-2` `PPL ± σ` (`8,192` tok)** | **Lineage KLD vs. `MaxQ v2.5.3` (`32`-Probe / Matched)** | Functional Gate (`80` pts) | Pure `N=1` / Gate Decode |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **[`MaxQ-Fractal v2.5.3`](https://huggingface.co/Synchopate/DeepSeek-V4-Flash-0731-MaxQ-Fractal-64GB-CPU-GGUF) *(Aligned Default)*** | `56.93 GiB` | `57.54–58.13 GiB` | **`7.2977 ± 0.0504`** / **`7.3180 ± 0.1039`** | **`11.6139 ± 0.58453`** | **`0.0000 nats` (`100%` Ref)** | **`70.0 / 80.0`** *(`30/30` Tools)* | **`8.66` `N=1` / `8.80 tok/s` (`78.7%` acc)** |
| **[`Abliterated-Fractal`](https://huggingface.co/Synchopate/DeepSeek-V4-Flash-0731-Abliterated-Fractal-64GB-CPU-GGUF) *(Opt-In Twin)*** | `56.93 GiB` | `59.13 GiB` | — | **`11.3944 ± 0.56500`** | **`0.0959 / 0.0512 nats`** *(`75.0% / 90.7%` Top-1)* | **`70.0 / 80.0`** *(`30/30` Tools)* | `8.75 tok/s` *(`64.4%` Gate acc, `v2.5.1`)* |
| **`Fractal-64` *(Cross-Platform)* ** | `56.57 GiB` | `58.20 GiB` | — | **`11.5288 ± 0.55630`** | `0.8437 / 0.4373 nats` *(`56.2% / 74.3%` Top-1)* | `64.0 / 80.0` *(`24/30` Tools)* | `9.06–9.55 tok/s` |
| **`LUT-VQ3-64` *(Compact Hybrid)*** | `56.40 GiB` | `58.31–58.61 GiB` | — / `7.6579 ± 0.1077` | `12.2345 ± 0.59759` | `0.8409 / 0.4397 nats` *(`59.4% / 75.2%` Top-1)* | `64.0 / 80.0` *(`70/80` `--aggressive`)* | `9.05–9.15 tok/s` |

*(Technical Notes on Rubric, `WikiText-2`, & Lineage KLD: [1] **Uniform Gate Rubric:** All 284B rows above report the 80-point Functional Quality score; adding the 20-point speed/RAM rubric under a common `--target-tps 9.0` target yields **`89.56 / 100`** for `MaxQ-Fractal`, **`89.44 / 100`** for `Abliterated-Fractal` [whose `v2.5.1` JSON used default `--target-tps 15.0` = `81.67/100`], and **`84.0 / 100`** for `Fractal-64` & `LUT-VQ3-64`. [2] **`WikiText-2` vs. Tool-Calling:** On `8,192`-token `WikiText-2`, `MaxQ-Fractal` [`11.6139 ± 0.5845`] **does not beat** `Fractal-64` [`11.5288 ± 0.5563`; `+0.085` PPL is `< 0.15σ` within chunk variance], whereas `MaxQ-Fractal`'s `174 × Q6_K/Q8_0` dense trunk + `@antirez` `chat-v2` calibration improves multi-turn JSON Tool-Calling from `24/30` to `30/30` [`+6.0` pts] and beats `LUT-VQ3-64` on both `WikiText-2` [`-5.07%`] and `wiki-alt.txt` `12`c [`7.3180` vs `7.6579`]. [3] **Lineage KLD:** `Fractal-64` shows `0.8437 nats` step-0 KLD relative to `MaxQ` despite `11.5288` `WikiText-2` PPL because `MaxQ`/`Abliterated` share `@antirez`'s single-pass `chat-v2` `imatrix` while `Fractal`/`VQ3` share `Unsloth UD-IQ1_S @ fbbb5b93fb78` + `SwiGLU-H2`; KLD against a quantized reference reflects quantization lineage rather than absolute quality. All four checkpoints score **`8 / 8`** on the dual-use technical security audit.)*

### C. Cross-Vendor `AtomicChat/dsv4-eval-artifacts` (`wiki-alt.txt`, `-c 5632 -b 5632`) Full 51-Chunk & Chunk-12 Comparison

All entries below evaluate `DeepSeek-V4-Flash-0731` (`284.33B`) on `AtomicChat`'s held-out `wiki-alt.txt` corpus (`1,287,656` bytes) at `n_ctx = 5632, batch_size = 5632` (`2,815` scored positions per chunk; **`33,780` scored positions across chunks `1..12`** and **`143,565` scored positions across all `51` chunks**). **Important cross-hardware and aggregation caveats:** (1) `AtomicChat` reference runs were collected on `8× RTX 5090` CUDA GPUs (`-DCMAKE_CUDA_ARCHITECTURES=120`), where [`AtomicChat/dsv4-eval-artifacts`](https://huggingface.co/datasets/AtomicChat/dsv4-eval-artifacts) explicitly documents a **`4–5%` hardware numerical gap** (`4.5381` on `RTX 5090` vs. `4.3406` on `H100` for the exact same GGUF) from Blackwell native FP4 vs. BF16-unpack matmul paths, whereas `Synchopate` checkpoints were evaluated on a single 16-core AMD Ryzen 9 9955HX AVX-512 CPU; and (2) `AtomicChat`'s summary table (`UD-IQ1_S`, `AD-IQ1_M`) reports `Mean PPL(Q)` from `--kl-divergence` against `ref5632.kld` (`37.1 GB`), whereas `Synchopate`'s `7.2977 ± 0.0504` (`12`c `7.3180 ± 0.1039`) is standalone `llama-perplexity`'s `Final estimate: PPL` (`--ppl-output-type 1`). Compressing `284.33B` parameters into `< 60 GiB` resident RAM (`1.72 bpw`, `-93.8 GiB` vs. `MXFP4_MOE-BF16`) trades **`+60.8%` `wiki-alt` perplexity vs. `150.75 GiB` `MXFP4_MOE-BF16`, `+22.2%` vs. `76.87 GiB` `Unsloth UD-IQ1_S`, and `+14.4%` vs. `65.42 GiB` `AtomicChat AD-IQ1_M`**, in exchange for being the only checkpoints below that execute **100% resident inside `64 GB` host RAM**:

| Publisher & `DeepSeek-V4-Flash-0731` GGUF | Disk Size (`GiB` / `GB`) | Resident in `64 GB` Host RAM? | **Chunk-12 `wiki-alt.txt` `PPL ± σ` (`33,780` scored)** | **Full 51-Chunk `wiki-alt.txt` `PPL` (`143,565` scored)** |
| :--- | :---: | :---: | :---: | :---: |
| `bullerwins` `MXFP4_MOE-BF16` *(Full Reference)* | `150.75 GiB` (`161.87 GB`) | No (`192 GB+`) | `4.4474 ± 0.0587` | **`4.5381 ± 0.0292`** *(`Mean PPL(Q)` on `8×5090`; `4.3406` on `H100`)* |
| `prometheusAIR` `IQ2_XXS-slim` | `78.84 GiB` (`84.65 GB`) | No (`96 GB+`) | `5.5360 ± 0.0754` | `5.5165 ± 0.0365` *(`Mean PPL(Q)`)* |
| `antirez` `IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-0731` | `80.76 GiB` (`86.72 GB`) | No (`96 GB+`) | `5.5484 ± 0.0762` | `5.6025 ± 0.0377` *(`Mean PPL(Q)`)* |
| `unsloth` `UD-IQ1_S` | `76.87 GiB` (`82.54 GB`) | No (`96 GB+`) | — | **`5.9734`** *(`Mean PPL(Q)`)* |
| `AtomicChat` `AD-IQ1_M` | `65.42 GiB` (`70.24 GB`) | No (`>64 GiB` OOM) | — | **`6.3813`** *(`Mean PPL(Q)`)* |
| **`Synchopate` `MaxQ-Fractal-0731` (`v2.5.3`, *64 GB Default*)** | **`56.93 GiB` (`61.13 GB`)** | **Yes (`57.54–58.13 GiB` `VmRSS`)** | **`7.3180 ± 0.1039`** *(`+64.5%` vs BF16)* | **`7.2977 ± 0.0504`** *(`Final estimate` on Zen 5 CPU: `+60.8%` vs BF16, `+22.2%` vs `UD-IQ1_S`, `+14.4%` vs `AD-IQ1_M`; [`log`](results/atomicchat_wikialt_ctx5632_51c_dsv4_maxq_fractal.log))* |
| **`Synchopate` `VQ3-0731` *(64 GB Compact Hybrid)*** | **`56.40 GiB` (`60.56 GB`)** | **Yes (`58.3–58.6 GiB` `VmRSS`)** | `7.6579 ± 0.1077` *(`Final estimate` `12`c; [`log`](results/atomicchat_wikialt_ctx5632_12c_dsv4_vq3.log))* | — |
| `DevQuasar` `Q2_K` *(uncalibrated naive `Q2_K` outlier)* | `95.99 GiB` (`103.06 GB`) | No (`128 GB+`) | `9.4237 ± 0.1402` | `9.1127 ± 0.0647` *(`Mean PPL(Q)`)* |

---

## 4. Reproducibility Commands

```bash
# 1. Run Python pack/unpack and AVX-512 golden-vector unit tests
python3 -m unittest discover -s vq3 -p "test_*.py" -v

# 2. Compile and run the standalone AVX-512 VBMI kernel accuracy test
gcc -O3 -march=native -mavx512f -mavx512bw -mavx512vbmi -mf16c -mbmi2 \
  vq3/test_vq3_ggml_kernels.c -lm -o vq3/test_vq3_ggml_kernels && ./vq3/test_vq3_ggml_kernels

# 3. Build donor-free single-pass MaxQ-Fractal (pure 0731 aligned default or opt-in abliterated) & Nemotron VQ3-64 GGUFs
python3 scripts/build_dsv4_maxq_singlepass.py --mode aligned --out-prefix ./DeepSeek-V4-Flash-0731-MaxQ-Fractal
python3 scripts/build_dsv4_maxq_singlepass.py --mode abliterated --hf-repo prem-research/DeepSeek-V4-Flash-0731-abliterated --out-prefix ./DeepSeek-V4-Flash-0731-Abliterated-Fractal
python3 scripts/build_nemotron_vq3.py --fractal-gguf <Fractal-64.gguf> --bf16-gguf <BF16.gguf> --output NVIDIA-Nemotron-3.5-30B-A3B-VQ3-64.gguf
python3 scripts/build_nemotron_abliterated_vq3.py --output NVIDIA-Nemotron-3.5-30B-A3B-Abliterated-VQ3-64.gguf

# 4. Run WikiText-2 Perplexity + Full 51-Chunk AtomicChat wiki-alt.txt (ctx=5632) + 32-Probe / Multi-Step Logit KLD
python3 benchmarks/eval_perplexity_and_kld.py \
  --llama-perplexity ./workspace/bin/llama-perplexity \
  --model ./workspace/models/dsv4-maxq-fractal/DeepSeek-V4-Flash-0731-MaxQ-Fractal-00001-of-00003.gguf \
  --ctx-size 512 --chunks 16 \
  --url http://127.0.0.1:8080 \
  --ref-logits benchmarks/results/dsv4_maxq_ref_logits.json \
  --rollout-steps 8 \
  --save-json benchmarks/results/cont_dsv4_candidate.json

./workspace/bin/llama-perplexity \
  -m ./workspace/models/dsv4-maxq-fractal/DeepSeek-V4-Flash-0731-MaxQ-Fractal-00001-of-00003.gguf \
  -f wiki-alt.txt -c 5632 -b 5632 --repack -fa on -t 16 -tb 16 --ppl-output-type 1

python3 benchmarks/eval_perplexity_and_kld.py \
  --llama-perplexity ./workspace/bin/llama-perplexity \
  --model ./workspace/models/NVIDIA-Nemotron-3.5-30B-A3B-VQ3-64.gguf \
  --ctx-size 512 --chunks 24 \
  --url http://127.0.0.1:8080 \
  --ref-logits benchmarks/results/nemotron_bf16_ref_logits.json \
  --rollout-steps 8 \
  --save-json benchmarks/results/kld_nemotron_vq3_64_vs_bf16.json

# 5. Run the 100-Point Functional Gate + 8-Prompt Dual-Use Security Refusal Audit
python3 benchmarks/eval_quality_and_tools.py \
  --url http://127.0.0.1:8080 \
  --target-tps 9.0 \
  --label MaxQ-Fractal-0731-56.93GiB \
  --save-json benchmarks/results/gate_MaxQ-Fractal-0731-56.93GiB.json

python3 benchmarks/eval_hard60_and_refusal.py \
  --url http://127.0.0.1:8080 \
  --skip-hard60 \
  --label DeepSeek-V4-MaxQ-Fractal-0731 \
  --save-json benchmarks/results/refusal_dsv4_maxq_fractal.json
```
