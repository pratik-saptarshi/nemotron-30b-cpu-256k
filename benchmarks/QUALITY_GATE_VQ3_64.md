# Quality, Tool-Use, and Signal Precision Report: `LUT-VQ3-64` (3.00 bpw), `Fractal-64` (3.75 bpw), & Single-Pass `MaxQ-Fractal`

- **Release:** `v2.5.2`
- **Target Models:**
  - `NVIDIA-Nemotron-3.5-Lightning-30B-A3B` (`52` blocks: `23` Mamba-2 SSM + `23` Sparse MoE + `6` GQA Attention, unpruned `Top-6` active)
  - `DeepSeek-V4-Flash-0731` (`284.33B` total / `13.17B` active, `43` MLA + `256+1s` MoE layers, unpruned `Top-6` active)
- **Testbed:** AMD Ryzen 9 9955HX (`16` Zen 5 cores, `64 MB` dual-CCD L3 cache), DDR5-5600, Linux, Transparent HugePages (`madvise`).
- **Evaluation Suites:**
  1. [`benchmarks/eval_perplexity_and_kld.py`](eval_perplexity_and_kld.py): Teacher-forced Perplexity on uncompressed **`WikiText-2` (`wiki.test.raw`)** (`12,288` tokens on `30B`, `8,192` tokens on `284B`) + Cross-Vendor **`AtomicChat/dsv4-eval-artifacts` (`wiki-alt.txt`, `-c 5632 -b 5632`)** (`67,584` tokens / `33,780` scored positions across chunks `1..12`) + 2-pass single-host Top-64 Logit KL-Divergence (`nats/tok` at `T=1.0`).
  2. [`benchmarks/eval_quality_and_tools.py`](eval_quality_and_tools.py): `30` pts Agentic Tool-Calling, `30` pts Executable Python Code Gen, `20` pts Long-Context Needle & Arithmetic (`80` quality pts) + `20` pts Throughput & Memory Telemetry.
  3. [`benchmarks/eval_hard60_and_refusal.py`](eval_hard60_and_refusal.py): `8`-prompt dual-use technical security refusal audit (`--skip-hard60`; small `n=6` `Hard-60` retired from headline comparisons due to discrete binomial variance).

---

## 1. Format Specification & Checkpoint Topology

`llama-modchip v2.5.2` pairs two block-64 CPU quantization formats:

- **`Fractal-64` (`GGML_TYPE_Q2_0 = 42`, `30 bytes / 64 weights = 3.75 bpw`):**
  - `2 bytes` `FP16` super-block scale $d$, `4 bytes` storing eight 4-bit unsigned sub-scales ($m_b = 4 + u_b \in \{4..19\}$ for eight 8-weight sub-blocks), and `24 bytes` (`qs[16]` + `qh[8]`) of 3-bit indices into the symmetric non-linear dyadic codebook `{-14, -8, -4, -1, +1, +4, +8, +14}`.
  - Achieves **`17.75 dB` SQNR (`99.15%` cosine similarity)** directly on uncompressed `BF16` MoE weights (`+1.45 dB` over `ggml` `Q3_K`), decoded via single-cycle `VPSHUFB` lookup and sub-scale fusion in `VPMADDWD` (note: enum slot `42` repurposes upstream `GGML_TYPE_Q2_0 = 42` [`18 B / 32 W`] in-place).
- **`LUT-VQ3-64` (`GGML_TYPE_VQ3_0 = 43`, `24 bytes / 64 weights = 3.00 bpw`):**
  - `2 bytes` `FP16` super-block scale $d$ (scaled by `GGML_VQ3_G = 8.6373016238e-3f`) and `22 bytes` (`qs[22]`):
    - `qs[0..19]` bits `0..6`: `20` 7-bit 3D Lloyd-Max indices (`60` weights at stride 22: `j, j+22, j+44`).
    - `qs[20..21]` bits `0..5`: `2` 6-bit 2D tail indices (`4` weights at `20, 42` and `21, 43`, half-table selected by bit 6).
    - `s24` (`24 bits` packed in bit 7 of `qs[0..21]` and bit 6 of `qs[20..21]`): **eight 3-bit unsigned sub-scales** ($m_b = 2 + u_b \in \{2..9\}$ for eight 8-weight sub-blocks).
  - Achieves **`13.57 dB` SQNR (`97.78%` cosine similarity)** directly on `BF16` weights (`+3.42 dB` over `ggml` `Q2_K`), decoded in registers via three AVX-512 VBMI `VPERMI2B` (`_mm512_mask2_permutex2var_epi8`) lookups against a 384-byte codebook held in 6 ZMM registers (with portable `AVX2` fallback).

### Published Checkpoint Variants (`v2.5.2`)

1. **`NVIDIA-Nemotron-3.5-30B-A3B` (`16–24 GB+` RAM, NVIDIA Open Model License):**
   - **Aligned Default `VQ3-64` (`13.30 GiB`, `14,282,559,040` B):** Boundary MoE layers (`blk.1, 3, 49, 51`) and all 23 `ffn_down_exps` in `Fractal-64` (`3.75 bpw`); 19 interior `blk.6..47.ffn_up_exps` (pre-`ReLU²`) in `LUT-VQ3-64` (`3.00 bpw`). Passing `--all-up-exps` produces the `13.08 GiB` 23-layer ablation ([`results/eval_nemotron_vq3_64.json`](results/eval_nemotron_vq3_64.json), `100.0 / 100.0` at `24.18 tok/s`, `70.19%` `Stockfish-PVS` acceptance, `16.04 GiB` `VmRSS`).
   - **Aligned `Fractal-64` (`14.36 GiB`, `15,420,034,624` B):** All 23 routed MoE layers in `Fractal-64` (`3.75 bpw`), recommended for non-VBMI `AVX2` CPUs and Apple Silicon Metal (`M1–M5`).
   - **Opt-In Uncensored `Abliterated-VQ3-64` (`13.30 GiB`, `14,282,559,488` B):** Single-pass `VQ3-64 + Fractal-64` quantization of `HangGlidersRule/Darkstar-Nemotron-3.5-Lightning-30B-A3B-Abliterated-BF16` ([`scripts/build_nemotron_abliterated_vq3.py`](../scripts/build_nemotron_abliterated_vq3.py)).
2. **`DeepSeek-V4-Flash-0731 284B` (`64 GB+` RAM, MIT License):**
   - **Aligned Default [`MaxQ-Fractal`](https://huggingface.co/Synchopate/DeepSeek-V4-Flash-0731-MaxQ-Fractal-64GB-CPU-GGUF) (`56.93 GiB`, `61,129,363,936` B):** Built single-pass cleanly from `deepseek-ai/DeepSeek-V4-Flash-0731` (`7872f01b1d1fe23eabc4c98b48bffcef5a386062`, [`scripts/build_dsv4_maxq_singlepass.py`](../scripts/build_dsv4_maxq_singlepass.py)) with **174 dense 100%-active bottleneck tensors** (`173 × Q6_K + 1 × Q8_0`: `token_embd`, `output`, `43× attn_q_a`, `43× ffn_{gate,up,down}_shexp`), **317 MLA & Lightning Indexer tensors** in `Fractal-64` (`3.75 bpw`), and all **129 3D routed MoE tensors** quantized single-pass from `0731` `MXFP4` with `@antirez`'s `1.5M`-token `imatrix` ([`results/gate_MaxQ-Fractal-0731-56.93GiB.json`](results/gate_MaxQ-Fractal-0731-56.93GiB.json), [`results/ppl_wikitext2_dsv4_maxq_fractal.json`](results/ppl_wikitext2_dsv4_maxq_fractal.json), [`results/atomicchat_wikialt_ctx5632_12c_dsv4_maxq_fractal.log`](results/atomicchat_wikialt_ctx5632_12c_dsv4_maxq_fractal.log)).
   - **Opt-In Uncensored [`Abliterated-Fractal`](https://huggingface.co/Synchopate/DeepSeek-V4-Flash-0731-Abliterated-Fractal-64GB-CPU-GGUF) (`56.93 GiB`, `61,129,364,000` B):** Identical single-pass `174 × Q6_K/Q8_0` + `Fractal-64` recipe applied to `prem-research/DeepSeek-V4-Flash-0731-abliterated` (`457210de3a9d`, which modifies `o_b_proj`, shared `down_proj`, and top-8 routed experts in layers `17..36` via ARA orthogonal projection; [`results/gate_Abliterated-Fractal-MaxQ-56.93GiB.json`](results/gate_Abliterated-Fractal-MaxQ-56.93GiB.json)).
   - **Compact Aligned `VQ3` (`56.40 GiB` shipped / `56.07 GiB` `--aggressive`):** 100% MLA in `Fractal-64` + 68 interior `shexp` in `LUT-VQ3-64` ([`results/gate_VQ3-DoubleQuant-Baseline.json`](results/gate_VQ3-DoubleQuant-Baseline.json): `64.0 / 80` Gate, `9.15 tok/s` served / `9.05 tok/s` raw `N=1`; [`results/atomicchat_wikialt_ctx5632_12c_dsv4_vq3.log`](results/atomicchat_wikialt_ctx5632_12c_dsv4_vq3.log): `7.6579 ± 0.1077` `wiki-alt.txt` `PPL`). Note that [`results/eval_dsv4_vq3_64.json`](results/eval_dsv4_vq3_64.json) (`70.0 / 80` Gate, `10.74 tok/s`) records the `--aggressive` `56.07 GiB` ablation **with `Stockfish-PVS` speculation active** (`67.95%` draft acceptance across `156` drafted tokens), whereas raw `N=1` unpruned `Top-6` decode without speculation is `9.05 tok/s`.
   - **Cross-Platform Aligned `Fractal` (`56.57 GiB`):** Pure `Fractal-64` MLA + `Q5_K`/`Q6_K` `shexp` ([`results/gate_Fractal-Reference-56.57GiB.json`](results/gate_Fractal-Reference-56.57GiB.json): `64.0 / 80` Gate, `9.55 tok/s` served).

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

## 3. Continuous `WikiText-2` & Cross-Vendor `AtomicChat` `wiki-alt.txt` Perplexity + 2-Pass Logit KLD

Discrete functional gates (`80` quality pts across JSON tool-calling, unit-tested code generation, and long-context needle/math) verify structural non-collapse, while continuous teacher-forced perplexity and Top-64 Logit KL-divergence resolve fine quantization differences across the full model graph:
1. **Standard Uncompressed `WikiText-2` (`wiki.test.raw`) Teacher-Forced Perplexity (`PPL ± σ`)**: Auto-downloaded from `cosmo.zip` (`wiki.test.raw`, `1,290,590` bytes, zero `<unk>` pre-masking), evaluated at `--ctx-size 512` across `24` chunks (`12,288` tokens) on `Nemotron-30B` and `16` chunks (`8,192` tokens) on `DeepSeek-V4 284B`.
2. **Cross-Vendor `AtomicChat/dsv4-eval-artifacts` (`wiki-alt.txt`, `-c 5632 -b 5632`, `12` Chunks = `67,584` Tokens / `33,780` Scored Positions)**: Evaluated with identical `llama-perplexity -f wiki-alt.txt -c 5632 -b 5632 --ppl-output-type 1 --chunks 12` parameters (`first = n_ctx/2 = 2816`) against the cumulative chunk-12 lines of the unedited vendor logs in [`AtomicChat/dsv4-eval-artifacts`](https://huggingface.co/datasets/AtomicChat/dsv4-eval-artifacts).
3. **Two-Pass Single-Host Top-64 Logit KL-Divergence (`nats/tok` at `T=1.0`)**: Compares output distributions (`n_probs = 64`) across 12 deterministic C++20, Python, SQL/HTTP, number-theory, and MoE/MLA probes against committed reference logit snapshots ([`results/dsv4_maxq_ref_logits.json`](results/dsv4_maxq_ref_logits.json) and [`results/nemotron_fractal_ref_logits.json`](results/nemotron_fractal_ref_logits.json)).

### A. `NVIDIA-Nemotron-3.5-30B-A3B` — `WikiText-2` (`12,288` tok) Perplexity & Logit KLD

| `Nemotron-3.5-30B-A3B` Checkpoint | Effective MoE `bpw` | Disk Size | **`WikiText-2` `PPL ± σ` (`12,288` tok)** | **2-Pass Logit KLD vs. `Fractal-64`** | Functional Gate (`100` pts) | Security Audit (`8` prompts) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Fractal-MXFP4` (`15.95 GiB`, Aligned)** | `4.25 bpw` | `15.95 GiB` | **`7.2927 ± 0.25382`** | — | `100.0 / 100.0` | — |
| **`Fractal-64` (`14.36 GiB`, Aligned)** | `3.75 bpw` | `14.36 GiB` | **`7.3610 ± 0.26017`** *(`+0.94%` vs MXFP4)* | **`0.0000 nats/tok` (`100%` Ref)** | `100.0 / 100.0` | — |
| **`VQ3-64` (`13.30 GiB`, Aligned Default)** | `3.00 / 3.75 bpw` | `13.30 GiB` | **`7.4400 ± 0.26263`** *(`+1.07%` vs Fractal)* | **`0.0561 nats/tok (83.3% Top-1)`** | `100.0 / 100.0` | `8 / 8` |
| **`Abliterated-VQ3-64` (`13.30 GiB`, Opt-In)** | `3.00 / 3.75 bpw` | `13.30 GiB` | **`7.7721 ± 0.28352`** *(`+4.46%` vs Aligned VQ3)* | **`0.1357 nats/tok (75.0% Top-1)`** | `100.0 / 100.0` | `8 / 8` |

*(Note: Independent community audit `@khairullinr` in [Issue #16](https://github.com/synchopate/llama-modchip/issues/16) additionally measured `PPL = 2.086 ± 0.037` on `Q4_0` and `Logit KLD = 0.0566 nats/tok` (`94.35%` Top-1 agreement) between `Fractal-64` and `Q4_0` on a deterministic Python source-code corpus.)*

### B. `DeepSeek-V4-Flash-0731 284B` — Same-Revision (`0731`) `MaxQ-Fractal`, `VQ3`, `Fractal`, & `Abliterated-Fractal`

| `DeepSeek-V4-Flash-0731 284B` Checkpoint (`64 GB` Host) | Disk Size | Anonymous `VmRSS` | **`wiki-alt.txt` `PPL ± σ` (`ctx=5632`, `12`c)** | **`WikiText-2` `PPL ± σ` (`8,192` tok)** | **2-Pass Logit KLD vs. `MaxQ-0731`** | Functional Gate (`80` pts) | Security Audit (`8` prompts) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **[`MaxQ-Fractal`](https://huggingface.co/Synchopate/DeepSeek-V4-Flash-0731-MaxQ-Fractal-64GB-CPU-GGUF) (`56.93 GiB`, Aligned Default)** | `56.93 GiB` | `57.54–59.13 GiB` | **`7.5133 ± 0.10541`** | **`11.7434 ± 0.57170`** | **`0.0000 nats/tok` (`100%` Ref)** | `64.0 / 80.0` *(`84.0/100` Total)* | `8 / 8` |
| **`LUT-VQ3-64` (`56.40 GiB` Shipped Hybrid)** | `56.40 GiB` | `58.31–58.61 GiB` | **`7.6579 ± 0.10771`** *(`+1.92%`)* | **`12.2345 ± 0.59759`** *(`+4.18%`)* | **`0.3146 nats/tok (83.3% Top-1, 100% Top-5)`** | `64.0 / 80.0` *(`70/80` `--aggressive`)* | `8 / 8` |
| **[`Abliterated-Fractal`](https://huggingface.co/Synchopate/DeepSeek-V4-Flash-0731-Abliterated-Fractal-64GB-CPU-GGUF) (`56.93 GiB`, Opt-In)** | `56.93 GiB` | `59.13 GiB` | — | **`11.3944 ± 0.56500`** | **`1.0219 nats/tok (58.3% Top-1, 91.7% Top-5)`** | `70.0 / 80.0` *(`30/30` Tools via Gateway)* | `8 / 8` |
| **`Fractal-64` (`56.57 GiB` Cross-Platform)** | `56.57 GiB` | `58.20 GiB` | *(transcoded `Q5_K` shexp)* | — | — | `64.0 / 80.0` | `8 / 8` |

### C. Cross-Vendor `AtomicChat/dsv4-eval-artifacts` (`wiki-alt.txt`, `-c 5632 -b 5632`) Apples-to-Apples Comparison

All entries below evaluate `DeepSeek-V4-Flash-0731` (`284.33B`) on `AtomicChat`'s held-out `wiki-alt.txt` corpus (`1,287,656` bytes) at `n_ctx = 5632, batch_size = 5632` (`2,815` scored positions per chunk; `33,780` scored positions across chunks `1..12`):

| Publisher & `DeepSeek-V4-Flash-0731` GGUF | Disk Size (`GiB` / `GB`) | Resident in `64 GB` Host RAM? | **Chunk-12 `wiki-alt.txt` `PPL ± σ` (`33,780` scored)** | 51-Chunk `PPL ± σ` (`AtomicChat` Logs) |
| :--- | :---: | :---: | :---: | :---: |
| `bullerwins` `MXFP4_MOE-BF16` *(Full Reference)* | `150.75 GiB` (`161.87 GB`) | No (`192 GB+`) | **`4.4474 ± 0.0587`** | `4.5381 ± 0.0292` |
| `prometheusAIR` `IQ2_XXS-slim` | `78.84 GiB` (`84.65 GB`) | No (`96 GB+`) | `5.5360 ± 0.0754` | `5.5165 ± 0.0365` |
| `antirez` `IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-0731` | `80.76 GiB` (`86.72 GB`) | No (`96 GB+`) | `5.5484 ± 0.0762` | `5.6025 ± 0.0377` |
| `unsloth` `UD-IQ1_S` | `76.87 GiB` (`82.54 GB`) | No (`96 GB+`) | — | `5.9734` |
| `AtomicChat` `AD-IQ1_M` | `65.42 GiB` (`70.24 GB`) | No (`>64 GiB` OOM) | — | `6.3813` |
| **`Synchopate` `MaxQ-Fractal-0731` *(64 GB Default)*** | **`56.93 GiB` (`61.13 GB`)** | **Yes (`57.5–59.1 GiB` `VmRSS`)** | **`7.5133 ± 0.1054`** | *(raw log in [`results/`](results/atomicchat_wikialt_ctx5632_12c_dsv4_maxq_fractal.log))* |
| **`Synchopate` `VQ3-0731` *(64 GB Compact Hybrid)*** | **`56.40 GiB` (`60.56 GB`)** | **Yes (`58.3–58.6 GiB` `VmRSS`)** | **`7.6579 ± 0.1077`** | *(raw log in [`results/`](results/atomicchat_wikialt_ctx5632_12c_dsv4_vq3.log))* |
| `DevQuasar` `Q2_K` | `95.99 GiB` (`103.06 GB`) | No (`128 GB+`) | `9.4237 ± 0.1402` | `9.1127 ± 0.0647` |

---

## 4. Reproducibility Commands

```bash
# 1. Run Python pack/unpack and AVX-512 golden-vector unit tests
python3 -m unittest discover -s vq3 -p "test_*.py" -v

# 2. Compile and run the standalone AVX-512 VBMI kernel accuracy test
gcc -O3 -march=native -mavx512f -mavx512bw -mavx512vbmi -mf16c -mbmi2 \
  vq3/test_vq3_ggml_kernels.c -lm -o vq3/test_vq3_ggml_kernels && ./vq3/test_vq3_ggml_kernels

# 3. Build single-pass MaxQ-Fractal (pure 0731 aligned default or opt-in abliterated) & Nemotron VQ3-64 GGUFs
python3 scripts/build_dsv4_maxq_singlepass.py --mode aligned --output-prefix ./DeepSeek-V4-Flash-0731-MaxQ-Fractal
python3 scripts/build_nemotron_vq3.py --fractal-gguf <Fractal-64.gguf> --bf16-gguf <BF16.gguf> --output NVIDIA-Nemotron-3.5-30B-A3B-VQ3-64.gguf
python3 scripts/build_nemotron_abliterated_vq3.py --output NVIDIA-Nemotron-3.5-30B-A3B-Abliterated-VQ3-64.gguf

# 4. Run WikiText-2 Perplexity + AtomicChat wiki-alt.txt (ctx=5632) + 2-Pass Logit KLD
python3 benchmarks/eval_perplexity_and_kld.py \
  --llama-perplexity ./workspace/bin/llama-perplexity \
  --model ./workspace/models/dsv4-maxq-fractal/DeepSeek-V4-Flash-0731-MaxQ-Fractal-00001-of-00003.gguf \
  --ctx-size 512 --chunks 16 \
  --url http://127.0.0.1:8080 \
  --ref-logits benchmarks/results/dsv4_maxq_ref_logits.json \
  --save-json benchmarks/results/cont_dsv4_candidate.json

./workspace/bin/llama-perplexity \
  -m ./workspace/models/dsv4-maxq-fractal/DeepSeek-V4-Flash-0731-MaxQ-Fractal-00001-of-00003.gguf \
  -f wiki-alt.txt -c 5632 -b 5632 --repack -fa on -t 16 -tb 16 --ppl-output-type 1 --chunks 12

python3 benchmarks/eval_perplexity_and_kld.py \
  --llama-perplexity ./workspace/bin/llama-perplexity \
  --model ./workspace/models/NVIDIA-Nemotron-3.5-30B-A3B-VQ3-64.gguf \
  --ctx-size 512 --chunks 24 \
  --url http://127.0.0.1:8080 \
  --ref-logits benchmarks/results/nemotron_fractal_ref_logits.json \
  --save-json benchmarks/results/kld_nemotron_vq3_vs_fractal.json

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
