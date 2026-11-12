# `DeepSeek-V4-Flash 284B` (`13.17B` Active) — Mathematical & Kernel Guide

> **Back to:** [Main Overview (`README.md`)](../README.md) · [LUT-VQ3-64 Quality Gate (`benchmarks/QUALITY_GATE_VQ3_64.md`)](../benchmarks/QUALITY_GATE_VQ3_64.md) · [Hardware & Chip Matrix (`docs/HARDWARE_OPTIMIZATIONS.md`)](HARDWARE_OPTIMIZATIONS.md) · [Nemotron Guide (`docs/NEMOTRON_OPTIMIZATIONS.md`)](NEMOTRON_OPTIMIZATIONS.md) · [Full Paper (`ARCHITECTURE.md`)](../ARCHITECTURE.md)

---

## 1. Architectural Profile (`deepseek4` — `284.33B` Total / `13.17B` Active)

`DeepSeek-V4-Flash-0731` is a **43-layer Mixture-of-Experts Transformer** with:
- **Multi-Head Latent Attention (`MLA`) + Sparse Indexer Compressors (`CSA` / `LID` / `HCA`)**: `64` query heads (`q_lora_rank = 1024`), `1` shared latent KV head (`key_length = 512`, `value_length = 512`, `rope_dim = 64`), and `8` output head-groups (`output_lora_rank = 1024`, `wo_a` `[4096, 8192]` $\to$ `wo_b` `[8192, 4096]`), plus `512`-point Fast Walsh-Hadamard Transforms (`GGML_OP_HADAMARD`, `396` calls/token).
- **4-Stream Sinkhorn-20 Hyper-Connections (`hc = 4`)**: Every attention and FFN sub-layer applies `HC_PRE` and `HC_POST` mixing across a `4 × 4096` residual stream.
- **Dual-Regime 256-Expert MoE (`1` Shared + `6` Routed Experts Active per Layer, `w_scale = 1.5`)**:
  - **Layers `0..2` (`dsv4_hash_layer_count = 3`)**: **Deterministic Token-ID Hash-Routed MoE** (`ffn_gate_tid2eid` maps each token ID directly to 6 lexical experts with near-uniform weights `~0.25`).
  - **Layers `3..42` (`40` layers)**: **Learned Top-K Router MoE** (`gating_func = SQRT_SOFTPLUS` with expert group top-k selection).

---

## 2. Operator-by-Operator Profiling (Prefill vs Decode on 16C/32T Zen 5)

Measured via `LLAMA_PROFILE_OPS` inside `ggml_graph_compute_thread` ([`patches/deepseek-v4-flash-fractal-256k.patch`](../patches/deepseek-v4-flash-fractal-256k.patch)) on an **AMD Ryzen 9 9955HX (`16C/32T Zen 5`, `0 GPUs`)**:

### A. Single-Token Decode (`N=1`, `4,551` Graph Nodes) — Evolution Across Optimizations

| Operator / Subsystem (`43` Layers) | Tensor Types | Calls / Tok | `v2.3.0` Initial (`169.3 ms`) | Pre-Micro-Opt (`121.9 ms`) | **`v2.5.0` Unpruned `Top-6` (`110.3 ms`)** | **`v2.5.0` `Top-4 L2` (`98.8 ms`)** | What Was Optimized |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1. MLA & Shared Expert `MUL_MAT`** (`wq_b`, `wo_a`, `wo_b`, `wkv`, `shexp`) | `q2_0` / `vq3_0` | `512` | `78.40 ms` | `62.20 ms` | **`61.81 ms`** | **`61.81 ms`** | AVX-512 `vprorvq` / `vpermi2b`, 3D L1D tiling (`wo_a`), 8-slot `Q8_0` cache, barrier elision (`nr0 < 4096`) |
| **2. Routed MoE `MUL_MAT_ID`** (`up_exps`, `gate_exps`, `down_exps` — Bulk) | `iq1_s` (`SwiGLU-H2`) | `87` | `25.10 ms` | `22.79 ms` | **`22.27 ms`** | **`15.17 ms`** *(`-33.4%`)* | 8-slot `Q8_K` cache, `MUL_MAT_ID` row-group reuse, `ones256` AVX2 hoist, $L_2$-quadrature tail pruning |
| **3. Routed MoE `MUL_MAT_ID`** (Anchor Layers `{0,1,26,28,41,42}`) | `iq2_xxs` | `18` | `4.50 ms` | `4.04 ms` | **`4.01 ms`** | **`2.75 ms`** *(`-31.9%`)* | `MUL_MAT_ID` row-group reuse + $L_2$-quadrature tail pruning |
| **4. Routed MoE `MUL_MAT_ID`** (Mid-Sensitivity Layers) | `iq1_m` | `24` | `2.10 ms` | `1.81 ms` | **`1.80 ms`** | **`1.16 ms`** *(`-35.9%`)* | `MUL_MAT_ID` row-group reuse + $L_2$-quadrature tail pruning |
| **5. Output Head (`lm_head`) + Router Gates** | `q4_K` / `q6_K` / `f32` | `298` | `11.20 ms` | `9.29 ms` | **`9.05 ms`** | **`9.05 ms`** | `_MM_HINT_NTA` non-temporal prefetch in `q4_K` & `q6_K` |
| **6. Fast Walsh-Hadamard (`HADAMARD`), `RMS_NORM`, `HC_PRE/POST`, `FLASH_ATTN`** | `f32` / `f16` | `1,810` | `19.60 ms` | `10.12 ms` | **`9.85 ms`** | **`8.90 ms`** | 9-stage in-register AVX2 FWHT (`ops.cpp`), AVX2 `HC_PRE/POST`, fused `RMS_NORM+MUL`, MLA 8-step lookahead |
| **7. Graph Rebuild & `CSA/LID` Compressor Overhead** | Graph / C++ | `1` | `28.40 ms` | `11.67 ms` | **`1.54 ms`** | **`0.00 ms`** | Zero-rebuild dynamic `CSA/LID` skip (`flags |= 0x100`, `ggml_dsv4_skip_csa_comp`) |
| **TOTAL SINGLE-TOKEN LATENCY (`ms/tok` & `tok/s`)** | **All (`3.20 GB` -> `2.85 GB`)** | **`4,551`** | **`169.30 ms` (`5.91 tok/s`)** | **`121.92 ms` (`8.20 tok/s`)** | **`110.33 ms` (`9.06 tok/s`)** | **`98.84 ms` (`10.12 tok/s`)** | **`+53.3%` Unpruned `Top-6` / `+71.2%` with `Top-4 L2` (`11.5–25.3 tok/s` w/ `Stockfish-PVS`)** |

### B. Batched Prompt Prefill (`pp512`, `n_tokens = 512`, `5,375` Graph Nodes)

| Operator Category (`pp512` Batch) | Total Time (`512` Tokens) | Time per Token (`ms/tok`) | Share of Prefill | Optimization Applied |
| :--- | :---: | :---: | :---: | :--- |
| **`MUL_MAT` (`q2_0` MLA + Shared Expert GEMM)** | `4,250.33 ms` | `8.30 ms/tok` | `34.5%` | AVX-512 `vprorvq` (`_mm512_rorv_epi64`) + L1D cache tiling |
| **`MUL_MAT_ID` (`iq1_s` + `iq2_xxs` + `iq1_m` Routed MoE GEMM)** | `5,849.85 ms` | `11.43 ms/tok` | `47.5%` | Batched expert bucket aggregation (`cne1 > 1` GEMM reuse across 512 tokens) |
| **`FLASH_ATTN_EXT` + `DSA_INDEXER` (MLA + Sparse Indexer)** | `1,401.10 ms` | `2.74 ms/tok` | `11.4%` | 8-step sparse lookahead prefetch + AVX-512 F16 dot products |
| **`HADAMARD` + `HC_PRE/POST` + `RMS_NORM` + Misc** | `811.78 ms` | `1.59 ms/tok` | `6.6%` | In-register AVX2 9-stage FWHT + vectorized Sinkhorn-20 Hyper-Connections |
| **TOTAL `pp512` PREFILL (`24.70 tok/s` $\to$ `41.58 tok/s`)** | **`12,313.06 ms`** *(was `20,730 ms`)* | **`24.05 ms/tok`** *(was `40.49 ms/tok`)* | **`100.0%`** | **`+68.3%` Faster Prompt Prefill (`-8.42 s` per 512 tokens)** |

---

## 3. Mathematical Co-Design: Low-Rank & Orthogonal Structure

### 3.1. SVD Spectrum of the Linear `wo_b * wo_a` Operator (`31%` of DRAM Traffic = `991.6 MB/token`)

In [`deepseek4.cpp`](../patches/deepseek-v4-flash-fractal-256k.patch), the attention output projection applies two consecutive `Q2_0` matrices across `8` head-groups (`4096` input MLA group dim $= 8 \text{ heads} \times 512 \text{ latent dim}$, `1024` inner LoRA rank per group, `8192` total) with **no activation function, no normalization, and no bias** between them:
$$\mathbf{y}_{\text{attn}} = \sum_{g=1}^{8} \mathbf{W}_{o,b}^{(g)} \left( \mathbf{W}_{o,a}^{(g)} \mathbf{x}_g \right), \qquad \mathbf{W}_{o,a}^{(g)} \in \mathbb{R}^{1024 \times 4096},\; \mathbf{W}_{o,b}^{(g)} \in \mathbb{R}^{4096 \times 1024}$$

De-quantizing the GGUF tensors (`blk.{0,3,4,10}.attn_output_{a,b}.weight`) and computing the Singular Value Decomposition $\mathbf{W}_{o,b}^{(g)} \mathbf{W}_{o,a}^{(g)} = \mathbf{U}_g \boldsymbol{\Sigma}_g \mathbf{V}_g^\top$ shows rapid spectral decay ($\sigma_0 = 15.398 \to \sigma_{768} = 0.901 \to \sigma_{1023} = 0.048$):

| Retained Rank $r$ (per `1024`-dim Group) | DRAM Saved on `wo_a + wo_b` | Latency Saved / Token | Frobenius Energy $\sum_{i=1}^r \sigma_i^2 / \sum_{i=1}^{1024} \sigma_i^2$ | **Output Cosine Similarity $\cos(\mathbf{y}, \hat{\mathbf{y}})$** | Relative $L_2$ Error |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **$r = 1024$ (Full Rank)** | `0 MB` (`991.6 MB/tok`) | `0.0 ms` | `100.00%` | **`1.000000`** | `0.00%` |
| **$r = 768$ (`-25.0%` Rank)** | **`-247.9 MB/tok`** | **`-6.9 ms/tok`** | **`99.64%`** (`96.8% – 100.0%`) | **`0.998204` (`99.82%`)** | **`5.99%`** |
| **$r = 640$ (`-37.5%` Rank)** | **`-371.9 MB/tok`** | **`-10.3 ms/tok`** | **`98.61%`** (`96.5% – 99.6%`) | **`0.992999` (`99.30%`)** | **`11.80%`** |
| **$r = 512$ (`-50.0%` Rank)** | **`-495.8 MB/tok`** | **`-13.8 ms/tok`** | **`96.02%`** (`87.9% – 98.7%`) | **`0.979718` (`97.97%`)** | **`20.02%`** |

### 3.2. Why MoE Tail Pruning Requires $L_2$-Quadrature Rescaling (Not Linear $L_1$ Rescaling)

Each MoE layer sums the dense shared expert $\mathbf{e}_{\text{shared}}(\mathbf{x})$ and 6 routed experts $\mathbf{e}_{k_1}(\mathbf{x}), \dots, \mathbf{e}_{k_6}(\mathbf{x}) \in \mathbb{R}^{4096}$:
$$\mathbf{y}_{\text{FFN}}(\mathbf{x}) = \mathbf{e}_{\text{shared}}(\mathbf{x}) + \sum_{i=1}^{6} w_i \,\mathbf{e}_{k_i}(\mathbf{x}), \qquad \sum_{i=1}^{6} w_i = 1.5$$

1. **Preserving Hash-Routed Lexical Layers (`il = 0..2`)**: Layers `0, 1, 2` select experts via 6 parallel token-ID hash tables (`tid2eid`) with flat weights (`[0.30, 0.29, 0.24, 0.23, 0.22, 0.22]`). Pruning any expert in `il = 0..2` drops 1/6th of the token's lexical lookup, so the engine always keeps `il = 0..2` at `6/6` experts.
2. **Power-Law Concentration in Learned-Router Layers (`il = 3..42`)**: In layers `3..42`, the top-4 experts capture `~81.2%` of the routed weight (`>90%` of total FFN energy including `shexp`), while the 6th expert carries `6%–9%` (`w_6 ≈ 0.08–0.14` out of `1.50`).
3. **$L_2$-Quadrature vs Linear $L_1$ Rescaling**: Because the 6 expert vectors in $\mathbb{R}^{4096}$ are nearly mutually orthogonal ($\langle \mathbf{e}_i, \mathbf{e}_j \rangle \approx 0$), their expected squared norm is $\mathbb{E}\|\sum_{i=1}^6 w_i \mathbf{e}_i\|_2^2 \approx \sum_{i=1}^6 w_i^2 \|\mathbf{e}_i\|_2^2$. Naive linear rescaling ($\alpha_{L_1} = \frac{\sum_{i=1}^6 w_i}{\sum_{i=1}^k w_i} \approx 1.23$ for $k=4$) inflates the routed MoE vector by `+23%` relative to the unscaled shared expert $\mathbf{e}_{\text{shared}}(\mathbf{x})$. Rescaling in **quadrature ($L_2$)** for $1 \le k \le K$:
   $$\alpha_{L_2} = \sqrt{\frac{\sum_{i=1}^{6} w_i^2}{\sum_{i=1}^{k} w_i^2}} \approx 1.015 \;\text{(for } k=5\text{)} \quad\text{or}\quad 1.042 \;\text{(for } k=4\text{)}$$
   preserves the variance ratio $\mathbb{E}\|\text{moe\_out}\|_2^2 / \mathbb{E}\|\mathbf{e}_{\text{shared}}\|_2^2$ entering the downstream `RMSNorm`, **cutting next-token KL-divergence by `-58.1%` (`0.0671 -> 0.0281 nats`)**. More generally, if routed experts exhibit small positive pairwise cosine correlation $\rho \in [0, 1)$, the variance-preserving factor is $\alpha_{\text{gen}}(\rho) = \sqrt{\frac{(1-\rho)\sum_{i=1}^6 w_i^2 + \rho(\sum_{i=1}^6 w_i)^2}{(1-\rho)\sum_{i=1}^k w_i^2 + \rho(\sum_{i=1}^k w_i)^2}}$, which reduces to $\alpha_{L_2}$ as $\rho \to 0$ and to $\alpha_{L_1}$ only in the collinear limit $\rho \to 1$.

| Runtime Configuration (`il = 0..2` always `6/6`) | Pure Decode (`tok/s`) | Latency (`ms/tok`) | **Prob Cosine Similarity** | **Top-10 $D_{\text{KL}}$ (nats)** | **Local Top-10 $\Delta$PPL** | **Top-1 / Top-3 Token Match** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. `Top-6` Unpruned (Default)** | **`9.06 tok/s`** | **`110.33 ms`** | **`1.00000`** | **`0.0000`** | **`0.00%` (Ref)** | **`100.0% / 100.0%`** |
| **2. Adaptive Tail (`min_w=0.11`, $L_2$)** | **`9.52 tok/s`** | **`105.02 ms`** | **`0.99634`** | **`0.0047`** | **`+0.47%`** | **`95.8% / 100.0%`** |
| **3. `Top-5` Sweet-Spot (`il >= 3`, $L_2$)** | **`9.68 tok/s`** | **`103.27 ms`** | **`0.99714`** | **`0.0042`** | **`+0.42%`** | **`95.8% / 100.0%`** |
| **4. Hybrid `6/4/6` (`il = 16..39` at 4, $L_2$)** | **`9.83 tok/s`** | **`101.74 ms`** | **`0.99660`** | **`0.0184`** | **`+1.86%`** | **`95.8% / 100.0%`** |
| **5. Hybrid `5/4/5` (`il = 12..40` at 4, $L_2$)** | **`9.95 tok/s`** | **`100.55 ms`** | **`0.99463`** | **`0.0215`** | **`+2.17%`** | **`95.8% / 100.0%`** |
| **6. `Top-4` Turbo (`il >= 3`, $L_2$)** | **`10.12 tok/s`** | **`98.84 ms`** | **`0.99587`** | **`0.0281`** | **`+2.85%`** | **`95.8% / 100.0%`** |

### 3.3. Zero-Overhead Live Switching (`Linux`, `macOS`, & `Windows`)

> **Important:** Never set `DSV4_MOE_TOPK` or `LLAMA_MOE_TOP_K` in the environment when loading `DeepSeek-V4-Flash` (doing so alters expected `ffn_gate_tid2eid` tensor dimensions at load time). Always control Top-K pruning through the 32-byte memory-mapped config file below.

```bash
# Linux / macOS: write 32-byte struct to /tmp/dsv4_moe_cfg.bin (takes 0.001 ms, no server restart)
python3 -c 'import struct; open("/tmp/dsv4_moe_cfg.bin","r+b").write(struct.pack("<8i", 5, 5, 0, 3, 43, 1, 0, 0))'   # Top-5 L2 (9.87 tok/s, +0.42% PPL)
python3 -c 'import struct; open("/tmp/dsv4_moe_cfg.bin","r+b").write(struct.pack("<8i", 5, 5, 0, 3, 43, 1, 80, 80))' # Top-5 L2 + Gate-First SwiGLU 80/128 (10.15 tok/s)
python3 -c 'import struct; open("/tmp/dsv4_moe_cfg.bin","r+b").write(struct.pack("<8i", 4, 4, 0, 3, 43, 1, 80, 80))' # Top-4 L2 + Gate-First SwiGLU 80/128 (10.51 tok/s)
python3 -c 'import struct; open("/tmp/dsv4_moe_cfg.bin","r+b").write(struct.pack("<8i", 6, 6, 0, 3, 43, 1, 0, 0))'   # Top-6 Unpruned (default, 0.00% routing KLD)
```

```powershell
# Windows PowerShell: launch via modchip_dsv4.ps1 -Fast (Top-4 L2) or update %TEMP%\dsv4_moe_cfg.bin in-place
powershell -ExecutionPolicy Bypass -File windows\modchip_dsv4.ps1 -Fast
```

---

## 4. L1-Local Stratified Gate-First SwiGLU Row Skipping with $L_2$ Gate-Energy Preservation (`cfg[6]`, `cfg[7]`)

In every SwiGLU FFN block (both the 43 Shared Experts `ffn_*_shexp` and the $43 \times K$ active Routed Experts `ffn_*_exps`), the hidden representation is:
$$\mathbf{h} = \text{SiLU}(\mathbf{W}_{\text{gate}} \mathbf{x}) \odot (\mathbf{W}_{\text{up}} \mathbf{x}), \qquad \mathbf{y} = \mathbf{W}_{\text{down}} \mathbf{h}, \qquad h_j = \frac{g_j}{1 + e^{-g_j}} \cdot u_j \quad (j \in \{0,\dots,2047\}).$$

1. **Topological Gate-First Execution in `ggml`**: Because `ggml_swiglu_split(ctx0, gate, up)` binds `gate` to `src[0]` and `up` to `src[1]`, `ggml`'s post-order DFS schedules `W_gate * x` (`ffn_gate` / `ffn_moe_gate`) **immediately before** `W_up * x` (`ffn_up` / `ffn_moe_up`). Because each of the 16 worker threads (`ith = 0..15`, pinned 1-to-1 to physical cores `0..15`) computes the contiguous 128-row slice `[128 * ith, 128 * (ith + 1))` in both `W_gate` and `W_up`, the 128 gate pre-activations $g_j$ are already hot in Core `ith`'s private L1D cache when `W_up` begins.
2. **Stratified Per-Thread Selection & $L_2$ Gate-Energy Rescaling (`ggml_swiglu_gate_mask_128`)**: Because $\max_{g \le 0} |\text{SiLU}(g)| = 0.2785$ and negative/near-zero gates attenuate $u_j$, each thread `ith` runs an $O(128)$ in-register Hoare quickselect on $s_j^2 = \text{SiLU}(g_j)^2$ over its 128-row stratum ($<40\text{ ns}$), skips fetching the bottom $128 - m$ rows of $\mathbf{W}_{\text{up}}$ from DDR5 DRAM, and rescales the $m$ surviving rows by the conditional $L_2$ gate-energy preservation factor:
   $$\alpha_{\text{SwiGLU}}^{(c)} = \sqrt{\frac{\sum_{j \in \mathcal{C}_c} \text{SiLU}(g_j)^2}{\sum_{j \in \mathcal{A}_c} \text{SiLU}(g_j)^2}}.$$
   Because $\mathbb{E}[h_j^2 \mid g_j] = \text{SiLU}(g_j)^2 \sigma_u^2$, rescaling by $\alpha_{\text{SwiGLU}}^{(c)}$ preserves 99.86%–100.00% of the output norm $\|\mathbf{W}_{\text{down}} \mathbf{h}\|_2$ entering the Hyper-Connections stream with 16-thread load balance and zero cross-thread synchronization.

| Combined Runtime Profile (`il = 0..2` always `100%`) | `cfg[0..7]` in `/tmp/dsv4_moe_cfg.bin` | Decode (`tok/s`) | Latency (`ms/tok`) | **Decode Prob CosSim** | **Agentic JSON Tool-Call (`execute_shell`)** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. `Top-6` + SwiGLU `80/128` (`62.5% W_up`)** | `[6, 6, 0, 3, 43, 1, 80, 80]` | **`9.71 tok/s`** | **`103.04 ms`** | **`0.99403`** | **`100% Exact Match`** |
| **2. `Top-5 L2` + Exps SwiGLU `96/128` (`75%`)** | `[5, 5, 0, 3, 43, 1, 96, 0]` | **`9.83 tok/s`** | **`101.76 ms`** | **`0.99624`** | **`100% Exact Match`** |
| **3. `Top-5 L2` + Both SwiGLU `80/128` (`62.5%`)** | `[5, 5, 0, 3, 43, 1, 80, 80]` | **`10.15 tok/s`** | **`98.49 ms`** | **`0.99385`** | **`100% Exact Match`** |
| **4. `Top-4 L2` + Both SwiGLU `96/128` (`75%`)** | `[4, 4, 0, 3, 43, 1, 96, 96]` | **`10.60 tok/s`** | **`94.37 ms`** | **`0.99126`** | **`100% Exact Match`** |
| **5. `Top-4 L2` + Both SwiGLU `64/128` (`50%`)** | `[4, 4, 0, 3, 43, 1, 64, 64]` | **`10.74 tok/s`** | **`93.15 ms`** | **`0.98059`** | **`100% Exact Match`** |

---

## 5. Shipped 64 GB Checkpoints: Single-Pass `Abliterated-Fractal` (`56.93 GiB` Default) & `LUT-VQ3-64` (`56.40 GiB` Aligned)

`v2.5.0` ships three 3-shard `DeepSeek-V4-Flash-0731` checkpoints tailored for 64 GB CPU hosts ([`patches/lut-vq3-64-avx512-vbmi.patch`](../patches/lut-vq3-64-avx512-vbmi.patch), [`scripts/build_dsv4_vq3.py`](../scripts/build_dsv4_vq3.py), [`benchmarks/QUALITY_GATE_VQ3_64.md`](../benchmarks/QUALITY_GATE_VQ3_64.md)):

- **Default DeepSeek Checkpoint — [`DeepSeek-V4-Flash-0731-Abliterated-Fractal`](https://huggingface.co/Synchopate/DeepSeek-V4-Flash-0731-Abliterated-Fractal-64GB-CPU-GGUF) (`56.93 GiB` / `61.13 GB`, `59.13 GiB` `VmRSS`):**
  - Quantizes all `130` dense bottleneck tensors (`token_embd`, `output`, `43× attn_q_a`, `43× ffn_{gate,up,down}_shexp`) in a single pass directly from `prem-research/DeepSeek-V4-Flash-0731-abliterated` (`BF16` + native `FP8_E4M3` block-128) into `Q6_K`, eliminating `FP8 -> Q5_K/Q4_K` generational loss while preserving 100% of MLA attention in `Fractal-64` (`GGML_TYPE_Q2_0 = 42`, `3.75 bpw`).
  - Achieves **`1.7597 ± 0.06534` `PPL` (`-3.76%` vs `1.8284`)**, **`70.0 / 80.0`** on the Functional Gate (`30/30` Tool-Calling, `20/20` Needle/Math), and **`50.0 / 60.0`** on the `Hard-60` suite at `8.75–9.50 tok/s` `N=1` decode.
  - *Responsible-Use Note:* Upstream refusal filters are disabled on this abliterated checkpoint (intended for authorized security research, CTF, auditing, and local offline development; do not expose unmoderated to public networks).
- **Standard Safety-Aligned Hybrid — [`DeepSeek-V4-Flash-0731-VQ3`](https://huggingface.co/Synchopate/DeepSeek-V4-Flash-0731-VQ3-64GB-CPU-GGUF) (`56.40 GiB` / `60.56 GB`, 3 shards on Hugging Face):**
  - Preserves **100% of MLA attention and compressor projections** (`attn_q_b`, `attn_output_a`, `attn_output_b`, `attn_kv`, `attn_compressor_*`, `indexer_compressor_*`) untouched in **`Fractal-64` (`GGML_TYPE_Q2_0 = 42`, `3.75 bpw`)** so latent KV phase geometry experiences zero double-quantization.
  - Preserves boundary shared experts (`blk.0..4`, `blk.39..42`) and all 43 `ffn_down_shexp` projections in `Q5_K` / `Q6_K`.
  - Encodes the **68 interior pre-`SiLU` shared-expert projections (`blk.5..38` `ffn_gate_shexp` & `ffn_up_shexp`)** into **`LUT-VQ3-64` (`GGML_TYPE_VQ3_0 = 43`, `24 B / 64 W = 3.00 bpw`)**, decoded in registers via AVX-512 VBMI `VPERMI2B`.
- **Aggressive 255-Tensor Full-VQ3 Ablation (`--aggressive`, `56.07 GiB` disk / `58.31 GiB` `VmRSS`):**
  - Also encodes MLA expansion projections (`attn_q_b`, `attn_output_a`, `compressor_*`) and all 43 `ffn_gate_shexp` + `ffn_up_shexp` tensors into `VQ3_0`, boosting raw `N=1` `tg64` decode by **`+16.9%` (`9.05 tok/s` vs `7.74 tok/s`)**, `pp128` prefill by **`+8.8%` (`41.11 tok/s`)**, and agentic tool bursts to **`19.4–25.3 tok/s`** while passing `30/30` Tool-Calling and `20/20` Long-Context Needle & Math.

```bash
# Build shipped conservative 56.40 GiB Hybrid (default) or 56.07 GiB aggressive ablation:
python3 scripts/build_dsv4_vq3.py --src-dir /path/to/Fractal-shards --dst-dir /path/to/VQ3-shards
python3 scripts/build_dsv4_vq3.py --src-dir /path/to/Fractal-shards --dst-dir /path/to/VQ3-aggressive --aggressive
```

---

## 6. 4-Level Register (`ZMM` AVX-512BW/VBMI) + L1D + L2 + L3 Whole-Model Prefill & Agentic KV-Cache Tiling

To accelerate prompt prefill ($N > 1$) and multi-turn agentic workflows without altering numerical output relative to the single-column kernel (`bit-identical` GEMM tiling), the CPU execution pipeline tiles across all four levels of the Zen 4 / Zen 5 cache hierarchy:

1. **Level 0 — 512-Bit `ZMM` Register-Stationary Multi-Column Kernels (`_1x8`, `_1x4`, `_1x3`, `_1x2`, `1x1`) Across All Quantized Formats**:
   - Implemented in `ggml/src/ggml-cpu/quants.c` (`Q2_0`, `VQ3_0`) and `ggml/src/ggml-cpu/arch/x86/quants.c` (`IQ1_S`, `IQ2_XXS`, `IQ1_M`, `Q5_K`, `Q6_K`, `VQ3_0`).
   - Because `quantize_row_q8_K` scales by `iscale = -127.0f / max`, every quantized input byte in `block_q8_K.qs` lies in `[-127, +127]` (never `-128`). This allows replacing 256-bit AVX2 `vpsignb` loops (`32` weights/instruction) with 512-bit **AVX-512BW** masked negation (`_mm512_abs_epi8`, `_mm512_movepi8_mask`, `_mm512_mask_sub_epi8`, `_mm512_maddubs_epi16`, `_mm512_madd_epi16`, `64` weights/instruction), plus a **7-instruction BMI2 `_pdep_u64` + SWAR XOR-parity** 64-bit sign mask generator for `IQ2_XXS`.
   - Each weight block (`Q2_0`, `VQ3_0`, `IQ1_S`, `IQ2_XXS`, `IQ1_M`, `Q5_K`, `Q6_K`) is unpacked **once** in `ZMM` registers and multiplied against $K \in \{8, 4, 3, 2\}$ token columns simultaneously, cutting weight-unpacking instruction count by up to **8x**.
2. **Level 1 & Level 2 — L1D + L2 Cache-Blocked GEMM & Zero-Remainder `{8, 4, 3, 2}` Decomposition**:
   - In `ggml_compute_forward_mul_mat_one_chunk` and `ggml_compute_forward_mul_mat_id_one_chunk` (`ggml-cpu.c`), an outer L2 loop tiles `blck_0_l2 = 128` weight rows (`64–276 KB` per core $\ll 1\text{ MB}$ L2) while an inner L1D loop tiles $K \in \{8, 4, 3, 2\}$ input columns.
   - Because any integer $M \ge 2$ is a nonnegative linear combination of $\{8, 4, 3, 2\}$ ($5 = 3+2$, $9 = 4+3+2$), ragged MoE expert buckets (`MUL_MAT_ID` across `256` experts) execute with **zero fallback to `1x1` whenever an expert has $\ge 2$ tokens**.
   - A dynamic atomic work queue (`ggml_compute_forward_mul_mat_id`) with cache-line-aligned (`64 B` = 16 floats) row chunks avoids cross-thread false sharing and load imbalance across all 256 routed experts.
3. **Whole-Model Non-Matmul Vectorization & Cross-CCD Infinity Fabric Isolation**:
   - **AVX-512 `quantize_row_q8_K`**: Replaces the scalar `quantize_row_q8_K_ref` fallback in `arch/x86/quants.c` with a bit-exact AVX-512 implementation (`16` `ZMM` registers + `_mm512_cvtepi32_epi8` + tree-reduced `bsums`).
   - **Register-Stationary AVX-512 `MQA8` FlashAttention (`FLASH_ATTN_EXT`)**: Converts each 16-element `FP16` K/V slice once in a `ZMM` register (`_mm512_cvtph_ps`) and accumulates across 8 query heads simultaneously (`2.0x` speedup: `415.1 ms -> 208.5 ms`).
   - **Token-to-Core L1D/L2 Affinity across `DSV4_HC_POST -> DSV4_HC_PRE -> RMS_NORM` & Row-Parallel `from_float`**: Maps token `it` to thread `ith = it % nth` across Hyper-Connections and `RMS_NORM`, keeping each token's `16 KB` activation vector hot in the local core's L1D cache and avoiding 16-way cross-CCD cache-line snoops.

### Verified Prefill & Multi-Turn Agentic KV-Cache Benchmarks (`DeepSeek-V4-Flash 284B`, 16-Core Zen 4/5)

| Benchmark Scenario | Prompt / KV Tokens | Unoptimized Baseline | `v2.4.0+` 4-Level Cache-Tiled | Speedup & Decode Rate |
| :--- | :---: | :---: | :---: | :---: |
| **Cold Prefill (`pp63`)** | `63` tok | `25.05 tok/s` | **`36.65 tok/s`** (`1,718.9 ms`) | **`+46.3%`** (`8.95–9.02 tok/s` decode) |
| **Cold Prefill (`pp298`)** | `298` tok | `25.19 tok/s` | **`43.16 tok/s`** (`6,904.6 ms`) | **`+71.3%`** (`8.74–9.01 tok/s` decode) |
| **Cold Prefill (`pp392`)** | `392` tok | `25.10 tok/s` | **`43.03 tok/s`** (`9,109.6 ms`) | **`+71.4%`** (**`9.55–9.90 tok/s`** decode) |
| **Cold Prefill (`pp580`)** | `580` tok | `24.73 tok/s` | **`43.98 tok/s`** (`13,189.3 ms`) | **`+77.8%`** (**`9.60 tok/s`** decode) |
| **Agentic Turn 1 (`cache_prompt=True`)** | `total_kv = 403` (`new=339`) | `25.12 tok/s` | **`44.36 tok/s`** | **`+76.6%`** (**`9.95 tok/s`** decode) |
| **Agentic Turn 2 (`cache_prompt=True`)** | `total_kv = 806` (`new=407`) | `24.90 tok/s` | **`43.89 tok/s`** | **`+76.3%`** (**`9.86 tok/s`** decode) |
| **Agentic Turn 3 (`cache_prompt=True`)** | `total_kv = 1209` (`new=407`) | `24.65 tok/s` | **`43.28 tok/s`** | **`+75.6%`** (**`9.74 tok/s`** decode) |
| **Agentic Turn 4 (`cache_prompt=True`)** | `total_kv = 1612` (`new=407`) | `24.40 tok/s` | **`42.67 tok/s`** | **`+74.9%`** (**`9.69 tok/s`** decode) |
| **Agentic Turn 5 (`cache_prompt=True`)** | `total_kv = 2015` (`new=407`) | `23.95 tok/s` | **`40.48 tok/s`** | **`+69.0%`** (**`9.64 tok/s`** decode) |
| **Agentic Turn 6 (`cache_prompt=True`)** | `total_kv = 2418` (`new=407`) | `23.40 tok/s` | **`38.44 tok/s`** | **`+64.3%`** (**`9.16 tok/s`** decode) |
