# Technical FAQ, Architectural Trade-Offs & Server CPU Scaling (`Xeon` / `EPYC`)

*Navigation: [Back to Overview & Quickstart (`README.md`)](README.md) · [Technical Paper (`ARCHITECTURE.md`)](ARCHITECTURE.md) · [LUT-VQ3-64 Quality Gate (`benchmarks/QUALITY_GATE_VQ3_64.md`)](benchmarks/QUALITY_GATE_VQ3_64.md)*

---

## Q1: How does `LLAMA_ATTN_WINDOW=4096` (SWA) affect exact Needle-In-A-Haystack (NIAH) retrieval at 256K versus `LLAMA_ATTN_WINDOW=0` (Full Global Attention)?

In `NVIDIA-Nemotron-3.5-30B-A3B` (`52` blocks total: `23` Mamba-2 SSM + `23` Sparse MoE + `6` GQA Attention layers), sequence state is split across two mechanisms (*Jelassi et al., 2024; NVIDIA Nemotron-H Technical Report*):
- **23 Mamba-2 SSM Layers (+ 23 Stateless Sparse MoE FFN Blocks = 46 Non-Attention Blocks)**: Compress sequence history into a fixed-size recurrent state $h_t$ (`~18.4 MB` total across the 23 Mamba-2 layers) in $O(1)$ memory, preserving global semantic coherence and state tracking across 256K tokens. Fixed-size recurrent states cannot perform verbatim copy-paste of arbitrary high-entropy strings (such as a random hexadecimal address 100,000 tokens earlier) without global attention or external indexing.
- **6 GQA Attention Layers**: Perform exact token-level retrieval (induction heads and verbatim copy-paste).

Because `LLAMA_ATTN_WINDOW` in [`patches/zen5-nemotron-256k-optimizations.patch`](patches/zen5-nemotron-256k-optimizations.patch) is a **runtime environment selector** (`ops.cpp`), you can choose the operating regime required by your workload without recompiling:
1. **`LLAMA_ATTN_WINDOW=4096` (L3-Resident SWA Mode)**: Bounds the 6 GQA layers to the 16 initial attention sinks + the latest 4,096 tokens (**6.31–6.94 MB** fixed working set, fitting inside the 64 MB L3 cache). This keeps prefill throughput flat (~157–170 tok/s) at long depths, while exact long-range retrieval outside the 4K window can be delegated to external SQLite FTS / RAG indexing (`<15 ms`).
2. **`LLAMA_ATTN_WINDOW=0` (Full 256K Global GQA Mode, Default)**: Disables sliding-window truncation so all 6 GQA layers attend across the **entire 262,144-token KV cache**, providing native 100% verbatim NIAH retrieval across all 256K tokens identical to the unmodified model. Because Nemotron 3.5 has only 12 KV heads across the entire 30B model (`6` GQA layers $\times$ `2` KV heads), the **256K KV cache occupies 384 MB raw / 432 MB in symmetric `Q4_0` (`~432 MB` in dynamic EAKV `Q8_0/Q4_0`, or `624 MB` in static `-ctk q8_0 -ctv q4_0`)** (versus 7.08 GB on a 48-layer GQA MoE), while `LLAMA_KV_L3_WINDOW=2048` issues `_MM_HINT_NTA` non-temporal prefetches for older KV tokens so full-context sweeps do not evict hot state from L3.
3. **`LLAMA_SWA_STRIDE=4` (Safety-Locked 3:1 Interleaved SWA for Pure GQA Models)**: When [`scripts/llama_modchip.py`](scripts/llama_modchip.py) detects a Pure GQA Transformer (`0` SSM layers, e.g., `Qwen3-30B-A3B` with 48 GQA layers), it blocks `stride=1` SWA and sets `LLAMA_SWA_STRIDE=4`: 36 layers use L3 SWA (`4096 + 16 sinks`) while every 4th layer (`12` layers) retains full 256K global attention.

---

## Q2: Why do 6 GQA layers vs. AVX-512 / HugePages solve two separate bottlenecks?

- **Why 1.536 KB/token (6 GQA layers) addresses the long-context KV wall**: Having only 12 KV heads across 52 blocks (`1.536 KB/tok` raw, `1.728 KB/tok` with `block_q4_0` scales = **432 MB at 256K**) prevents long-context KV bandwidth collapse on CPU (compared to ~7.1 GB on a standard 48-layer GQA model).
- **Why unpatched `llama.cpp` still runs at 17.0 tok/s at short context**: Even when the KV cache is ~0 MB, every generated token streams **~1.4–1.8 GB of active MoE weights** scattered across a `13.3–18.7 GB` virtual address space, causing millions of 4KB Data-TLB misses/sec and cross-CCD Infinity Fabric snoops. **2MB HugePages** (`MADV_HUGEPAGE`, covering 6.14 GB directly in L2 STLB), **AVX-512 VNNI/VBMI kernels**, and **CCD-0/CCD-1 pinning** raise the base decode floor independent of KV cache size.

---

## Q3: What are the measured Latency (TTFT) and Throughput numbers across workloads?

Live measurements on a single **AMD Ryzen 9 9955HX** (16 Zen 5 physical cores, 64 MB L3, dual-channel DDR5, zero GPU) running **Claude Code** with **40 tools loaded (`22.3K–35K` active context)**:

| Workload / Metric | Verified Speed (`Ryzen 9 9955HX`) | Notes |
| :--- | :---: | :--- |
| **Slot 0 Cold Restore from `/dev/shm` (`2,776 – 22,237` tok)** | **`10.4 – 39.3 ms`** | `POST /slots/0?action=restore` from `/dev/shm` snapshot |
| **Warm Interactive TTFT (LCP + HARP-Cache hit)** | **`0.16 – 0.50 s`** | Evaluates only the ~35–50 new user delta tokens |
| **End-to-End Autonomous `<tool_call>` Turn** | **`1.88 – 2.45 s total`** | `426 ms` prompt eval + `34.52 tok/s` XML/JSON tool generation (`mean len = 4.18`) |
| **Cold Prompt Prefill Throughput** | **`165.2 – 241.0 tok/s`** | Up to **306.8 tok/s** peak at `-b 2048` |
| **Decode — Base Physical Floor (No Speculation)** | **`17.2 tok/s`** *(at 33K ctx)* / **`25.3–32.7 tok/s`** *(short ctx, opt-in `Top-K=4`)* | High-entropy free prose where certainty gate halts drafting (`v2.5.2+` defaults to unpruned `Top-6`) |
| **Decode — Technical Prose + Security Audit (`27K` ctx)** | **`18.76 – 20.9 tok/s`** *(Peak **`39.28 tok/s`**)* | **`85.00%`** `Stockfish-PVS v2` draft acceptance (`mean len = 2.48–2.68`) |
| **Decode — Agent `<tool_call>` / JSON / XML (`22.3K` ctx)** | **`28.0 – 34.52 tok/s`** | **`83.33%`** draft acceptance (`mean len = 4.18`) |
| **Decode — Mixed Code & Architecture Refactor (`26.4K` ctx)** | **`28.61 tok/s` avg / `48.47 tok/s` peak** | **`86.10%`** draft acceptance (`mean len = 4.66`) |
| **Decode — In-Context Refactoring & Macro Burst (`M=7`)** | **`48.39 tok/s` avg / `60.40–65.0 tok/s` peak** | **`96.39%`–`98.0%`** acceptance (**`mean len = 6.90` tok/step** via Singular Horizon `M=7`) |

---

## Q4: Doesn't `_MM_HINT_NTA` on MoE weights in `quants.c` prevent shared/hot experts from staying in L3 across tokens or during speculative batch verification ($N=5$)?

No, due to the capacity ratio between the active MoE weight stream and the Zen 5 L3 cache:
1. **Why MoE weights cannot persist across tokens in L3 ($1,400\text{–}1,800\text{ MB} \gg 64\text{ MB}$)**:
   The Ryzen 9 9955HX has **64 MB of total L3 cache** ($2 \times 32\text{ MB}$ CCDs). Every token step streams **~1.4–1.8 GB** of active MoE weights sequentially from Block 0 to Block 51—more than **22x the size of the entire L3 cache**. Even if an expert in Layer 1 is selected on 100% of tokens, by the time execution passes through the remaining layers, Layer 1's weights have long been evicted from L3. Without `_MM_HINT_NTA`, streaming 1.8 GB of MoE weights through a 64 MB victim L3 evicts the **KV cache and 18.4 MB Mamba-2 state** (which do fit in L3) and is then immediately evicted itself by subsequent layers.
2. **Why intra-layer reuse during speculative batch verification ($N=5$) is preserved**:
   On AMD Zen 5, `_mm_prefetch(..., _MM_HINT_NTA)` (`PREFETCHNTA`) loads the cache line into the **L1d / private 1 MB L2 cache** and tags it for minimum retention when cast out into the **shared Victim L3**. During speculative batch verification ($N=5$) or prefill, all token activations in the batch are multiplied against the loaded weight block **within the same inner loop inside L1d/L2 and AVX-512 `zmm` registers** before eviction to L3.

---

## Q5: How is `RouteCrack` integrated into `llama-server`, and why is zero-rollback certainty gating required for Mamba-2?

[`speculative/cpp/ngram-map.cpp`](speculative/cpp/ngram-map.cpp) implements `RouteCrackNemotronEngine` directly inside `common_ngram_simple_draft()` in `libllama-common`:
1. **Startup Pre-Heating**: A static constructor (`RCPreheater`) loads the 8 Laplace-smoothed FNV-64 Hash Cascade tables (`hc_top4_k{10,8,6,5,4,3,2,1}.bin`), 300 Hot Token Rules (`hot_rules.bin`), and 11,382 Macro Codebook entries (`macro_codebook.bin`), and warms the Online Trie (`372,907` nodes), PPM (`k=2..6`), and Skipgram in ~100 ms at library load.
2. **Why Mamba-2 Requires Zero-Rollback Certainty Gating (`n_rollback == 0`)**: Unlike Transformer GQA KV caches (which drop rejected draft tokens by decrementing the KV index), Mamba-2's recurrent state tensor $h_t$ cannot be rewound token-by-token. In `llama-server` (`server-context.cpp`), before each speculative batch, the 23-layer Mamba-2 recurrent state (`~18.4 MB` per slot across `ssm_states` + `conv1d` caches) is snapshotted into `slot.spec_ckpt` (`~0.28 ms` `memcpy`). Whenever even 1 drafted token is rejected (`n_rollback > 0`), Mamba-2 restores `slot.spec_ckpt` and sets `slot.spec_is_replay = true` to replay the accepted prefix on the next step. When **100% of drafted tokens are accepted (`n_rollback == 0`)**, Mamba-2 commits the batch in a single pass with zero checkpoint restore and zero replay.
3. **Two-Zone In-Context (`16K Hot + 24K Prefix`) + RouteCrack Cross-Validation**: For every generated token, `common_ngram_simple_draft()` updates the online learners incrementally (`< 0.0003 ms`), scans both the 16,384-token Hot Zone and 24,576-token Prefix Zone, and cross-validates single-hit in-context candidates against the RouteCrack ensemble with per-step certainty thresholds (`84%` at $d=0$, `92%` at $d=1$, `96%` at $d=2$).

---

## Q6: How does Mamba-2 prefix checkpointing interact with multi-turn agentic CLI prompts and service restarts?

In `llama-server` (`server-context.cpp`), recurrent slot checkpoints are created at `last_user_pos` (the start of `<|im_start|>user`). To keep warm TTFT at **0.16–0.50 s** across turns and **10.4–32.6 ms** across service restarts:
1. Static project memory (`CLAUDE.md` / `MEMORY.md`) and tool definitions reside inside the `system` block **before** `<|im_start|>user`.
2. Per-turn volatile metadata injected by CLI wrappers after the user prompt (such as decrementing `<total_tokens>` counters or dynamic reminders) are stripped by [`gateway/sse_gateway.py`](gateway/sse_gateway.py) so they do not invalidate the prefix.
3. At service startup, [`scripts/prewarm_agent_prefix.py`](scripts/prewarm_agent_prefix.py) restores the binary snapshot (`claude_40tools_slot0.bin`) into `/dev/shm` via `POST /slots/0?action=restore` in **10.4–18.9 ms**, bypassing cold prompt evaluation.

---

## Q7: How does this architecture scale on Intel Xeon Scalable (Sapphire / Emerald / Granite Rapids) or AMD EPYC for concurrent tasks?

While our baseline numbers are measured on a dual-channel DDR5 Ryzen 9 9955HX (`~89 GB/s` theoretical peak, `64 MB` L3), server-grade CPUs widen both memory bandwidth and LLC capacity:

1. **8-to-12 Channel DDR5 Bandwidth (`300–600+ GB/s` vs `89 GB/s`)**:
   Every generated token streams ~1.4–1.8 GB of active MoE weights. Moving from dual-channel DDR5 to an 8-channel Threadripper/Xeon (`111.62 tok/s` verified on Threadripper PRO 7985WX) or 12-channel Xeon 6 / EPYC scales decode throughput roughly **3x–4x**.
2. **Large L3 / LLC (`256–504 MB`) + Intel AMX / Zen 5 AVX-512**:
   Because the 256K KV + Mamba-2 state occupies **432 MB** in full global attention (or **6.3 MB** with SWA), a server LLC can hold nearly the entire long-context KV cache plus shared MoE experts in L3 simultaneously, while Intel AMX or 512-bit AVX-512 accelerates prompt prefill and speculative batch verification.
3. **Scaling Concurrent Agent Tasks (Sub-NUMA Pinning)**:
   For multi-agent workloads, splitting a single 3B-active MoE across sockets introduces cross-socket UPI/Infinity Fabric latency. Instead, `mmap` the read-only weights once into page cache and pin **1 independent worker per Sub-NUMA Cluster (SNC)**. Because each 256K context slot takes `<450 MB` of RAM and restores from `/dev/shm` in `<20 ms`, a single multi-socket server can run **8 to 16 parallel 256K-context coding agents**.

---

## Q8: Can this run as a GPU + CPU Hybrid (offloading the hot backbone to a consumer GPU while keeping cold MoE experts on CPU)?

Yes. **Nemotron 3.5 30B-A3B** works well with **Tensor-Role Offloading** (`--n-cpu-moe` / `-ot`) on 8–16 GB GPUs:

1. **Always-Active Backbone + Full 256K State on GPU (`~3.8 GB` VRAM)**:
   Out of the `17.60 GiB` (`18.90 GB`) `Q4_0` model, **~15.5 GB** consists of the 128 sparse routed MoE experts across the 23 MoE layers (`blk.*.ffn_*_exps.weight`), of which only 4–6 out of 128 are touched per token. All tensors executed on **100% of tokens** across the 52-block stack—all 23 Mamba-2 SSM layers (`ssm_in`, `ssm_conv1d`, `ssm_out`), all 6 GQA Attention Q/K/V/O projections, Shared FFN Experts (`ffn_*_shexp`), MoE router gates (`ffn_gate_inp`), `output.weight`, and the **432 MB 256K KV + Mamba-2 state**—fit inside **< 4.0 GB of VRAM**.
2. **Sparse Routed MoE Experts on CPU (`~15.5 GB` System DDR5)**:
   ```bash
   llama-server -m NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0.gguf \
     -ngl 99 -ot "\.ffn_.*_exps\.=CPU" ...
   ```
   keeps sparse routed experts in system DDR5 RAM while only small activation vectors (`2,688` floats $\approx 10.7\text{ KB}$) cross PCIe per layer.
3. **Verified AMD Vulkan/RADV Community Benchmark (`Ryzen 9 7945HX3D` + `Radeon RX 9070 XT 16GB`)**:
   - **`Fractal-MXFP4` (`15.95 GiB` file / `~15.2 GB` active)**: **`75.1 tok/s` (new code) / `80.8 tok/s` (refactor)** at `256K` context (`--n-cpu-moe 30`).
   - **`Q4_0` (`17.60 GiB`)**: **`69.7 tok/s` (new code) / `76.2 tok/s` (refactor)**.
   - **`Fractal-64` (`Q2_0 = 42`, `14.36 GiB`, Pure CPU)**: **`44.2 tok/s` (new code) / `45.1 tok/s` (refactor)** on pure CPU.
4. **Backend Compatibility for Custom Quantization Formats (`MXFP4` vs `Fractal-64` vs `LUT-VQ3-64`)**:
   > [!IMPORTANT]
   > - **For Vulkan or CUDA GPU+CPU Hybrids**: Use **`Fractal-MXFP4` (`15.95 GiB`)** or **`Q4_0` (`17.60 GiB`)** — both have native Vulkan and CUDA compute shaders plus fast CPU kernels.
   > - **For Apple Silicon Metal**: Use **`Fractal-64` (`GGML_TYPE_Q2_0 = 42`, `14.36 GiB` / `56.57 GiB`)** on `16–24 GB` Macs (saves `-3.24 GiB` unified RAM so long context fits without swap) or **`Q4_0` (`17.60 GiB`)** on `36 GB+` Macs (`patches/apple-silicon/` and `patches/nemotron-apple-silicon/` in [PR #24](https://github.com/synchopate/llama-modchip/pull/24) reach `57.65 tok/s` M1 Max, `78.60 tok/s` M3 Max, `72.5 tok/s` M4 Max in `Fractal-64`, while `Q4_0` runs ~8% faster at `77.95 tok/s` when unified memory is unconstrained, [#16](https://github.com/synchopate/llama-modchip/issues/16)).
   > - **For x86-64 Pure CPU**: Use **`LUT-VQ3-64` (`GGML_TYPE_VQ3_0 = 43`, `13.30 GiB` / `56.40 GiB`)** on AVX-512 VBMI CPUs (Zen 4/5, Ice Lake+) for lowest RAM footprint, or **`Fractal-64` (`Q2_0 = 42`)** on AVX2-only CPUs. Neither `Q2_0` nor `VQ3_0` has a Vulkan/CUDA GPU shader yet.

---

## Q9: How does `scripts/llama_modchip.py` generalize across GGUF models, and why does `LLAMA_MOE_TOP_K=4` work while `Top-K=2` fails?

1. **Runtime Profiler & Controller (`scripts/llama_modchip.py`)**:
   [`scripts/llama_modchip.py`](scripts/llama_modchip.py) parses any `.gguf` binary header in `< 5 ms`, detects whether the model is a **Hybrid SSM+GQA** (`ssm_layers > 0`) or a **Pure GQA Transformer** (`ssm_layers == 0`) and whether it is a **Sparse MoE** (`expert_count > 1`) or **Dense**, sizes `LLAMA_KV_L3_WINDOW` dynamically so hot KV occupies $\le 25\%$ of L3 cache, applies architecture safety locks (`LLAMA_SWA_STRIDE=4` on Pure GQA; locking MoE Top-K pruning on Dense models; skipping `LLAMA_MOE_TOP_K` env export on `deepseek4` hash-routed layers), and writes `0600` live runtime controls to per-UID `/dev/shm/llama_modchip_<uid>.bin` (`$XDG_RUNTIME_DIR/llama_modchip.bin`).
2. **Why `LLAMA_MOE_TOP_K=4` (`6->4` / `8->4`) Succeeds While `Top-K=2` Fails**:
   - In `Nemotron-3.5-30B-A3B`, the model activates `6` routed experts (`ffn=1856`) + `1` double-width shared expert (`ffn=3712` = 2 expert-units) with `expert_weights_norm = True` and `expert_weights_scale = 2.5` (`v2.5.2+` runs unpruned `Top-6` by default; `Top-4` is opt-in via `LLAMA_MOE_TOP_K=4` at server boot). Pruning routed experts from `6 -> 4` cuts **33.3% of routed MoE DRAM bandwidth** while renormalizing the surviving 4 gates ($\sum_{i=1}^4 g_i = 2.5$).
   - On `Qwen3-30B-A3B` (`8 -> 4` routed experts, `norm_w = true` in `src/models/qwen3moe.cpp`), `LLAMA_MOE_TOP_K=4` cuts **50% of routed MoE DRAM traffic**, boosting non-speculative `tg64` decode from **`25.12` to `34.80 tok/s` (`+38.5%`)**.
   - Dropping to `LLAMA_MOE_TOP_K=2` collapses the directional covariance spectrum of the FFN output across 52 blocks (see [`ARCHITECTURE.md` §5.1](ARCHITECTURE.md)).

---

## Q10: With only 6 attention layers, isn't decode bound by streaming MoE weights from DDR5? How does it behave at full 256K?

On Dual-Channel DDR5-5600 (`~71.7 GB/s` sustained multi-core read bandwidth on Ryzen 9 9955HX):

### 1. Short Context ($0\text{ MB}$ KV Cache)
At $N=1$ decode on `Q4_0`, each token reads **~1.80 GB** of active weights (`~1.4–1.5 GB` in `Fractal-64` / `LUT-VQ3-64`). Stock `llama.cpp` stalls at **`17.0 tok/s`** (`58.8 ms/tok`) due to `4KB` page-table walks (`14.6M` L2 STLB misses/sec) and cross-CCD snoops. Enabling **2MB Transparent HugePages (`MADV_HUGEPAGE`)** (`6.14 GB` L2 STLB coverage), **AVX-512 kernels**, and 1:1 physical core pinning lifts pure $N=1$ decode (with opt-in `LLAMA_MOE_TOP_K=4` router truncation; `24.18–26.08 tok/s` unpruned `Top-6` Served Gate decode) to **`25.3–32.7 tok/s`**.

### 2. Full 256K (`262,144` Tokens)
At **262,144 tokens**, the 6-layer GQA KV cache occupies **`384 MB` raw / `432 MB` in `Q4_0` or dynamic EAKV** (`624 MB` in static `-ctk q8_0 -ctv q4_0`), which is $16.4\times$ smaller than a 48-layer GQA model (`7.08 GB`) but still larger than the `64 MB` Zen 5 L3 cache:

| 256K Operating Mode (`262,144` Tokens, `Q4_0` Weights) | Active KV Streamed per Step | Total DDR5 Read per Step | Pure $N=1$ Decode (`No Speculation`) | Effective Decode (`with RouteCrack + PVS`) |
| :--- | :---: | :---: | :---: | :---: |
| **1. Hybrid SWA Mode (`LLAMA_ATTN_WINDOW=4096` + 16 Sinks)** | **`6.31 MB`** *(L3-Resident)* + `18.4 MB` Mamba-2 state | **`~1.80 GB`** *(Flat from 4K to 256K)* | **`25.0 – 26.5 tok/s`** | **`38.0 – 50.5+ tok/s`** |
| **2. Unwindowed 256K Global GQA (`LLAMA_ATTN_WINDOW=0` + `_MM_HINT_NTA`)** | **`432 MB`** *(Streamed via `PREFETCHNTA`, L3 bypassed)* | **`~2.23 GB`** *($1.80\text{ GB MoE} + 0.43\text{ GB KV}$)* | **`19.5 – 20.5 tok/s`** | **`28.0 – 48.4+ tok/s`** *(1 sweep per 3–7 verified tokens)* |
| *Reference: Unwindowed 256K WITHOUT `_MM_HINT_NTA`* | `432 MB` *(Temporal loads flush 64 MB Victim L3)* | `~2.23 GB` + L3 Eviction Thrashing | *`14.8 – 15.2 tok/s`* | *`21.0 – 26.5 tok/s`* |

---

## Q11: Why combine Zobrist Transposition Tables (`Stockfish-PVS`) with `RouteCrack` for speculative decoding, and does it pollute L3 cache?

See **[`speculative/ROUTECRACK_TECHNIQUES.md`](speculative/ROUTECRACK_TECHNIQUES.md)** for the full 96-technique catalog and ablations:

1. **Why Small Neural Draft Models Underperform on CPU**:
   - GPU speculative decoding (`EAGLE-2`, `Medusa`) assumes `3+ TB/s` HBM where a 1B neural draft model runs in `< 1 ms`. On Dual-Channel DDR5 (`71.7 GB/s`), streaming a 600 MB 1B draft model for 5 steps (`3.0 GB` read) takes **~40 ms — slower than a forward pass of the 30B MoE target model itself (`~25 ms`)**, while flushing the 64 MB L3 cache.
   - `RouteCrack`'s 8-level FNV-64 Hash Cascade + Online Arena Trie + Two-Zone Context Matcher narrows the 131,072-token vocabulary down to $4\text{–}8$ candidate tokens in `< 0.01 ms`, while **`StockfishDrafter` (`rc_stockfish.h`)** acts as an orthogonal live-session verifier.
2. **Cache Footprint (`192 KB` = `< 19%` of One Core's 1 MB L2 Cache)**:
   - [`StockfishDrafter`](speculative/cpp/rc_stockfish.h) occupies **exactly `192 KB`** (`3 × 4,096 × 16 B = 3 × 64 KB` Zobrist Transposition Tables `tt5/tt4/tt3` storing `key32`, `best_move`, `refuted_move`, `history`, `wins`, `total`), fitting inside a single Zen 5 core's `1 MB` L2 cache (`0.3%` of the 64 MB L3) with **zero heap allocations** during inference.
3. **Four `RouteCrack` Techniques Enabled by `Stockfish-PVS v2` (`M=7`)**:
   - Because `tt5/tt4/tt3` are updated continuously from target-model verifications in the active conversation rather than static training text, they provide an independent filter that enables **Online Session Trie Rescue (`#1`, `92.19%` precision)**, **Top-4 Cascade Disambiguation (`#2`)**, **11,382-Entry Macro Codebook Latching (`#5 / #89`, `96.44%` precision)**, and **Singular Horizon Extension (`#17 / #73`, `M=7`, `98.74%` marginal accuracy at $D=5..7$)**.

---

## Q12: How do `Fractal-64` (`GGML_TYPE_Q2_0 = 42`, `3.75 bpw`) and `LUT-VQ3-64` (`GGML_TYPE_VQ3_0 = 43`, `3.00 bpw`) retain `100/100` Quality Gate parity, and why do 8-CCD Threadrippers need `-t == -tb` + per-CCD `taskset`?

1. **`Fractal-64` (`block_q2_0`, `30 B / 64 W = 3.75 bpw`) & `LUT-VQ3-64` (`block_vq3_0`, `24 B / 64 W = 3.00 bpw`)**:
   - Standard `Q3_K` (`3.44 bpw`) fails on `Nemotron-3.5-30B-A3B` because its `256`-weight super-block does not divide the `1856` MoE intermediate dimension (`1856 % 256 = 64`), forcing expert tensors to fall back to `Q4_0`/`Q8_0`.
   - **`Fractal-64` (`GGML_TYPE_Q2_0 = 42`)** uses a **64-weight block (`1856 = 29 × 64`, `2688 = 42 × 64`, zero fallback tensors)**: `2B` FP16 super-scale + `4B` (`8 × 4-bit` shifted `{4..19}` sub-scales) + `24B` (`qs[16]` + `qh[8]`) indexing `{-14, -8, -4, -1, +1, +4, +8, +14}`, achieving **`17.75 dB` SQNR** on `BF16` weights (`+1.45 dB` over `Q3_K`) and shrinking Nemotron to **`14.36 GiB`** (`100/100` Quality Gate).
   - **`LUT-VQ3-64` (`GGML_TYPE_VQ3_0 = 43`)** packs `64` weights into **`24 bytes` (`3.00 bpw`)**: `2B` FP16 super-scale `d` + `22B` (`qs[22]`, packing `20 × 7-bit` 3D + `2 × 6-bit` 2D spherical Lloyd-Max indices in the low 7 bits and `8 × 3-bit` `{2..9}` sub-scales across bit 7 + tail half-codebooks), decoded in 6 ZMM registers via single-cycle AVX-512 VBMI `VPERMI2B` (`13.57 dB` SQNR, `+3.42 dB` over `Q2_K`), shrinking Nemotron to **`13.30 GiB` Hybrid (`13.08 GiB` full `up_exps`, `16.04 GiB` `VmRSS`, `100/100` Quality Gate)** and `DeepSeek-V4 284B` to **`56.40 GiB` Hybrid (`56.07 GiB` aggressive, `+16.9%` raw decode)**.
2. **8-CCD Threadripper PRO 7985WX (`80.14–111.62 tok/s` Decode / `386.00 tok/s` Prefill) Thread Placement**:
   - On 8-CCD workstations (`64` cores = `8 CCDs × 8 cores`), setting `-t 16` while leaving `-tb 64` causes `llama.cpp` to switch between the 16-thread and 64-thread pools during speculative batch verification ($N > 1$), leaving `48` extra threads spinning in `GGML_DEFAULT_N_SPIN`.
   - Setting `-t == -tb` (`-t 32 -tb 32`, `4C/CCD`) and pinning symmetrically across all 8 CCDs (`scripts/llama_modchip.py`) eliminates thread-pool handoff spinning and utilizes all 8 memory controllers.

---

## Q13: How do `306.8 tok/s` Prefill, `48.39–60.40 tok/s` Speculative Decode, and `DeepSeek-V4 284B` in `< 59 GiB` RAM obey physical DDR5 bandwidth (`~75 GB/s`) and RAM limits?

Applying the **single-token ($N=1$, GEMV) bandwidth formula** (`Throughput × ActiveBytesPerToken`) to **batched prompt prefill ($N=512$, GEMM)** or **multi-token speculative verification ($N=8$)**, or assuming all `284.33B` weights in `DeepSeek-V4-Flash` use a uniform bitrate, leads to misleading estimates:

1. **Why `241.0–306.8 tok/s` Prompt Prefill Uses `~10.8 GB/s` of DDR5 Bandwidth (GEMM vs. GEMV)**:
   - During **Prompt Prefill (`-ub 512 -b 2048`)**, `llama.cpp` executes **batched Matrix-Matrix multiplications (GEMM)** across $B = 512$ tokens simultaneously. Even if every expert in the `17.60 GiB` model is touched once during a 512-token micro-batch, weights are streamed from DDR5 **once per 512 tokens**:
     $$\text{DDR5 Read per Prefill Token} \le \frac{17.60\text{ GiB}}{512\text{ tokens}} \approx 35.2\text{ MB/token} \implies 306.8\text{ tok/s} \times 35.2\text{ MB} \approx \mathbf{10.8\text{ GB/s}}$$
   - At $B = 512$, arithmetic intensity rises from $I \approx 1\text{ FLOP/byte}$ (GEMV) to **$I > 160\text{ FLOPs/byte}$ (GEMM)**: prefill is compute-bound on AVX-512 cores, not DDR5 bandwidth-bound (`386.00 tok/s` on Threadripper 7985WX, [#1](https://github.com/synchopate/llama-modchip/issues/1); `101.67 tok/s` on 8-vCPU Graviton3, [#2](https://github.com/synchopate/llama-modchip/issues/2)).

2. **Why `48.39 tok/s` Avg / `60.40 tok/s` Peak Speculative Decode Uses `~45–52 GB/s` of DDR5 Bandwidth**:
   - When `RouteCrack + Stockfish-PVS` drafts $D = 6$ tokens (`M=7`, verifying $1 + 6 = 7$ tokens in **one batched forward pass**, `mean len = 6.90` tokens/step at `96.4%` acceptance), the CPU runs **one** batched verification pass rather than 7 separate passes.
   - Across a 7-token batch in `Nemotron-3.5-30B-A3B`, all **dense and shared tensors** (`token_embd`, 23 `Mamba-2` `ssm_in`/`ssm_out` layers, 6 `GQA` layers, 23 double-width `shared experts`, and `output.weight`) are read from DDR5 **once** for all 7 tokens, while overlapping routed experts bring total batch reads to `~3.1–3.4 GB` (`~0.48 GB` per accepted token, `~23–52 GB/s` bus utilization).

3. **Why `DeepSeek-V4-Flash` (`284.33B` Total / `13.17B` Active) Occupies `56.40–56.93 GiB` on Disk (`58.3–59.1 GiB VmRSS`, `0` Swap)**:
   - `DeepSeek-V4-Flash` is quantized heterogeneously by tensor sensitivity ([`scripts/dsv4_singlepass.c`](scripts/dsv4_singlepass.c), [`scripts/build_dsv4_maxq_singlepass.py`](scripts/build_dsv4_maxq_singlepass.py), [`scripts/transcode_dsv4.c`](scripts/transcode_dsv4.c), [`scripts/build_dsv4_vq3.py`](scripts/build_dsv4_vq3.py)):
     - **Dense MLA & Indexer Compressor projections** (`~11.8B` params) use **`Fractal-64` (`Q2_0 = 42`, `3.75 bpw`)** (`~5.5 GiB`).
     - **Dense Bottlenecks (`token_embd`, `output`, `attn_q_a`, & Shared Experts `shexp`, `174` tensors)** use single-pass **`Q6_K` (`173` tensors) + `Q8_0` (`indexer_k_b`)** directly from `FP8/BF16` in the default **`MaxQ-Fractal` (`56.93 GiB` / `61,129,364,000 B`)** and opt-in `Abliterated-Fractal` builds, or **`LUT-VQ3-64` (`VQ3_0 = 43`, `3.00 bpw`)** on the 68 interior `blk.5..38` `gate`/`up` projections in the `56.40 GiB` (`60,558,856,608 B`) VQ3 Hybrid (`Q5_K`/`Q6_K` in the `56.57 GiB` Fractal baseline).
     - **256-Expert Routed MoE layers** (`~265B` params) use **`IQ1_S` (`1.5625 bpw`)** weighted by `@antirez`'s 1.5M-token chat-v2 `imatrix` (in `MaxQ-Fractal` / `Abliterated-Fractal`) or the analytical second-moment SwiGLU proxy $I_{e,j} = \|W_{\text{gate}}^{(e)}[j,:]\|_2^2 \cdot \|W_{\text{up}}^{(e)}[j,:]\|_2^2$, with sensitivity-anchored layers in **`IQ1_M` (`1.75 bpw`)** and **`IQ2_XXS` (`2.06 bpw`)** (`~48.2 GiB`).
   - Total resident anonymous-heap memory is **`58.31–59.13 GiB VmRSS`** at `-c 32768` (`58.9–59.9 GiB` at `-c 262144`) inside a `64 GB` host with zero swap.

---

## Q14: How does the 4-Level Register (`ZMM` AVX-512BW/VBMI) + L1D + L2 + L3 Tiler accelerate `DeepSeek-V4-Flash 284B` prefill (`41.1–44.0 tok/s` cold, `44.36 tok/s` warm prefix)?

1. **Bottleneck Breakdown**:
   - In a 284B MoE (`13.17B` active per token), prompt prefill ($N > 1$) and multi-turn agentic turns spend ~85% of wall-clock time inside dense (`Q2_0`, `VQ3_0`, `Q5_K`, `Q6_K`) and sparse MoE (`IQ1_S`, `IQ2_XXS`, `IQ1_M`) matrix-matrix contractions, and ~15% inside `FLASH_ATTN_EXT`, `quantize_row_q8_K`, `RMS_NORM`, and `DSV4_HC_PRE/POST`.
2. **4-Level Co-Design**:
   - **Level 0 (`ZMM` Register-Stationary AVX-512BW/VBMI `_1x8`, `_1x4`, `_1x3`, `_1x2`, `1x1`)**: Unpacks each weight block **once in `ZMM` registers** across up to `8` token columns (`64` weights per SIMD instruction).
   - **Levels 1 & 2 (L1D `{8, 4, 3, 2}` Zero-Remainder Stepping + L2 `128`-Row Tiling + Cache-Aligned MoE Dynamic Queue)**: Decomposes any column count $M \ge 2$ into $\{8, 4, 3, 2\}$ ($5 = 3+2$, $9 = 4+3+2$), eliminating `1x1` fallback across ragged 256-expert MoE buckets (`MUL_MAT_ID`).
   - **Register-Stationary AVX-512 `MQA8` FlashAttention & Token-Affinity `HC_PRE/POST -> RMS_NORM`**: Converts each 16-element `FP16` K/V slice once per 8 query heads (`FLASH_ATTN_EXT` `2.0x` faster: `415.1 ms -> 208.5 ms`), vectorizes `quantize_row_q8_K` in AVX-512, and pins token `it` to thread `ith = it % nth` across `DSV4_HC_POST -> DSV4_HC_PRE -> RMS_NORM`.
   - **Measured Result**: Cold prefill rises from `24.73–25.19 tok/s` to **`41.11–43.98 tok/s` (`+77.8%`)**, and multi-turn agentic turns with KV prefix reuse sustain **`44.36 tok/s` prefill** (`339` new / `403` total KV tokens) and **`9.05–9.55 tok/s` unpruned `Top-6` decode** (`19.4–25.3 tok/s` with `Stockfish-PVS` / tool bursts).

---

*Navigation: [Back to Overview & Quickstart (`README.md`)](README.md) · [Technical Paper (`ARCHITECTURE.md`)](ARCHITECTURE.md) · [`RouteCrack + Stockfish-PVS` Techniques (`speculative/ROUTECRACK_TECHNIQUES.md`)](speculative/ROUTECRACK_TECHNIQUES.md) · [Release Notes (`RELEASE_NOTES.md`)](RELEASE_NOTES.md)*
