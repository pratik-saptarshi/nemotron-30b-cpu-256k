#!/usr/bin/env python3
"""
eval_perplexity_and_kld.py — Continuous End-to-End Perplexity & Logit KL-Divergence Harness

Addresses the ceiling effect of pass/fail functional unit-test gates by measuring two
continuous, non-saturating end-to-end signals on quantized GGUF checkpoints:

1. Teacher-Forced Perplexity (PPL +/- std) via `llama-perplexity` on WikiText-2 (`wiki.test.raw`)
   or any custom evaluation corpus (`--corpus`).
2. End-to-End Next-Token Logit KL-Divergence D_KL(P_ref || P_cand) [nats/token] and Top-1 / Top-5
   token agreement across a fixed 32-prompt multi-domain suite (systems C/C++/Rust/Go/Python code,
   SQL, math derivations, algorithms, networking, and JSON tool-call prefixes), evaluating both
   Step-0 prompt-end distributions (32 probes, plus classic-12 subset) and all prefix-identical
   greedy continuation positions (~180–256 exact teacher-forced token positions).
   Records model SHA-256 provenance metadata in both reference logit snapshots and KLD JSONs.
"""

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

WIKITEXT2_URLS = [
    "https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip",
    "https://cosmo.zip/pub/datasets/wikitext-2-raw-v1.zip",
]

# Fixed deterministic 32-probe multi-domain contexts for next-token logit KLD.
# Probes 0..11 are the original Classic-12 suite; probes 12..31 expand statistical power across
# systems programming, numerical linear algebra, compilers, databases, networking, and tool calls.
KLD_PROBES: List[Tuple[str, str]] = [
    # --- Classic 12 ---
    (
        "C_SIMD_Dot",
        "static inline float dot_f32_avx2(const float * a, const float * b, int n) {\n"
        "    __m256 acc = _mm256_setzero_ps();\n"
        "    for (int i = 0; i + 8 <= n; i += 8) {\n"
        "        acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), ",
    ),
    (
        "Py_Async_Pool",
        "import asyncio\n\nclass BoundedWorkerPool:\n"
        "    def __init__(self, concurrency: int = 8) -> None:\n"
        "        self._sem = asyncio.Semaphore(concurrency)\n\n"
        "    async def map_ordered(self, fn, items):\n"
        "        async def _run(idx, item):\n"
        "            async with self._sem:\n"
        "                return idx, await ",
    ),
    (
        "Py_LRU_Lock",
        "from collections import OrderedDict\nimport threading\n\nclass ThreadSafeLRU:\n"
        "    def __init__(self, capacity: int):\n"
        "        self.capacity = capacity\n"
        "        self.cache = OrderedDict()\n"
        "        self.lock = threading.Lock()\n\n"
        "    def get(self, key: int) -> int:\n"
        "        with self.lock:\n"
        "            if key not in self.cache:\n"
        "                return -1\n"
        "            self.cache.move_to_end(",
    ),
    (
        "SQL_Window_Rank",
        "SELECT department_id, employee_id, salary,\n"
        "       DENSE_RANK() OVER (PARTITION BY department_id ORDER BY salary DESC) AS rk\n"
        "FROM employees\n"
        "WHERE hire_date >= DATE '2024-01-01'\n"
        "QUALIFY rk <= ",
    ),
    (
        "Math_Euler_Totient",
        "To count integers 1 <= n <= 200 coprime to 30 = 2 * 3 * 5, Euler's totient gives "
        "phi(30) = 30 * (1 - 1/2) * (1 - 1/3) * (1 - 1/5) = ",
    ),
    (
        "Systems_KV_Bandwidth",
        "For 43 MLA layers with compressed KV latent dimension 512 + 64 RoPE in Q8_0 (34 bytes per 32 elements), "
        "each token stores 576 / 32 * 34 = ",
    ),
    (
        "JSON_Tool_Schema",
        '{"tool_calls": [{"type": "function", "function": {"name": "Edit", "arguments": "{\\"file_path\\": '
        '\\"/opt/app/settings.py\\", \\"old_string\\": \\"DEBUG = True\\", \\"replace_all\\": ',
    ),
    (
        "Rust_Lifetime_Slice",
        "pub fn split_frame<'a>(buf: &'a [u8], hdr_len: usize) -> Option<(&'a [u8], &'a [u8])> {\n"
        "    if buf.len() < hdr_len {\n"
        "        return None;\n"
        "    }\n"
        "    Some(buf.split_at(",
    ),
    (
        "Linux_Proc_Meminfo",
        "On a 64 GiB Linux workstation, `/proc/meminfo` exposes `MemAvailable`, which estimates reclaimable "
        "page cache plus free pages without triggering the kernel OOM ",
    ),
    (
        "Quant_Lloyd_Max",
        "In 3-dimensional vector quantization, encoding triples of heavy-tailed weights jointly onto 128 "
        "Lloyd-Max centroids eliminates the Cartesian cube corner waste of independent scalar ",
    ),
    (
        "Algo_Dijkstra_Heap",
        "import heapq\n\ndef shortest_path(n: int, adj: list[list[tuple[int, int]]], src: int) -> list[int]:\n"
        "    dist = [10**18] * n\n"
        "    dist[src] = 0\n"
        "    pq = [(0, src)]\n"
        "    while pq:\n"
        "        d, u = heapq.heappop(pq)\n"
        "        if d > dist[u]:\n"
        "            continue\n"
        "        for v, w in adj[u]:\n"
        "            if d + w < dist[",
    ),
    (
        "HTTP_Chunked_Parser",
        "In RFC 9112 HTTP/1.1 message framing, when both `Transfer-Encoding: chunked` and `Content-Length` "
        "headers are present in a request, a compliant proxy must reject the message to prevent request ",
    ),
    # --- Extended 20 Probes (12..31) ---
    (
        "CPP_RingBuffer_Acquire",
        "template <typename T, size_t Cap>\nclass SpscRing {\n"
        "    alignas(64) std::atomic<size_t> head_{0};\n"
        "    alignas(64) std::atomic<size_t> tail_{0};\n"
        "    T buf_[Cap];\npublic:\n"
        "    bool pop(T & out) noexcept {\n"
        "        const size_t h = head_.load(std::memory_order_relaxed);\n"
        "        if (h == tail_.load(std::memory_order_",
    ),
    (
        "Go_Context_Cancel",
        "func fetchAll(ctx context.Context, urls []string) ([][]byte, error) {\n"
        "    ctx, cancel := context.WithCancel(ctx)\n"
        "    defer ",
    ),
    (
        "SQL_Recursive_CTE",
        "WITH RECURSIVE org_tree AS (\n"
        "    SELECT id, manager_id, name, 0 AS depth FROM staff WHERE manager_id IS NULL\n"
        "    UNION ALL\n"
        "    SELECT s.id, s.manager_id, s.name, t.depth + 1\n"
        "    FROM staff s INNER JOIN org_tree t ON s.manager_id = ",
    ),
    (
        "Math_Geometric_Series",
        "For |r| < 1, differentiating sum_{k=0}^{infty} r^k = 1 / (1 - r) with respect to r yields "
        "sum_{k=1}^{infty} k * r^{k-1} = ",
    ),
    (
        "Math_Eigen_Trace",
        "If A is a 3x3 real symmetric matrix with eigenvalues lambda_1 = 4, lambda_2 = -1, and lambda_3 = 7, "
        "then trace(A) = 10 and det(A) = ",
    ),
    (
        "Py_Bisect_JobDP",
        "from bisect import bisect_right\n\n"
        "def schedule_weighted(jobs: list[tuple[int, int, int]]) -> int:\n"
        "    jobs.sort(key=lambda x: x[1])\n"
        "    ends = [e for _, e, _ in jobs]\n"
        "    dp = [0] * (len(jobs) + 1)\n"
        "    for i, (s, e, w) in enumerate(jobs, 1):\n"
        "        prev = bisect_right(ends, s, 0, i - 1)\n"
        "        dp[i] = max(dp[i - 1], ",
    ),
    (
        "C_Mmap_Hugepage",
        "void * alloc_aligned_thp(size_t nbytes) {\n"
        "    void * ptr = NULL;\n"
        "    if (posix_memalign(&ptr, 2 * 1024 * 1024, nbytes) != 0) return NULL;\n"
        "    madvise(ptr, nbytes, ",
    ),
    (
        "Rust_Iter_Fold",
        "pub fn xor_checksum(payload: &[u8]) -> u8 {\n"
        "    payload.iter().copied().fold(0u8, |acc, b| ",
    ),
    (
        "JSON_Bash_ToolCall",
        '{"tool_calls": [{"id": "call_01", "type": "function", "function": {"name": "Bash", '
        '"arguments": "{\\"command\\": \\"git status --short\\", \\"timeout\\": ',
    ),
    (
        "POSIX_Open_NoFollow",
        "To prevent symlink race attacks in a shared directory on Linux, a privileged or multi-user daemon "
        "should pass `O_NOFOLLOW` to `open()` and verify ownership on the descriptor via ",
    ),
    (
        "Numerical_LogSumExp",
        "import math\n\ndef logsumexp(logits: list[float]) -> float:\n"
        "    m = max(logits)\n"
        "    return m + math.log(sum(math.exp(x - ",
    ),
    (
        "Compiler_SSA_Phi",
        "In Static Single Assignment (SSA) intermediate representation, control-flow join blocks merge "
        "distinct reaching definitions of a variable at the dominance frontier by inserting ",
    ),
    (
        "TCP_Congestion_BBR",
        "Unlike loss-based TCP CUBIC, which halves its congestion window upon packet loss, BBR paces "
        "transmissions using online estimates of bottleneck bandwidth and minimum round-trip ",
    ),
    (
        "Arch_AVX512_VBMI",
        "On AMD Zen 4/5 and Intel Ice Lake+, `_mm512_permutex2var_epi8` (`vpermi2b`) performs a 64-way "
        "byte table lookup across two 512-bit ZMM registers in a single SIMD ",
    ),
    (
        "Py_Tarjan_Topo",
        "def has_cycle_directed(n: int, adj: list[list[int]]) -> bool:\n"
        "    color = [0] * n  # 0=unvisited, 1=visiting, 2=done\n"
        "    def dfs(u: int) -> bool:\n"
        "        color[u] = 1\n"
        "        for v in adj[u]:\n"
        "            if color[v] == 1 or (color[v] == 0 and dfs(v)):\n"
        "                return True\n"
        "        color[u] = ",
    ),
    (
        "DB_MVCC_Snapshot",
        "Under PostgreSQL Multi-Version Concurrency Control (MVCC), readers never block writers because "
        "each statement observes tuple versions (`xmin`, `xmax`) committed prior to its transaction ",
    ),
    (
        "Crypto_HMAC_ConstTime",
        "import hmac\n\ndef verify_webhook_sig(secret: bytes, body: bytes, hex_sig: str) -> bool:\n"
        "    expected = hmac.new(secret, body, 'sha256').hexdigest()\n"
        "    return hmac.compare_digest(",
    ),
    (
        "LLM_Mamba_State",
        "Whereas standard causal self-attention scales KV cache memory linearly O(L) with sequence length L, "
        "a Mamba-2 selective state-space layer updates a recurrent state of constant ",
    ),
    (
        "Bash_Trap_Cleanup",
        "#!/usr/bin/env bash\nset -euo pipefail\ntmp_dir=\"$(mktemp -d)\"\n"
        "cleanup() { rm -rf \"$tmp_dir\"; }\ntrap cleanup ",
    ),
    (
        "Italian_Quant_Prose",
        "Nella quantizzazione vettoriale di modelli MoE da 284 miliardi di parametri, l'uso di una matrice "
        "di importanza calibrata sulle attivazioni reali consente di preservare i pesi più sensibili senza ",
    ),
]

CLASSIC_12_NAMES = {name for name, _ in KLD_PROBES[:12]}


def compute_file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(8 * 1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def query_server_model_id(base_url: str) -> str:
    try:
        req = urllib.request.Request(f"{base_url.rstrip('/')}/v1/models", method="GET")
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        data = body.get("data") or []
        if data and isinstance(data[0], dict):
            return str(data[0].get("id", ""))
    except Exception:
        pass
    return ""


def ensure_default_corpus(cache_dir: str) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    target = os.path.join(cache_dir, "wiki.test.raw")
    if os.path.isfile(target) and os.path.getsize(target) > 1_000_000:
        with open(target, "r", encoding="utf-8", errors="replace") as f:
            head_sample = f.read(8192)
        if "<unk>" not in head_sample:
            return target

    import io
    import zipfile

    for url in WIKITEXT2_URLS:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "llama-modchip-eval/2.5.3"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw = resp.read()
            with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                for name in zf.namelist():
                    if name.endswith("wiki.test.raw"):
                        data = zf.read(name)
                        sample = data[:8192].decode("utf-8", errors="replace")
                        if "<unk>" in sample:
                            raise RuntimeError(f"Rejected pre-tokenized <unk> corpus from {url}")
                        with open(target, "wb") as out:
                            out.write(data)
                        return target
        except Exception as exc:
            sys.stderr.write(f"[!] Corpus download fallback ({url}): {exc}\n")
    raise RuntimeError("Could not download genuine raw WikiText-2 (wiki.test.raw); pass --corpus /path/to/wiki.test.raw")


def run_llama_perplexity(
    bin_path: str,
    model_path: str,
    corpus_path: str,
    ctx_size: int = 512,
    chunks: int = 8,
    threads: int = 16,
) -> Dict[str, Any]:
    if not os.path.isfile(bin_path):
        raise FileNotFoundError(f"llama-perplexity binary not found: {bin_path}")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"GGUF model not found: {model_path}")

    cmd = [
        bin_path,
        "-m", model_path,
        "-f", corpus_path,
        "-c", str(ctx_size),
        "-b", str(ctx_size),
        "--chunks", str(chunks),
        "-t", str(threads),
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    wall = round(time.perf_counter() - t0, 2)
    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
    m = re.search(r"Final estimate:\s*PPL\s*=\s*([0-9.]+)\s*\+/-\s*([0-9.]+)", combined)
    if not m:
        raise RuntimeError(f"llama-perplexity failed (rc={proc.returncode}):\n{combined[-800:]}")
    return {
        "ppl": float(m.group(1)),
        "ppl_std": float(m.group(2)),
        "ctx_size": ctx_size,
        "chunks": chunks,
        "eval_tokens": ctx_size * chunks,
        "corpus": os.path.basename(corpus_path),
        "wall_sec": wall,
    }


def collect_completion_logprobs(base_url: str, top_k: int = 64, steps: int = 8) -> List[Dict[str, Any]]:
    """Collect multi-step top-k token log-probabilities from llama-server `/v1/completions`."""
    records: List[Dict[str, Any]] = []
    endpoint = f"{base_url.rstrip('/')}/v1/completions"
    for name, prompt in KLD_PROBES:
        payload = {
            "prompt": prompt,
            "max_tokens": steps,
            "temperature": 0.0,
            "logprobs": top_k,
            "stream": False,
        }
        req = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        choice = (body.get("choices") or [{}])[0]
        lp_obj = choice.get("logprobs") or {}
        top_list = lp_obj.get("top_logprobs") or []
        chosen_tokens: List[str] = list(lp_obj.get("tokens") or [])
        if not top_list and "content" in lp_obj:
            chosen_tokens = []
            for item in lp_obj["content"]:
                chosen_tokens.append(str(item.get("token", "")))
                step_map = {}
                for cand in item.get("top_logprobs", []):
                    tok = cand.get("token", "")
                    if tok not in step_map:
                        step_map[tok] = float(cand.get("logprob", -30.0))
                if step_map:
                    top_list.append(step_map)
        records.append({
            "probe": name,
            "prompt": prompt,
            "generated_text": choice.get("text", ""),
            "chosen_tokens": chosen_tokens,
            "steps": top_list,
        })
    return records


def _step_kld_and_agreement(p_map: Dict[str, float], q_map: Dict[str, float]) -> Tuple[float, str, str, int, int]:
    max_lp_p = max(p_map.values())
    p_unnorm = {k: math.exp(v - max_lp_p) for k, v in p_map.items()}
    z_p = sum(p_unnorm.values()) or 1.0
    p_probs = {k: v / z_p for k, v in p_unnorm.items()}

    min_q_lp = min(q_map.values()) - 2.0 if q_map else -20.0
    max_lp_q = max(q_map.values()) if q_map else 0.0
    union_keys = set(p_map.keys()) | set(q_map.keys())
    q_unnorm = {k: math.exp(q_map.get(k, min_q_lp) - max_lp_q) for k in union_keys}
    z_q = sum(q_unnorm.values()) or 1.0
    q_probs = {k: max(1e-12, v / z_q) for k, v in q_unnorm.items()}

    kl = 0.0
    for tok, p_val in p_probs.items():
        if p_val > 0.0:
            kl += p_val * math.log(p_val / q_probs[tok])
    kl = max(0.0, kl)

    ref_sorted = sorted(p_map.items(), key=lambda kv: kv[1], reverse=True)
    cand_sorted = sorted(q_map.items(), key=lambda kv: kv[1], reverse=True)
    ref_top1 = ref_sorted[0][0]
    cand_top1 = cand_sorted[0][0]
    cand_top5 = {k for k, _ in cand_sorted[:5]}
    return kl, ref_top1, cand_top1, int(ref_top1 == cand_top1), int(ref_top1 in cand_top5)


def load_logits_snapshot(path: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return {"source_file": os.path.basename(path)}, obj
    if isinstance(obj, dict) and "records" in obj:
        return dict(obj.get("metadata") or {}), list(obj["records"])
    raise ValueError(f"Unrecognized logits JSON structure in {path}")


def compute_kld_from_records(
    ref_records: List[Dict[str, Any]],
    cand_records: List[Dict[str, Any]],
    ref_meta: Optional[Dict[str, Any]] = None,
    cand_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Compute truncated top-K D_KL(P_ref || P_cand) across:
      1. Step-0 prompt-end distributions on all probes (100% identical prompt conditioning).
      2. Classic-12 subset Step-0 mean KLD for backward continuity with prior reports.
      3. All prefix-matched continuation positions (steps t=0..T-1 up to the first greedy
         token divergence, guaranteeing 100% identical teacher-forced prefix conditioning).
    """
    ref_by_name = {r["probe"]: r for r in ref_records}
    cand_by_name = {r["probe"]: r for r in cand_records}

    klds_step0: List[float] = []
    klds_classic12: List[float] = []
    klds_matched_all: List[float] = []
    top1_hits = 0
    top5_hits = 0
    matched_top1_hits = 0
    matched_top5_hits = 0
    per_probe = []

    for name, ref_r in ref_by_name.items():
        cand_r = cand_by_name.get(name)
        if not cand_r or not ref_r.get("steps") or not cand_r.get("steps"):
            continue
        ref_steps = ref_r["steps"]
        cand_steps = cand_r["steps"]
        kl0, ref_top1, cand_top1, t1, t5 = _step_kld_and_agreement(ref_steps[0], cand_steps[0])
        klds_step0.append(kl0)
        if name in CLASSIC_12_NAMES:
            klds_classic12.append(kl0)
        top1_hits += t1
        top5_hits += t5

        # Accumulate all prefix-identical positions starting at t=0
        ref_chosen = ref_r.get("chosen_tokens") or []
        cand_chosen = cand_r.get("chosen_tokens") or []
        max_t = min(len(ref_steps), len(cand_steps))
        probe_matched_steps = 0
        for t in range(max_t):
            if t == 0:
                kl_t, _, _, mt1, mt5 = kl0, ref_top1, cand_top1, t1, t5
            else:
                if not ref_steps[t] or not cand_steps[t]:
                    break
                kl_t, _, _, mt1, mt5 = _step_kld_and_agreement(ref_steps[t], cand_steps[t])
            klds_matched_all.append(kl_t)
            matched_top1_hits += mt1
            matched_top5_hits += mt5
            probe_matched_steps += 1
            # Advance to t+1 ONLY if greedy chosen token at step t is identical
            r_tok = ref_chosen[t] if t < len(ref_chosen) else sorted(ref_steps[t].items(), key=lambda kv: kv[1], reverse=True)[0][0]
            c_tok = cand_chosen[t] if t < len(cand_chosen) else sorted(cand_steps[t].items(), key=lambda kv: kv[1], reverse=True)[0][0]
            if r_tok != c_tok:
                break

        per_probe.append({
            "probe": name,
            "kld_nats": round(kl0, 6),
            "matched_prefix_steps": probe_matched_steps,
            "ref_top1": ref_top1,
            "cand_top1": cand_top1,
            "top1_match": bool(t1),
            "top5_match": bool(t5),
        })

    n = max(1, len(klds_step0))
    mean_kld = sum(klds_step0) / n
    n_c12 = max(1, len(klds_classic12))
    mean_c12 = sum(klds_classic12) / n_c12
    n_all = max(1, len(klds_matched_all))
    mean_matched = sum(klds_matched_all) / n_all

    res: Dict[str, Any] = {
        "num_probes": len(klds_step0),
        "mean_kld_nats": round(mean_kld, 6),
        "equiv_ppl_delta_pct": round((math.exp(mean_kld) - 1.0) * 100.0, 3),
        "top1_agreement_pct": round(100.0 * top1_hits / n, 2),
        "top5_agreement_pct": round(100.0 * top5_hits / n, 2),
        "classic12_num_probes": len(klds_classic12),
        "classic12_mean_kld_nats": round(mean_c12, 6),
        "prefix_matched_positions": len(klds_matched_all),
        "prefix_matched_mean_kld_nats": round(mean_matched, 6),
        "prefix_matched_equiv_ppl_delta_pct": round((math.exp(mean_matched) - 1.0) * 100.0, 3),
        "prefix_matched_top1_agreement_pct": round(100.0 * matched_top1_hits / n_all, 2),
        "prefix_matched_top5_agreement_pct": round(100.0 * matched_top5_hits / n_all, 2),
    }
    if ref_meta:
        res["reference_model"] = ref_meta
    if cand_meta:
        res["candidate_model"] = cand_meta
    res["probes"] = per_probe
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description="Continuous Perplexity & Logit KL-Divergence Evaluation")
    ap.add_argument("--llama-perplexity", default="", help="Path to llama-perplexity binary")
    ap.add_argument("--model", default="", help="Path to GGUF model (shard 1 for multi-shard)")
    ap.add_argument("--model-sha256", default="", help="Optional precomputed SHA-256 of model/shard-1 for provenance metadata")
    ap.add_argument("--label", default="", help="Human-readable checkpoint label for provenance metadata")
    ap.add_argument("--corpus", default="", help="Evaluation text file (defaults to cached WikiText-2 test)")
    ap.add_argument("--ctx-size", type=int, default=512, help="Perplexity context window (default: 512)")
    ap.add_argument("--chunks", type=int, default=8, help="Number of context chunks to evaluate (default: 8 = 4096 tok)")
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 16, help="CPU threads for llama-perplexity")
    ap.add_argument("--url", default="", help="Live llama-server base URL (e.g. http://127.0.0.1:8080) for Logit KLD")
    ap.add_argument("--steps", type=int, default=8, help="Max tokens collected per KLD probe (default: 8)")
    ap.add_argument("--save-logits", default="", help="Save collected reference top-logprobs JSON to path")
    ap.add_argument("--ref-logits", default="", help="Compare live --url candidate against saved reference logits JSON")
    ap.add_argument("--save-json", default="", help="Write combined continuous metrics JSON to path")
    args = ap.parse_args()

    ts = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    model_sha = args.model_sha256.strip()
    if not model_sha and args.model and os.path.isfile(args.model) and os.path.getsize(args.model) <= 20 * 1024**3:
        # Auto-hash Shard 1 (5.25 MB) or single-file <=20 GiB GGUF when --model is supplied
        model_sha = compute_file_sha256(args.model)

    report: Dict[str, Any] = {"timestamp": ts}
    if args.label:
        report["label"] = args.label
    if args.model:
        report["model"] = os.path.basename(args.model)
    if model_sha:
        report["model_sha256"] = model_sha

    if args.llama_perplexity and args.model:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        corpus = args.corpus or ensure_default_corpus(os.path.join(repo_root, "workspace", "cache"))
        print(f"[*] Running teacher-forced Perplexity ({args.chunks} x {args.ctx_size} tokens) on {os.path.basename(args.model)}...")
        ppl_res = run_llama_perplexity(
            args.llama_perplexity,
            args.model,
            corpus,
            ctx_size=args.ctx_size,
            chunks=args.chunks,
            threads=args.threads,
        )
        if model_sha:
            ppl_res["model_sha256"] = model_sha
        report["perplexity"] = ppl_res
        print(
            f"[+] Perplexity: {ppl_res['ppl']:.4f} +/- {ppl_res['ppl_std']:.5f} "
            f"({ppl_res['eval_tokens']} tokens on {ppl_res['corpus']}, {ppl_res['wall_sec']}s)"
        )

    if args.url:
        srv_model = query_server_model_id(args.url)
        cur_meta: Dict[str, Any] = {
            "label": args.label or (os.path.basename(args.model) if args.model else srv_model or "live-model"),
            "server_model_id": srv_model,
            "model_file": os.path.basename(args.model) if args.model else "",
            "model_sha256": model_sha,
            "num_probes": len(KLD_PROBES),
            "steps_per_probe": args.steps,
            "top_k": 64,
            "timestamp": ts,
        }
        print(f"[*] Sampling top-64 log-probabilities across {len(KLD_PROBES)} probes (up to {args.steps} steps/probe) from {args.url}...")
        cur_logits = collect_completion_logprobs(args.url, top_k=64, steps=args.steps)
        if args.save_logits:
            os.makedirs(os.path.dirname(os.path.abspath(args.save_logits)), exist_ok=True)
            snapshot = {"metadata": cur_meta, "records": cur_logits}
            with open(args.save_logits, "w", encoding="utf-8") as f:
                json.dump(snapshot, f, indent=2)
            print(f"[+] Saved reference logit distributions ({len(KLD_PROBES)} probes) to: {args.save_logits}")
        if args.ref_logits:
            ref_meta, ref_logits = load_logits_snapshot(args.ref_logits)
            kld_res = compute_kld_from_records(ref_logits, cur_logits, ref_meta=ref_meta, cand_meta=cur_meta)
            report["kld"] = kld_res
            print(
                f"[+] Step-0 KLD ({kld_res['num_probes']} probes): {kld_res['mean_kld_nats']:.5f} nats/tok "
                f"(+{kld_res['equiv_ppl_delta_pct']:.2f}% equiv PPL) | "
                f"Top-1: {kld_res['top1_agreement_pct']:.1f}% | Top-5: {kld_res['top5_agreement_pct']:.1f}%"
            )
            print(
                f"[+] Prefix-Matched Multi-Step KLD ({kld_res['prefix_matched_positions']} positions): "
                f"{kld_res['prefix_matched_mean_kld_nats']:.5f} nats/tok "
                f"(+{kld_res['prefix_matched_equiv_ppl_delta_pct']:.2f}% equiv PPL) | "
                f"Top-1: {kld_res['prefix_matched_top1_agreement_pct']:.1f}%"
            )

    if not ("perplexity" in report or "kld" in report or args.save_logits):
        ap.print_help()
        return 1

    if args.save_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_json)), exist_ok=True)
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"[+] Saved continuous evaluation JSON to: {args.save_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
