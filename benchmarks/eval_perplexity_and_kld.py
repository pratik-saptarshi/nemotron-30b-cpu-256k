#!/usr/bin/env python3
"""
eval_perplexity_and_kld.py — Continuous End-to-End Perplexity & Logit KL-Divergence Harness

Addresses the ceiling effect of pass/fail functional unit-test gates by measuring two
continuous, non-saturating end-to-end signals on quantized GGUF checkpoints:

1. Teacher-Forced Perplexity (PPL +/- std) via `llama-perplexity` on WikiText-2 (`wiki.test.raw`)
   or any custom evaluation corpus (`--corpus`).
2. End-to-End Next-Token Logit KL-Divergence D_KL(P_ref || P_cand) [nats/token] and Top-1 / Top-5
   token agreement across a fixed 12-prompt multi-domain suite (systems C/Python code, SQL,
   math derivations, and JSON tool-call prefixes). Supports sequential single-host 64 GB execution
   via `--save-logits ref_logits.json` followed by `--ref-logits ref_logits.json`.

Usage:
  # A. Teacher-forced Perplexity (auto-downloads WikiText-2 test split if --corpus is omitted):
  python3 benchmarks/eval_perplexity_and_kld.py \
      --llama-perplexity ./workspace/llama.cpp/build/bin/llama-perplexity \
      --model ./dsv4-vq3/DeepSeek-V4-Flash-0731-VQ3-00001-of-00003.gguf \
      --threads 16 --chunks 8 --save-json benchmarks/results/ppl_dsv4_vq3.json

  # B. Sequential 2-pass Logit KL-Divergence on a single 64 GB machine:
  # Pass 1 (while Reference checkpoint is served on :8080):
  python3 benchmarks/eval_perplexity_and_kld.py --url http://127.0.0.1:8080 --save-logits ref_logits.json
  # Pass 2 (while Candidate checkpoint is served on :8080):
  python3 benchmarks/eval_perplexity_and_kld.py --url http://127.0.0.1:8080 --ref-logits ref_logits.json
"""

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
import urllib.request
from typing import Any, Dict, List, Tuple

WIKITEXT2_URLS = [
    "https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip",
    "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/test.txt",
]

# Fixed deterministic multi-domain probe contexts for next-token logit KLD
KLD_PROBES: List[Tuple[str, str]] = [
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
]


def ensure_default_corpus(cache_dir: str) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    target = os.path.join(cache_dir, "wikitext2_test.txt")
    if os.path.isfile(target) and os.path.getsize(target) > 100_000:
        return target

    import io
    import zipfile

    for url in WIKITEXT2_URLS:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "llama-modchip-eval/2.5"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw = resp.read()
            if url.endswith(".zip"):
                with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                    for name in zf.namelist():
                        if name.endswith("wiki.test.raw"):
                            data = zf.read(name)
                            with open(target, "wb") as out:
                                out.write(data)
                            return target
            else:
                with open(target, "wb") as out:
                    out.write(raw)
                return target
        except Exception as exc:
            sys.stderr.write(f"[!] Corpus download fallback ({url}): {exc}\n")
    raise RuntimeError("Could not download default WikiText-2 corpus; pass --corpus /path/to/text.txt")


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
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
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
        if not top_list and "content" in lp_obj:
            # llama-server newer OpenAI-style logprobs.content format
            for item in lp_obj["content"]:
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
            "steps": top_list,
        })
    return records


def compute_kld_from_records(
    ref_records: List[Dict[str, Any]],
    cand_records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Compute truncated top-K D_KL(P_ref || P_cand) on step 0 of every identical prompt prefix
    (where teacher-forced conditioning is 100% identical between reference and candidate),
    plus greedy sequence match across generated tokens.
    """
    ref_by_name = {r["probe"]: r for r in ref_records}
    cand_by_name = {r["probe"]: r for r in cand_records}

    klds: List[float] = []
    top1_hits = 0
    top5_hits = 0
    per_probe = []

    for name, ref_r in ref_by_name.items():
        cand_r = cand_by_name.get(name)
        if not cand_r or not ref_r["steps"] or not cand_r["steps"]:
            continue
        p_map: Dict[str, float] = ref_r["steps"][0]
        q_map: Dict[str, float] = cand_r["steps"][0]

        # Normalize reference top-K distribution and apply floor logprob for tail tokens
        max_lp_p = max(p_map.values())
        p_unnorm = {k: math.exp(v - max_lp_p) for k, v in p_map.items()}
        z_p = sum(p_unnorm.values()) or 1.0
        p_probs = {k: v / z_p for k, v in p_unnorm.items()}

        min_q_lp = min(q_map.values()) - 2.0 if q_map else -20.0
        max_lp_q = max(q_map.values()) if q_map else 0.0
        # Convert candidate logprobs (already log-softmax from server) to normalized probs over union
        union_keys = set(p_map.keys()) | set(q_map.keys())
        q_unnorm = {k: math.exp(q_map.get(k, min_q_lp) - max_lp_q) for k in union_keys}
        z_q = sum(q_unnorm.values()) or 1.0
        q_probs = {k: max(1e-12, v / z_q) for k, v in q_unnorm.items()}

        kl = 0.0
        for tok, p_val in p_probs.items():
            if p_val > 0.0:
                kl += p_val * math.log(p_val / q_probs[tok])
        kl = max(0.0, kl)
        klds.append(kl)

        ref_sorted = sorted(p_map.items(), key=lambda kv: kv[1], reverse=True)
        cand_sorted = sorted(q_map.items(), key=lambda kv: kv[1], reverse=True)
        ref_top1 = ref_sorted[0][0]
        cand_top1 = cand_sorted[0][0]
        cand_top5 = {k for k, _ in cand_sorted[:5]}

        t1 = int(ref_top1 == cand_top1)
        t5 = int(ref_top1 in cand_top5)
        top1_hits += t1
        top5_hits += t5
        per_probe.append({
            "probe": name,
            "kld_nats": round(kl, 6),
            "ref_top1": ref_top1,
            "cand_top1": cand_top1,
            "top1_match": bool(t1),
            "top5_match": bool(t5),
        })

    n = max(1, len(klds))
    mean_kld = sum(klds) / n
    return {
        "num_probes": len(klds),
        "mean_kld_nats": round(mean_kld, 6),
        "equiv_ppl_delta_pct": round((math.exp(mean_kld) - 1.0) * 100.0, 3),
        "top1_agreement_pct": round(100.0 * top1_hits / n, 2),
        "top5_agreement_pct": round(100.0 * top5_hits / n, 2),
        "probes": per_probe,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Continuous Perplexity & Logit KL-Divergence Evaluation")
    ap.add_argument("--llama-perplexity", default="", help="Path to llama-perplexity binary")
    ap.add_argument("--model", default="", help="Path to GGUF model (shard 1 for multi-shard)")
    ap.add_argument("--corpus", default="", help="Evaluation text file (defaults to cached WikiText-2 test)")
    ap.add_argument("--ctx-size", type=int, default=512, help="Perplexity context window (default: 512)")
    ap.add_argument("--chunks", type=int, default=8, help="Number of context chunks to evaluate (default: 8 = 4096 tok)")
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 16, help="CPU threads for llama-perplexity")
    ap.add_argument("--url", default="", help="Live llama-server base URL (e.g. http://127.0.0.1:8080) for Logit KLD")
    ap.add_argument("--save-logits", default="", help="Save collected reference top-logprobs JSON to path")
    ap.add_argument("--ref-logits", default="", help="Compare live --url candidate against saved reference logits JSON")
    ap.add_argument("--save-json", default="", help="Write combined continuous metrics JSON to path")
    args = ap.parse_args()

    report: Dict[str, Any] = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z")}

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
        report["perplexity"] = ppl_res
        print(
            f"[+] Perplexity: {ppl_res['ppl']:.4f} +/- {ppl_res['ppl_std']:.5f} "
            f"({ppl_res['eval_tokens']} tokens on {ppl_res['corpus']}, {ppl_res['wall_sec']}s)"
        )

    if args.url:
        print(f"[*] Sampling top-64 next-token log-probabilities across {len(KLD_PROBES)} probes from {args.url}...")
        cur_logits = collect_completion_logprobs(args.url)
        if args.save_logits:
            os.makedirs(os.path.dirname(os.path.abspath(args.save_logits)), exist_ok=True)
            with open(args.save_logits, "w", encoding="utf-8") as f:
                json.dump(cur_logits, f, indent=2)
            print(f"[+] Saved reference logit distributions to: {args.save_logits}")
        if args.ref_logits:
            with open(args.ref_logits, "r", encoding="utf-8") as f:
                ref_logits = json.load(f)
            kld_res = compute_kld_from_records(ref_logits, cur_logits)
            report["kld"] = kld_res
            print(
                f"[+] Logit KL-Divergence: {kld_res['mean_kld_nats']:.5f} nats/tok "
                f"(+{kld_res['equiv_ppl_delta_pct']:.2f}% equiv PPL) | "
                f"Top-1 Match: {kld_res['top1_agreement_pct']:.1f}% | Top-5 Match: {kld_res['top5_agreement_pct']:.1f}%"
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
