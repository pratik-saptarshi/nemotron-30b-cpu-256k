#!/usr/bin/env python3
"""
Standalone Single-Pass 64GB GGUF Builder for DeepSeek-V4-Flash-0731 (56.93 GiB).

Reproduces both 56.93 GiB 64GB-CPU builds directly from upstream safetensors + pinned @antirez
1.5M-token routed-MoE importance matrix (NO donor GGUF weights required):

  1. Aligned Default (`--mode aligned`):
       Target:  Synchopate/DeepSeek-V4-Flash-0731-MaxQ-Fractal-64GB-CPU-GGUF
       Source:  deepseek-ai/DeepSeek-V4-Flash-0731 @ 7872f01b1d1fe23eabc4c98b48bffcef5a386062
       Imatrix: antirez/deepseek-v4-gguf @ f71f23d552d664e523b422157b2befbf74040380
                (imatrix/DeepSeek-V4-Flash-chat-v2-routed-moe-ds4-1p5m.dat,
                 SHA256: 02a7c78c29875e4653d6ce21d8821c02161e83ed90c506bdd8d275f76d4ac97e)

  2. Uncensored Opt-In (`--mode abliterated`):
       Target:  Synchopate/DeepSeek-V4-Flash-0731-Abliterated-Fractal-64GB-CPU-GGUF
       Source:  prem-research/DeepSeek-V4-Flash-0731-abliterated @ 457210de3a9d27d8142407cd0dac7b6ed6ad9325
       Imatrix: antirez/deepseek-v4-gguf @ f71f23d552d664e523b422157b2befbf74040380

Quantization Recipe Across All 1,328 Tensors (56.93 GiB across 3 GGUF shards):
  - 1,199 non-routed-MoE tensors quantized in a SINGLE PASS from native FP8_E4M3 / BF16 / F32 safetensors:
      * 173 x Q6_K + 1 x Q8_0 (174 dense 100%-active bottlenecks: token_embd, output, 43x attn_q_a, 43x3 shared_experts)
      * 317 x Q2_0 (Fractal-64, 3.75 bpw) across MLA & Lightning Indexer projections
      * 662 x F32 + 43 x BF16 + 3 x I32 across RMSNorms, MoE routers, attention sinks, Hyper-Connections, and tid2eid
  - 129 routed-MoE 3D tensors quantized in a SINGLE PASS directly from native MXFP4 + F8_E8M0 safetensors
    into IQ2_XXS / IQ1_M / IQ1_S weighted by @antirez's 1.5M-token chat-v2 routed-MoE importance matrix.

Example (reproducible from an empty directory after building libdsv4_singlepass.so):
  python3 scripts/build_dsv4_maxq_singlepass.py \\
      --mode aligned \\
      --lib-path ./workspace/bin/libdsv4_singlepass.so \\
      --out-prefix ./DeepSeek-V4-Flash-0731-MaxQ-Fractal
"""
import argparse
import ctypes
import hashlib
import json
import mmap
import os
import re
import shutil
import struct
import subprocess
import threading
import time
import urllib.request
import numpy as np

DTYPE_CODE = {"F32": 0, "BF16": 1, "F8_E4M3": 2, "I64": 3, "I32": 4}

ANTIREZ_REPO = "antirez/deepseek-v4-gguf"
ANTIREZ_REV = "f71f23d552d664e523b422157b2befbf74040380"
ANTIREZ_IMATRIX_RFILE = "imatrix/DeepSeek-V4-Flash-chat-v2-routed-moe-ds4-1p5m.dat"
ANTIREZ_IMATRIX_SHA256 = "02a7c78c29875e4653d6ce21d8821c02161e83ed90c506bdd8d275f76d4ac97e"

DEFAULT_HEADER_REPO = "Synchopate/DeepSeek-V4-Flash-0731-MaxQ-Fractal-64GB-CPU-GGUF"
DEFAULT_HEADER_PREFIX = "DeepSeek-V4-Flash-0731-MaxQ-Fractal"

ROOT_MAP = {
    "token_embd.weight":      "embed.weight",
    "output_norm.weight":     "norm.weight",
    "output.weight":          "head.weight",
    "output_hc_base.weight":  "hc_head_base",
    "output_hc_fn.weight":    "hc_head_fn",
    "output_hc_scale.weight": "hc_head_scale",
}

LAYER_MAP = {
    "attn_compressor_ape.weight":      "attn.compressor.ape",
    "attn_compressor_gate.weight":     "attn.compressor.wgate.weight",
    "attn_compressor_kv.weight":       "attn.compressor.wkv.weight",
    "attn_compressor_norm.weight":     "attn.compressor.norm.weight",
    "attn_kv.weight":                  "attn.wkv.weight",
    "attn_kv_a_norm.weight":           "attn.kv_norm.weight",
    "attn_norm.weight":                "attn_norm.weight",
    "attn_output_a.weight":            "attn.wo_a.weight",
    "attn_output_b.weight":            "attn.wo_b.weight",
    "attn_q_a.weight":                 "attn.wq_a.weight",
    "attn_q_a_norm.weight":            "attn.q_norm.weight",
    "attn_q_b.weight":                 "attn.wq_b.weight",
    "attn_sinks.weight":               "attn.attn_sink",
    "exp_probs_b.bias":                "ffn.gate.bias",
    "ffn_down_shexp.weight":           "ffn.shared_experts.w2.weight",
    "ffn_gate_inp.weight":             "ffn.gate.weight",
    "ffn_gate_shexp.weight":           "ffn.shared_experts.w1.weight",
    "ffn_gate_tid2eid.weight":         "ffn.gate.tid2eid",
    "ffn_norm.weight":                 "ffn_norm.weight",
    "ffn_up_shexp.weight":             "ffn.shared_experts.w3.weight",
    "hc_attn_base.weight":             "hc_attn_base",
    "hc_attn_fn.weight":               "hc_attn_fn",
    "hc_attn_scale.weight":            "hc_attn_scale",
    "hc_ffn_base.weight":              "hc_ffn_base",
    "hc_ffn_fn.weight":                "hc_ffn_fn",
    "hc_ffn_scale.weight":             "hc_ffn_scale",
    "indexer.attn_q_b.weight":         "attn.indexer.wq_b.weight",
    "indexer.proj.weight":             "attn.indexer.weights_proj.weight",
    "indexer_compressor_ape.weight":   "attn.indexer.compressor.ape",
    "indexer_compressor_gate.weight":  "attn.indexer.compressor.wgate.weight",
    "indexer_compressor_kv.weight":    "attn.indexer.compressor.wkv.weight",
    "indexer_compressor_norm.weight":  "attn.indexer.compressor.norm.weight",
}

MOE_PROJ = {
    "ffn_gate_exps.weight": "w1",
    "ffn_down_exps.weight": "w2",
    "ffn_up_exps.weight":   "w3",
}


def row_bytes_for_type(ttype: int, ncol: int) -> int:
    table = {
        0:  ncol * 4,
        2:  (ncol // 32) * 18,
        8:  (ncol // 32) * 34,
        12: (ncol // 256) * 144,
        13: (ncol // 256) * 176,
        14: (ncol // 256) * 210,
        16: (ncol // 256) * 66,
        19: (ncol // 256) * 50,
        26: ncol * 4,
        29: (ncol // 256) * 56,
        30: ncol * 2,
        42: (ncol // 64) * 30,
        43: (ncol // 64) * 24,
    }
    return table[ttype]


def map_gguf_to_st_key(name: str):
    if name in ROOT_MAP:
        return ROOT_MAP[name], None
    m = re.match(r"blk\.(\d+)\.(.+)$", name)
    if not m:
        raise KeyError(f"Unmapped tensor name: {name}")
    bid = int(m.group(1))
    suf = m.group(2)
    if suf in MOE_PROJ:
        proj = MOE_PROJ[suf]
        return f"layers.{bid}.ffn.experts.0.{proj}.weight", (bid, proj)
    return f"layers.{bid}.{LAYER_MAP[suf]}", None


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(16 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def download_file_verified(url: str, dst_path: str, expected_sha256: str = None) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(dst_path)), exist_ok=True)
    part = dst_path + ".part"
    if shutil.which("aria2c"):
        for stale in (part, part + ".aria2"):
            if os.path.exists(stale):
                os.unlink(stale)
        subprocess.check_call([
            "aria2c", "-x", "16", "-s", "16", "-k", "32M",
            "--file-allocation=none", "--console-log-level=warn", "--summary-interval=0",
            "-d", os.path.dirname(os.path.abspath(part)),
            "-o", os.path.basename(part),
            "--allow-overwrite=true", "--auto-file-renaming=false", url,
        ])
    else:
        req = urllib.request.Request(url, headers={"User-Agent": "llama-modchip/2.5.3"})
        with urllib.request.urlopen(req, timeout=120) as resp, open(part, "wb") as out:
            shutil.copyfileobj(resp, out, length=16 * 1024 * 1024)
    if expected_sha256:
        got = sha256_file(part)
        if got != expected_sha256:
            os.unlink(part)
            raise RuntimeError(f"SHA256 mismatch for {url}: got {got}, expected {expected_sha256}")
    os.replace(part, dst_path)


def http_range_get(url: str, start_byte: int, end_byte_excl: int) -> bytes:
    expected = end_byte_excl - start_byte
    hdrs = {
        "Range": f"bytes={start_byte}-{end_byte_excl - 1}",
        "User-Agent": "llama-modchip/2.5.3",
    }
    for attempt in range(1, 7):
        try:
            req = urllib.request.Request(url, headers=hdrs)
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
                if len(data) == expected:
                    return data
                raise RuntimeError(f"Short Range read {len(data)} != {expected}")
        except Exception:
            if attempt == 6:
                raise
            time.sleep(1.5 * attempt)
    raise RuntimeError("unreachable")


def load_imatrix(path: str) -> dict:
    im = {}
    with open(path, "rb") as f:
        n_entries = struct.unpack("<i", f.read(4))[0]
        for _ in range(n_entries):
            slen = struct.unpack("<i", f.read(4))[0]
            name = f.read(slen).decode("utf-8")
            ncall, nval = struct.unpack("<ii", f.read(8))
            arr = np.frombuffer(f.read(nval * 4), dtype=np.float32).copy()
            if ncall > 0:
                arr /= ncall
            im[name] = arr
    return im


def encode_gguf_str(s: str) -> bytes:
    sb = s.encode("utf-8")
    return struct.pack("<Q", len(sb)) + sb


def encode_gguf_str_array(items: list) -> bytes:
    out = bytearray(struct.pack("<IQ", 8, len(items)))
    for item in items:
        out.extend(encode_gguf_str(item))
    return bytes(out)


def write_shard1_metadata(
    src_bytes: bytes,
    dst_path: str,
    overrides_str: dict,
    overrides_u32: dict,
    overrides_str_arr: dict,
    insert_after_str: dict = None,
) -> None:
    insert_after_str = insert_after_str or {}
    magic, ver, n_t, n_kv = struct.unpack_from("<4sIQQ", src_bytes, 0)
    assert magic == b"GGUF" and n_t == 0, "Expected metadata-only shard 1 (n_tensors == 0)"
    off = 24
    kv_chunks = []
    seen_keys = set()

    def skip_val(vtype, pos):
        sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
        if vtype in sizes:
            return pos + sizes[vtype]
        if vtype == 8:
            slen = struct.unpack_from("<Q", src_bytes, pos)[0]
            return pos + 8 + slen
        if vtype == 9:
            atype, alen = struct.unpack_from("<IQ", src_bytes, pos)
            pos += 12
            if atype in sizes:
                return pos + sizes[atype] * alen
            for _ in range(alen):
                pos = skip_val(atype, pos)
            return pos
        raise ValueError(f"Unsupported GGUF KV type {vtype}")

    parsed = []
    for _ in range(n_kv):
        klen = struct.unpack_from("<Q", src_bytes, off)[0]
        key_bytes = src_bytes[off + 8 : off + 8 + klen]
        key = key_bytes.decode("utf-8")
        vtype = struct.unpack_from("<I", src_bytes, off + 8 + klen)[0]
        val_start = off + 8 + klen + 4
        val_end = skip_val(vtype, val_start)
        parsed.append((key, vtype, src_bytes[val_start:val_end]))
        seen_keys.add(key)
        off = val_end

    for key, vtype, raw_val in parsed:
        entry = bytearray()
        entry.extend(encode_gguf_str(key))
        if key in overrides_str and vtype == 8:
            entry.extend(struct.pack("<I", 8))
            entry.extend(encode_gguf_str(overrides_str[key]))
        elif key in overrides_u32 and vtype == 4:
            entry.extend(struct.pack("<I", 4))
            entry.extend(struct.pack("<I", overrides_u32[key]))
        elif key in overrides_str_arr and vtype == 9:
            entry.extend(struct.pack("<I", 9))
            entry.extend(encode_gguf_str_array(overrides_str_arr[key]))
        else:
            entry.extend(struct.pack("<I", vtype))
            entry.extend(raw_val)
        kv_chunks.append(bytes(entry))

        if key in insert_after_str:
            for new_k, new_v in insert_after_str[key]:
                if new_k not in seen_keys:
                    ins = bytearray()
                    ins.extend(encode_gguf_str(new_k))
                    ins.extend(struct.pack("<I", 8))
                    ins.extend(encode_gguf_str(new_v))
                    kv_chunks.append(bytes(ins))
                    seen_keys.add(new_k)
                else:
                    # Will be overridden in-place when visited
                    pass

    out = bytearray(struct.pack("<4sIQQ", b"GGUF", ver, 0, len(kv_chunks)))
    for ch in kv_chunks:
        out.extend(ch)
    pad = ((len(out) + 31) // 32 * 32) - len(out)
    out.extend(b"\x00" * pad)
    with open(dst_path, "wb") as f:
        f.write(out)
    print(f"[Shard 1] Wrote clean metadata shard {dst_path} ({len(out)} B, {len(kv_chunks)} KV pairs)", flush=True)


def parse_gguf_header_buf(buf: bytes):
    off = 24
    magic, _ver, n_tensors, n_kv = struct.unpack_from("<4sIQQ", buf, 0)
    assert magic == b"GGUF"

    def rs():
        nonlocal off
        n = struct.unpack_from("<Q", buf, off)[0]
        off += 8
        s = buf[off : off + n].decode("utf-8", errors="replace")
        off += n
        return s

    def sv(vt):
        nonlocal off
        sz = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
        if vt in sz:
            off += sz[vt]
        elif vt == 8:
            rs()
        elif vt == 9:
            at, an = struct.unpack_from("<IQ", buf, off)
            off += 12
            if at in sz:
                off += sz[at] * an
            else:
                for _ in range(an):
                    sv(at)

    for _ in range(n_kv):
        rs()
        vt = struct.unpack_from("<I", buf, off)[0]
        off += 4
        sv(vt)

    tensors = []
    for _ in range(n_tensors):
        name = rs()
        nd = struct.unpack_from("<I", buf, off)[0]
        off += 4
        dims = struct.unpack_from(f"<{nd}Q", buf, off)
        off += 8 * nd
        tt, toff = struct.unpack_from("<IQ", buf, off)
        off += 12
        ncol = dims[0]
        nrow = 1
        for d in dims[1:]:
            nrow *= d
        raw_bytes = nrow * row_bytes_for_type(tt, ncol)
        tensors.append({
            "name": name,
            "dims": dims,
            "ttype": tt,
            "off": toff,
            "ncol": ncol,
            "nrow": nrow,
            "nbytes": raw_bytes,
        })
    data_start = (off + 31) // 32 * 32
    tensors.sort(key=lambda x: x["off"])
    last_end = tensors[-1]["off"] + ((tensors[-1]["nbytes"] + 31) // 32 * 32)
    expected_fsize = data_start + last_end
    # Wait: check exact trailing alignment in GGUF (final tensor does not require 32B pad in llama.cpp split files)
    return bytes(buf[:data_start]), data_start, tensors


def safetensor_is_complete(path: str) -> bool:
    if not os.path.isfile(path):
        return False
    try:
        sz = os.path.getsize(path)
        if sz < 16:
            return False
        with open(path, "rb") as f:
            hlen = struct.unpack("<Q", f.read(8))[0]
            if 8 + hlen > sz:
                return False
            hdr = json.loads(f.read(hlen).decode("utf-8"))
        max_end = 0
        for k, v in hdr.items():
            if k == "__metadata__":
                continue
            max_end = max(max_end, v["data_offsets"][1])
        return (8 + hlen + max_end) == sz
    except Exception:
        return False


class SafetensorReader:
    def __init__(self, path: str):
        self.path = path
        self.fd = os.open(path, os.O_RDONLY)
        self.size = os.path.getsize(path)
        self.mm = mmap.mmap(self.fd, self.size, access=mmap.ACCESS_READ)
        hlen = struct.unpack_from("<Q", self.mm, 0)[0]
        self.hdr = json.loads(self.mm[8 : 8 + hlen].decode("utf-8"))
        self.data_base = 8 + hlen
        self.base_addr = np.frombuffer(self.mm, dtype=np.uint8, count=1).ctypes.data

    def get_info(self, name: str):
        meta = self.hdr[name]
        s, e = meta["data_offsets"]
        return meta["dtype"], meta["shape"], self.base_addr + self.data_base + s, e - s

    def close(self):
        self.mm.close()
        os.close(self.fd)


def load_template_headers(template_prefix: str):
    if template_prefix:
        s1 = open(f"{template_prefix}-00001-of-00003.gguf", "rb").read()
        sh_info = {}
        for s_idx in ("00002", "00003"):
            p = f"{template_prefix}-{s_idx}-of-00003.gguf"
            with open(p, "rb") as f:
                peek = f.read(4 * 1024 * 1024)
            hdr_bytes, dstart, tensors = parse_gguf_header_buf(peek)
            fsize = dstart + tensors[-1]["off"] + tensors[-1]["nbytes"]
            sh_info[s_idx] = (hdr_bytes, dstart, fsize, tensors)
        return s1, sh_info

    base_url = f"https://huggingface.co/{DEFAULT_HEADER_REPO}/resolve/main/{DEFAULT_HEADER_PREFIX}"
    print(f"[Template] Fetching GGUF metadata headers via HTTP Range from {DEFAULT_HEADER_REPO}...", flush=True)
    req = urllib.request.Request(f"{base_url}-00001-of-00003.gguf", headers={"User-Agent": "llama-modchip/2.5.3"})
    s1 = urllib.request.urlopen(req, timeout=60).read()
    sh_info = {}
    for s_idx in ("00002", "00003"):
        u = f"{base_url}-{s_idx}-of-00003.gguf"
        peek = http_range_get(u, 0, 2 * 1024 * 1024)
        hdr_bytes, dstart, tensors = parse_gguf_header_buf(peek)
        fsize = dstart + tensors[-1]["off"] + tensors[-1]["nbytes"]
        sh_info[s_idx] = (hdr_bytes, dstart, fsize, tensors)
    return s1, sh_info


def main():
    ap = argparse.ArgumentParser(description="Standalone single-pass 64GB GGUF builder for DeepSeek-V4-Flash-0731")
    ap.add_argument("--mode", choices=["aligned", "abliterated"], default="aligned")
    ap.add_argument("--hf-repo", "--base-model", dest="hf_repo", default=None, help="Upstream HF safetensors repo")
    ap.add_argument("--hf-rev", default=None, help="Pinned upstream HF commit SHA")
    ap.add_argument("--imatrix", default=None, help="Path to @antirez routed-moe-ds4-1p5m.dat (auto-downloaded & SHA256-verified if omitted)")
    ap.add_argument("--template-prefix", default=None, help="Optional local GGUF prefix for header layout only (zero tensor bytes read)")
    ap.add_argument("--out-prefix", "--output-prefix", dest="out_prefix", required=True, help="Output GGUF prefix (writes -0000{1..3}-of-00003.gguf)")
    ap.add_argument("--lib-path", default="./workspace/bin/libdsv4_singlepass.so", help="Path to compiled libdsv4_singlepass.so")
    ap.add_argument("--shards-cache-dir", default=None, help="Temporary cache dir for streaming safetensors shards")
    ap.add_argument("--max-buffered-shards", type=int, default=4, help="Max concurrent safetensors shards buffered on disk")
    ap.add_argument("--rewrite-shard1-only", action="store_true", help="Only write clean Shard-1 metadata and exit")
    args = ap.parse_args()

    try:
        os.nice(19)
    except OSError:
        pass

    if args.mode == "aligned":
        hf_repo = args.hf_repo or "deepseek-ai/DeepSeek-V4-Flash-0731"
        hf_rev = args.hf_rev or "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
        overrides_str = {
            "general.name": "DeepSeek-V4-Flash-0731-MaxQ-Fractal-64GB",
            "general.quantized_by": "Synchopate (llama-modchip)",
            "general.repo_url": "https://github.com/synchopate/llama-modchip",
            "general.base_model.0.name": "DeepSeek-V4-Flash-0731",
            "general.base_model.0.version": "0731",
            "general.base_model.0.organization": "DeepSeek",
            "general.base_model.0.repo_url": f"https://huggingface.co/{hf_repo}",
            "general.base_model.0.revision": hf_rev,
            "quantize.imatrix.file": f"{ANTIREZ_REPO}@{ANTIREZ_REV[:8]}/{ANTIREZ_IMATRIX_RFILE}",
            "quantize.imatrix.dataset": "antirez DeepSeek-V4-Flash chat-v2 1.5M tokens (routed MoE)",
        }
    else:
        hf_repo = args.hf_repo or "prem-research/DeepSeek-V4-Flash-0731-abliterated"
        hf_rev = args.hf_rev or "457210de3a9d27d8142407cd0dac7b6ed6ad9325"
        overrides_str = {
            "general.name": "DeepSeek-V4-Flash-0731-Abliterated-Fractal-64GB",
            "general.quantized_by": "Synchopate (llama-modchip)",
            "general.repo_url": "https://github.com/synchopate/llama-modchip",
            "general.base_model.0.name": "DeepSeek-V4-Flash-0731-abliterated",
            "general.base_model.0.version": "0731",
            "general.base_model.0.organization": "Prem Research",
            "general.base_model.0.repo_url": f"https://huggingface.co/{hf_repo}",
            "general.base_model.0.revision": hf_rev,
            "quantize.imatrix.file": f"{ANTIREZ_REPO}@{ANTIREZ_REV[:8]}/{ANTIREZ_IMATRIX_RFILE}",
            "quantize.imatrix.dataset": "antirez DeepSeek-V4-Flash chat-v2 1.5M tokens (routed MoE)",
        }

    overrides_u32 = {
        "quantize.imatrix.entries_count": 129,
        "quantize.imatrix.chunks_count": 1,
    }
    overrides_str_arr = {
        "general.tags": ["llama-modchip", "deepseek-v4", "cpu-inference", "fractal-64", "64gb"],
    }
    insert_after = {
        "general.base_model.0.repo_url": [("general.base_model.0.revision", hf_rev)],
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out_prefix)), exist_ok=True)
    s1_bytes, sh_info = load_template_headers(args.template_prefix)
    write_shard1_metadata(
        s1_bytes,
        f"{args.out_prefix}-00001-of-00003.gguf",
        overrides_str,
        overrides_u32,
        overrides_str_arr,
        insert_after,
    )
    if args.rewrite_shard1_only:
        return

    shards_dir = args.shards_cache_dir or os.path.join(os.path.dirname(os.path.abspath(args.out_prefix)), "_sp_shards")
    os.makedirs(shards_dir, exist_ok=True)

    imatrix_path = args.imatrix or os.path.join(shards_dir, "routed-moe-ds4-1p5m.dat")
    if not os.path.isfile(imatrix_path):
        im_url = f"https://huggingface.co/{ANTIREZ_REPO}/resolve/{ANTIREZ_REV}/{ANTIREZ_IMATRIX_RFILE}"
        print(f"[Imatrix] Downloading {im_url} ...", flush=True)
        download_file_verified(im_url, imatrix_path, ANTIREZ_IMATRIX_SHA256)
    else:
        got_sha = sha256_file(imatrix_path)
        if got_sha != ANTIREZ_IMATRIX_SHA256:
            raise RuntimeError(f"Imatrix SHA256 mismatch at {imatrix_path}: {got_sha} != {ANTIREZ_IMATRIX_SHA256}")
        print(f"[Imatrix] Verified SHA256={got_sha[:16]}... ({imatrix_path})", flush=True)

    im = load_imatrix(imatrix_path)
    assert len(im) == 129, f"Expected 129 routed-MoE imatrix entries, got {len(im)}"

    lib = ctypes.CDLL(args.lib_path)
    lib.sp_quantize_2d.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int64, ctypes.c_int64,
    ]
    lib.sp_quantize_2d.restype = None
    lib.sp_quantize_moe_3d.argtypes = [
        ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p,
    ]
    lib.sp_quantize_moe_3d.restype = None

    t_global = time.time()
    hf_base = f"https://huggingface.co/{hf_repo}/resolve/{hf_rev}"
    print(f"[*] Fetching safetensors index from {hf_base}/model.safetensors.index.json ...", flush=True)
    idx_req = urllib.request.Request(f"{hf_base}/model.safetensors.index.json", headers={"User-Agent": "llama-modchip/2.5.3"})
    weight_map = json.loads(urllib.request.urlopen(idx_req, timeout=60).read().decode("utf-8"))["weight_map"]

    shard_plans = []
    sf_refcnt = {}
    sf_order = []

    def add_sf_ref(sf_name: str):
        if sf_name not in sf_refcnt:
            sf_refcnt[sf_name] = 0
            sf_order.append(sf_name)
        sf_refcnt[sf_name] += 1

    for s_idx in ("00002", "00003"):
        hdr_bytes, dstart, expected_fsize, tensors = sh_info[s_idx]
        for t in tensors:
            st_key, moe_info = map_gguf_to_st_key(t["name"])
            t["st_key"] = st_key
            t["moe_info"] = moe_info
            req_sfs = set()
            if moe_info is not None:
                bid, proj = moe_info
                n_exp = t["dims"][2]
                for e in range(n_exp):
                    req_sfs.add(weight_map[f"layers.{bid}.ffn.experts.{e}.{proj}.weight"])
                    req_sfs.add(weight_map[f"layers.{bid}.ffn.experts.{e}.{proj}.scale"])
            else:
                req_sfs.add(weight_map[st_key])
                s_key = st_key.removesuffix(".weight") + ".scale" if st_key.endswith(".weight") else ""
                if s_key in weight_map:
                    req_sfs.add(weight_map[s_key])
            t["req_sfs"] = sorted(req_sfs)
            for sfn in t["req_sfs"]:
                add_sf_ref(sfn)
        shard_plans.append((s_idx, hdr_bytes, dstart, expected_fsize, tensors))

    print(f"[Plan] Single-pass quantizing ALL 1,328 tensors (1,199 non-MoE + 129 routed-MoE) across {len(sf_order)} safetensor shards ({hf_repo} @ {hf_rev[:12]})...", flush=True)

    cond = threading.Condition()
    ready_set = set()
    active_on_disk = set()
    prefetch_err = []

    for fn in os.listdir(shards_dir):
        fp = os.path.join(shards_dir, fn)
        if fn in sf_refcnt and safetensor_is_complete(fp):
            ready_set.add(fn)
            active_on_disk.add(fn)
        elif fn.endswith((".safetensors", ".part", ".aria2")):
            try:
                os.unlink(fp)
            except OSError:
                pass

    def prefetch_worker():
        for sf_file in sf_order:
            with cond:
                if sf_file in ready_set:
                    continue
                while len(active_on_disk) >= args.max_buffered_shards:
                    cond.wait(timeout=2.0)
            final_path = os.path.join(shards_dir, sf_file)
            url = f"{hf_base}/{sf_file}"
            ok = False
            for attempt in range(1, 7):
                try:
                    download_file_verified(url, final_path)
                    if safetensor_is_complete(final_path):
                        ok = True
                        break
                except Exception as exc:
                    print(f"[Prefetch] Retry {attempt}/6 for {sf_file}: {exc}", flush=True)
                    time.sleep(2.0 * attempt)
            if not ok:
                with cond:
                    prefetch_err.append(f"Failed downloading {sf_file}")
                    cond.notify_all()
                return
            with cond:
                ready_set.add(sf_file)
                active_on_disk.add(sf_file)
                cond.notify_all()
            print(f"[Prefetch] Ready: {sf_file} (buffered={len(active_on_disk)})", flush=True)

    pf_thread = threading.Thread(target=prefetch_worker, daemon=True)
    pf_thread.start()

    open_readers = {}
    done_tensors = 0

    for s_idx, hdr_bytes, dstart, expected_fsize, tensors in shard_plans:
        out_path = f"{args.out_prefix}-{s_idx}-of-00003.gguf"
        t_shard = time.time()
        dst_fd = os.open(out_path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o644)
        os.write(dst_fd, hdr_bytes)
        written = len(hdr_bytes)
        print(f"\n[Shard {s_idx}] Writing {out_path} ({len(tensors)} tensors, expected {expected_fsize / (1024**3):.2f} GiB)...", flush=True)

        for t in tensors:
            for sfn in t["req_sfs"]:
                if sfn not in open_readers:
                    with cond:
                        while sfn not in ready_set and not prefetch_err:
                            cond.wait(timeout=1.0)
                        if prefetch_err:
                            raise RuntimeError(prefetch_err[0])
                    open_readers[sfn] = SafetensorReader(os.path.join(shards_dir, sfn))

            target_pos = dstart + t["off"]
            if written < target_pos:
                os.write(dst_fd, b"\x00" * (target_pos - written))
                written = target_pos
            assert written == target_pos, f"Offset mismatch on {t['name']}: {written} != {target_pos}"

            ttype = t["ttype"]
            dims = t["dims"]
            nbytes = t["nbytes"]
            out_buf = np.empty(nbytes, dtype=np.uint8)
            t0 = time.time()

            if t["moe_info"] is not None:
                bid, proj = t["moe_info"]
                ncol, nrow, n_exp = dims[0], dims[1], dims[2]
                w_arr = (ctypes.c_void_p * n_exp)()
                s_arr = (ctypes.c_void_p * n_exp)()
                for e in range(n_exp):
                    wk = f"layers.{bid}.ffn.experts.{e}.{proj}.weight"
                    sk = f"layers.{bid}.ffn.experts.{e}.{proj}.scale"
                    w_arr[e] = open_readers[weight_map[wk]].get_info(wk)[2]
                    s_arr[e] = open_readers[weight_map[sk]].get_info(sk)[2]
                qw = im[t["name"]]
                lib.sp_quantize_moe_3d(w_arr, s_arr, out_buf.ctypes.data, ttype, nrow, ncol, n_exp, qw.ctypes.data)
            else:
                st_key = t["st_key"]
                st_w = open_readers[weight_map[st_key]]
                dtype, _shape, w_ptr, _ = st_w.get_info(st_key)
                s_key = st_key.removesuffix(".weight") + ".scale" if st_key.endswith(".weight") else ""
                s_ptr = open_readers[weight_map[s_key]].get_info(s_key)[2] if s_key and s_key in weight_map else 0
                lib.sp_quantize_2d(w_ptr, DTYPE_CODE[dtype], s_ptr, out_buf.ctypes.data, ttype, t["nrow"], t["ncol"])

            mv = memoryview(out_buf)
            pos_b = 0
            while pos_b < nbytes:
                pos_b += os.write(dst_fd, mv[pos_b : pos_b + 64 * 1024 * 1024])
            written += nbytes

            try:
                os.posix_fadvise(dst_fd, (target_pos // 4096) * 4096, (nbytes // 4096) * 4096, os.POSIX_FADV_DONTNEED)
            except Exception:
                pass

            done_tensors += 1
            if t["moe_info"] is not None or t["name"] in ("token_embd.weight", "output.weight"):
                el = time.time() - t_global
                print(f"  [{done_tensors:4d}/1328 | {el/60:5.1f}m] {t['name']:32s} (t={ttype:2d}, {nbytes/(1024**2):6.1f} MiB) in {time.time()-t0:.2f}s", flush=True)

            for sfn in t["req_sfs"]:
                sf_refcnt[sfn] -= 1
                if sf_refcnt[sfn] == 0:
                    open_readers[sfn].close()
                    del open_readers[sfn]
                    sf_path = os.path.join(shards_dir, sfn)
                    try:
                        os.unlink(sf_path)
                    except OSError:
                        pass
                    with cond:
                        active_on_disk.discard(sfn)
                        cond.notify_all()
                    print(f"  [Cleanup] Released & deleted {sfn} (elapsed={(time.time()-t_global)/60:.1f}m)", flush=True)

        if written < expected_fsize:
            os.write(dst_fd, b"\x00" * (expected_fsize - written))
            written = expected_fsize
        assert written == expected_fsize, f"Shard {s_idx} size mismatch: {written} != {expected_fsize}"
        os.fsync(dst_fd)
        os.close(dst_fd)
        print(f"[Shard {s_idx}] COMPLETED {out_path} ({written} B) in {(time.time()-t_shard)/60:.1f}m!", flush=True)

    pf_thread.join(timeout=5.0)
    total_bytes = sum(os.path.getsize(f"{args.out_prefix}-{s}-of-00003.gguf") for s in ("00001", "00002", "00003"))
    print(f"\n[SUCCESS] Built {args.out_prefix}-0000{{1..3}}-of-00003.gguf ({total_bytes} B / {total_bytes / (1024**3):.3f} GiB) in {(time.time()-t_global)/60:.1f}m", flush=True)


if __name__ == "__main__":
    main()
