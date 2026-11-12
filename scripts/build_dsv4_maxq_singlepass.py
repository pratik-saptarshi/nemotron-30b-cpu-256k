#!/usr/bin/env python3
"""
Single-Pass Max-Quality 64GB GGUF Builder for DeepSeek-V4-Flash-0731 (56.93 GiB).

Reproduces both:
  1. Aligned Default:   Synchopate/DeepSeek-V4-Flash-0731-MaxQ-Fractal-64GB-CPU-GGUF
                        (base: deepseek-ai/DeepSeek-V4-Flash-0731 @ 7872f01b1d1fe23eabc4c98b48bffcef5a386062)
  2. Uncensored Opt-In: Synchopate/DeepSeek-V4-Flash-0731-Abliterated-Fractal-64GB-CPU-GGUF
                        (base: prem-research/DeepSeek-V4-Flash-0731-abliterated @ 457210de3a9d27d8142407cd0dac7b6ed6ad9325)

Architecture & Quantization Recipe (61,129,363,968 / 61,129,364,000 bytes / 56.93 GiB across 3 shards):
  - All 1,199 non-routed-MoE tensors quantized in a SINGLE PASS directly from native FP8_E4M3 / BF16 safetensors:
      * 173 x Q6_K + 1 x Q8_0 (174 dense 100%-active bottlenecks: token_embd, output, 43x attn_q_a, 43x3 shared_experts;
        upgrades 130 tensors that were Q4_K/Q5_K/VQ3_0 in the 56.40 GiB VQ3 / 56.57 GiB Fractal builds)
      * 317 x Q2_0 (Fractal-64, 3.75 bpw) across MLA & Lightning Indexer projections
      * 662 x F32 + 43 x BF16 + 3 x I32 across RMSNorms, MoE routers, attention sinks, Hyper-Connections, and tid2eid
  - All 129 routed-MoE 3D tensors (MXFP4 -> IQ2_XXS / IQ1_S / IQ1_M importance-calibrated on DeepSeek-V4-Flash-0731)
    copied byte-for-byte from the matching 0731 routed-MoE donor GGUF.
"""
import argparse
import concurrent.futures
import ctypes
import json
import mmap
import os
import re
import struct
import time
import urllib.request
import numpy as np

DTYPE_CODE = {"F32": 0, "BF16": 1, "F8_E4M3": 2, "I64": 3, "I32": 4}

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


def map_non_moe_to_st_key(name: str) -> str:
    if name in ROOT_MAP:
        return ROOT_MAP[name]
    m = re.match(r"blk\.(\d+)\.(.+)$", name)
    if not m:
        raise KeyError(f"Unmapped tensor name: {name}")
    return f"layers.{int(m.group(1))}.{LAYER_MAP[m.group(2)]}"


def write_shard1_metadata(src_path: str, dst_path: str, overrides_str: dict) -> None:
    buf = open(src_path, "rb").read()
    magic, _ver, n_t, n_kv = struct.unpack_from("<4sIQQ", buf, 0)
    assert magic == b"GGUF" and n_t == 0
    off = 24
    out = bytearray(buf[:24])

    def skip_val(vtype, pos):
        sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
        if vtype in sizes:
            return pos + sizes[vtype]
        if vtype == 8:
            slen = struct.unpack_from("<Q", buf, pos)[0]
            return pos + 8 + slen
        if vtype == 9:
            atype, alen = struct.unpack_from("<IQ", buf, pos)
            pos += 12
            if atype in sizes:
                return pos + sizes[atype] * alen
            for _ in range(alen):
                pos = skip_val(atype, pos)
            return pos
        raise ValueError(f"Unsupported GGUF KV type {vtype}")

    for _ in range(n_kv):
        klen = struct.unpack_from("<Q", buf, off)[0]
        key_bytes = buf[off + 8 : off + 8 + klen]
        key = key_bytes.decode("utf-8")
        vtype = struct.unpack_from("<I", buf, off + 8 + klen)[0]
        val_start = off + 8 + klen + 4
        val_end = skip_val(vtype, val_start)

        out.extend(struct.pack("<Q", klen))
        out.extend(key_bytes)
        out.extend(struct.pack("<I", vtype))
        if key in overrides_str and vtype == 8:
            sb = overrides_str[key].encode("utf-8")
            out.extend(struct.pack("<Q", len(sb)))
            out.extend(sb)
        else:
            out.extend(buf[val_start:val_end])
        off = val_end

    pad = ((len(out) + 31) // 32 * 32) - len(out)
    out.extend(b"\x00" * pad)
    with open(dst_path, "wb") as f:
        f.write(out)
    print(f"[Shard 1] Wrote metadata shard {dst_path} ({len(out)} bytes)", flush=True)


def parse_header(path: str):
    file_size = os.path.getsize(path)
    with open(path, "rb") as f:
        buf = bytearray(f.read(4 * 1024 * 1024))
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
        tensors.append({
            "name": name,
            "dims": dims,
            "ttype": tt,
            "off": toff,
        })
    data_start = (off + 31) // 32 * 32
    tensors.sort(key=lambda x: x["off"])
    for i in range(len(tensors)):
        nxt = tensors[i + 1]["off"] if i + 1 < len(tensors) else (file_size - data_start)
        # Compute exact payload size without trailing 32B pad on intermediate tensors
        tensors[i]["span"] = nxt - tensors[i]["off"]
    return bytearray(buf[:data_start]), data_start, file_size, tensors


def http_range_get(url: str, start_byte: int, end_byte_excl: int) -> bytes:
    expected = end_byte_excl - start_byte
    assert expected > 0
    if expected > 64 * 1024 * 1024:
        chunk_sz = 64 * 1024 * 1024
        ranges = []
        pos = start_byte
        while pos < end_byte_excl:
            nxt = min(pos + chunk_sz, end_byte_excl)
            ranges.append((pos, nxt))
            pos = nxt
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
            parts = list(ex.map(lambda r: http_range_get(url, r[0], r[1]), ranges))
        out = b"".join(parts)
        assert len(out) == expected
        return out

    hdrs = {
        "Range": f"bytes={start_byte}-{end_byte_excl - 1}",
        "User-Agent": "llama-modchip/2.5.1",
    }
    for attempt in range(1, 8):
        try:
            req = urllib.request.Request(url, headers=hdrs)
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
                if len(data) == expected:
                    return data
                raise RuntimeError(f"Short Range read {len(data)} != {expected}")
        except Exception:
            if attempt == 7:
                raise
            time.sleep(1.2 * attempt)
    raise RuntimeError("unreachable")


def row_bytes_for_type(ttype: int, ncol: int) -> int:
    table = {
        0:  ncol * 4,
        2:  (ncol // 32) * 18,
        8:  (ncol // 32) * 34,
        12: (ncol // 256) * 144,
        13: (ncol // 256) * 176,
        14: (ncol // 256) * 210,
        26: ncol * 4,
        30: ncol * 2,
        42: (ncol // 64) * 30,
        43: (ncol // 64) * 24,
    }
    return table[ttype]


def main():
    ap = argparse.ArgumentParser(description="Build single-pass MaxQ-Fractal 64GB GGUF for DeepSeek-V4-Flash-0731")
    ap.add_argument("--mode", choices=["aligned", "abliterated"], default="aligned")
    ap.add_argument("--hf-repo", default=None, help="Override upstream HF safetensors repo")
    ap.add_argument("--hf-rev", default=None, help="Override upstream HF commit SHA")
    ap.add_argument("--template-prefix", required=True, help="Path prefix of 3-shard MaxQ-Fractal template GGUF")
    ap.add_argument("--moe-src-prefix", required=True, help="Path prefix of 3-shard single-pass routed-MoE donor GGUF")
    ap.add_argument("--out-prefix", required=True, help="Output path prefix (writes -0000{1..3}-of-00003.gguf)")
    ap.add_argument("--lib-path", required=True, help="Path to compiled libdsv4_singlepass.so")
    ap.add_argument("--workers", type=int, default=8, help="Concurrent HTTP Range prefetch workers")
    args = ap.parse_args()

    try:
        os.nice(19)
    except OSError:
        pass

    if args.mode == "aligned":
        hf_repo = args.hf_repo or "deepseek-ai/DeepSeek-V4-Flash-0731"
        hf_rev = args.hf_rev or "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
        overrides = {
            "general.name": "DeepSeek-V4-Flash-0731-MaxQ-Fractal-64GB",
            "general.quantized_by": "Synchopate (llama-modchip)",
            "general.repo_url": "https://github.com/synchopate/llama-modchip",
            "general.base_model.0.name": "DeepSeek-V4-Flash-0731",
            "general.base_model.0.version": "0731",
            "general.base_model.0.organization": "DeepSeek",
            "general.base_model.0.repo_url": "https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731",
            "quantize.imatrix.file": "DeepSeek-V4-Flash-0731-GGUF/imatrix_unsloth.gguf",
            "quantize.imatrix.dataset": "unsloth_calibration_DeepSeek-V4-Flash-0731.txt",
        }
    else:
        hf_repo = args.hf_repo or "prem-research/DeepSeek-V4-Flash-0731-abliterated"
        hf_rev = args.hf_rev or "457210de3a9d27d8142407cd0dac7b6ed6ad9325"
        overrides = {
            "general.name": "DeepSeek-V4-Flash-0731-Abliterated-Fractal-64GB",
            "general.quantized_by": "Synchopate (llama-modchip)",
            "general.repo_url": "https://github.com/synchopate/llama-modchip",
            "general.base_model.0.name": "DeepSeek-V4-Flash-0731-abliterated",
            "general.base_model.0.organization": "Prem Research",
            "general.base_model.0.repo_url": "https://huggingface.co/prem-research/DeepSeek-V4-Flash-0731-abliterated",
            "quantize.imatrix.file": "antirez/deepseek-v4-gguf/imatrix/DeepSeek-V4-Flash-chat-v2-routed-moe-ds4-1p5m.dat",
            "quantize.imatrix.dataset": "antirez ds4 routed-moe 1.5M tokens chat-v2",
        }

    hf_base = f"https://huggingface.co/{hf_repo}/resolve/{hf_rev}"
    lib = ctypes.CDLL(args.lib_path)
    lib.sp_quantize_2d.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int64, ctypes.c_int64,
    ]
    lib.sp_quantize_2d.restype = None

    t0 = time.time()
    print(f"[*] Fetching safetensors index from {hf_base}/model.safetensors.index.json ...", flush=True)
    idx_bytes = urllib.request.urlopen(f"{hf_base}/model.safetensors.index.json", timeout=30).read()
    weight_map = json.loads(idx_bytes.decode("utf-8"))["weight_map"]

    write_shard1_metadata(
        f"{args.template_prefix}-00001-of-00003.gguf",
        f"{args.out_prefix}-00001-of-00003.gguf",
        overrides,
    )

    needed_sf = set()
    shard_jobs = []
    total_sp = 0

    for s_idx in ("00002", "00003"):
        tpl_path = f"{args.template_prefix}-{s_idx}-of-00003.gguf"
        moe_path = f"{args.moe_src_prefix}-{s_idx}-of-00003.gguf"
        hdr_buf, dstart, tpl_fsize, tpl_tensors = parse_header(tpl_path)
        _, moe_dstart, moe_fsize, moe_tensors = parse_header(moe_path)
        moe_map = {t["name"]: t for t in moe_tensors}

        sp_cnt = 0
        for t in tpl_tensors:
            n_el = 1
            for d in t["dims"]:
                n_el *= d
            t["n_el"] = n_el
            if t["name"].endswith("_exps.weight"):
                mt = moe_map[t["name"]]
                assert mt["ttype"] == t["ttype"] and mt["dims"] == t["dims"], f"MoE mismatch on {t['name']}"
                t["is_moe"] = True
                t["moe_off"] = moe_dstart + mt["off"]
                t["raw_bytes"] = mt["span"]
            else:
                t["is_moe"] = False
                ncol = t["dims"][0]
                nrow = n_el // ncol
                t["nrow"] = nrow
                t["ncol"] = ncol
                t["raw_bytes"] = nrow * row_bytes_for_type(t["ttype"], ncol)
                assert t["raw_bytes"] <= t["span"], f"Size overflow on {t['name']}: {t['raw_bytes']} > {t['span']}"
                st_key = map_non_moe_to_st_key(t["name"])
                sf_w = weight_map[st_key]
                needed_sf.add(sf_w)
                s_key = st_key.removesuffix(".weight") + ".scale" if st_key.endswith(".weight") else ""
                sf_s = weight_map.get(s_key)
                if sf_s:
                    needed_sf.add(sf_s)
                t["st_key"] = st_key
                t["sf_w"] = sf_w
                t["s_key"] = s_key
                t["sf_s"] = sf_s
                sp_cnt += 1
                total_sp += 1

        shard_jobs.append((s_idx, moe_path, moe_fsize, hdr_buf, dstart, tpl_fsize, sp_cnt, tpl_tensors))

    def fetch_sf_header(sf_file: str):
        url = f"{hf_base}/{sf_file}"
        b8 = http_range_get(url, 0, 8)
        hlen = struct.unpack("<Q", b8)[0]
        hb = http_range_get(url, 8, 8 + hlen)
        return sf_file, (8 + hlen, json.loads(hb.decode("utf-8")))

    print(f"[Plan] Single-pass quantizing {total_sp} non-MoE tensors across {len(needed_sf)} safetensor shards...", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
        sf_headers = dict(ex.map(fetch_sf_header, sorted(needed_sf)))
    print(f"[Headers] Fetched {len(sf_headers)} safetensor headers in {time.time() - t0:.1f}s", flush=True)

    def download_payload(t):
        sf_w = t["sf_w"]
        dbase_w, hdr_w = sf_headers[sf_w]
        meta_w = hdr_w[t["st_key"]]
        s0, e0 = meta_w["data_offsets"]
        w_bytes = http_range_get(f"{hf_base}/{sf_w}", dbase_w + s0, dbase_w + e0)
        dtype = meta_w["dtype"]
        s_bytes = None
        if dtype == "F8_E4M3":
            sf_s = t["sf_s"]
            assert sf_s is not None, f"Missing scale for {t['st_key']}"
            dbase_s, hdr_s = sf_headers[sf_s]
            s1, e1 = hdr_s[t["s_key"]]["data_offsets"]
            s_bytes = http_range_get(f"{hf_base}/{sf_s}", dbase_s + s1, dbase_s + e1)
        return dtype, w_bytes, s_bytes

    done_sp = 0
    for s_idx, moe_path, moe_fsize, hdr_buf, dstart, exp_fsize, sp_cnt, tensors in shard_jobs:
        out_path = f"{args.out_prefix}-{s_idx}-of-00003.gguf"
        t_sh = time.time()
        print(f"\n[Shard {s_idx}] Building {out_path} ({exp_fsize / (1024**3):.3f} GiB, {sp_cnt} single-pass non-MoE tensors)...", flush=True)

        moe_fd = os.open(moe_path, os.O_RDONLY)
        moe_mm = mmap.mmap(moe_fd, moe_fsize, access=mmap.ACCESS_READ)
        dst_fd = os.open(out_path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o644)
        os.write(dst_fd, hdr_buf)
        written = len(hdr_buf)

        sp_indices = [i for i, t in enumerate(tensors) if not t["is_moe"]]
        futures_map = {}
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=args.workers)
        next_ptr = 0

        def schedule_ahead(cur_pos):
            nonlocal next_ptr
            while next_ptr < len(sp_indices) and next_ptr < cur_pos + args.workers + 4:
                ti = sp_indices[next_ptr]
                futures_map[ti] = pool.submit(download_payload, tensors[ti])
                next_ptr += 1

        schedule_ahead(0)
        cur_seen = 0

        for idx, t in enumerate(tensors):
            target_pos = dstart + t["off"]
            if written < target_pos:
                os.write(dst_fd, b"\x00" * (target_pos - written))
                written = target_pos
            assert written == target_pos

            if t["is_moe"]:
                rem = t["raw_bytes"]
                pos = t["moe_off"]
                while rem > 0:
                    chunk = min(rem, 64 * 1024 * 1024)
                    os.write(dst_fd, moe_mm[pos : pos + chunk])
                    pos += chunk
                    rem -= chunk
                written += t["raw_bytes"]
                try:
                    os.posix_fadvise(moe_fd, (t["moe_off"] // 4096) * 4096, (t["raw_bytes"] // 4096) * 4096, os.POSIX_FADV_DONTNEED)
                except Exception:
                    pass
            else:
                dtype, w_bytes, s_bytes = futures_map.pop(idx).result()
                cur_seen += 1
                schedule_ahead(cur_seen)

                nsz = t["raw_bytes"]
                w_arr = np.frombuffer(w_bytes, dtype=np.uint8)
                s_ptr = np.frombuffer(s_bytes, dtype=np.uint8).ctypes.data if s_bytes is not None else 0
                out_buf = np.empty(nsz, dtype=np.uint8)
                lib.sp_quantize_2d(w_arr.ctypes.data, DTYPE_CODE[dtype], s_ptr, out_buf.ctypes.data, t["ttype"], t["nrow"], t["ncol"])

                mv = memoryview(out_buf)
                pos_b = 0
                while pos_b < nsz:
                    pos_b += os.write(dst_fd, mv[pos_b : pos_b + 64 * 1024 * 1024])
                written += nsz
                done_sp += 1
                if done_sp % 80 == 0 or done_sp == total_sp or t["name"] in ("token_embd.weight", "output.weight"):
                    print(f"  [{done_sp:4d}/{total_sp}] {t['name']:36s} ({dtype:7s} -> type={t['ttype']:2d}, {nsz/(1024**2):6.2f} MiB) [{time.time()-t0:.1f}s]", flush=True)

            try:
                os.posix_fadvise(dst_fd, (target_pos // 4096) * 4096, (t["raw_bytes"] // 4096) * 4096, os.POSIX_FADV_DONTNEED)
            except Exception:
                pass

        if written < exp_fsize:
            os.write(dst_fd, b"\x00" * (exp_fsize - written))
            written = exp_fsize

        pool.shutdown(wait=True)
        assert written == exp_fsize, f"Size mismatch {written} != {exp_fsize}"
        os.fsync(dst_fd)
        os.close(dst_fd)
        moe_mm.close()
        os.close(moe_fd)
        print(f"[Shard {s_idx}] Completed {out_path} ({written} B) in {time.time() - t_sh:.1f}s", flush=True)

    total_bytes = sum(os.path.getsize(f"{args.out_prefix}-{s}-of-00003.gguf") for s in ("00001", "00002", "00003"))
    print(f"\n[SUCCESS] Built {args.out_prefix}-0000{{1..3}}-of-00003.gguf ({total_bytes} B / {total_bytes / (1024**3):.3f} GiB) in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
