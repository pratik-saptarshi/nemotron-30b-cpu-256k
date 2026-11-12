#!/usr/bin/env python3
"""
Single-Pass Hybrid VQ3-64 + Fractal-64 GGUF Builder for Abliterated NVIDIA-Nemotron-3.5-30B-A3B (13.30 GiB).

Source weights:
  - HangGlidersRule/Darkstar-Nemotron-3.5-Lightning-30B-A3B-Abliterated-BF16
    (@ cc962dcdbc1fff0519ab3dfcabd6f25ecb52423a — single-direction orthogonalization on
     nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 @ d468880b, NVIDIA Open Model License / OpenMDW-1.1).
  - Structural & unedited-tensor donor:
    Synchopate/NVIDIA-Nemotron-3.5-30B-A3B-VQ3-64-CPU-GGUF (13.30 GiB).

Quantizes all 2,997 residual-writing BF16 tensors modified by Darkstar (76 GGUF tensors:
token_embd, 23x ssm_out, 6x attn_output, 23x ffn_down_shexp, and 23x 128-expert ffn_down_exps)
in a single pass from BF16 into their target GGUF types (Q4_0, Q8_0, and Fractal-64 Q2_0 = 42),
while copying all unedited tensors (including all 19 LUT-VQ3-64 VQ3_0 = 43 ffn_up_exps, SSM states,
routers, norms, and output head) byte-for-byte from the aligned VQ3-64 GGUF.
"""
import argparse
import concurrent.futures
import ctypes
import json
import mmap
import os
import re
import struct
import threading
import time
import urllib.request
import numpy as np


def parse_gguf_full(path: str):
    fsize = os.path.getsize(path)
    with open(path, "rb") as f:
        # Header of 30B GGUF with 131K vocab is ~6-8 MiB
        peek = bytearray(f.read(min(fsize, 16 * 1024 * 1024)))
    magic, ver, n_tensors, n_kv = struct.unpack_from("<4sIQQ", peek, 0)
    assert magic == b"GGUF"
    off = 24

    def skip_val(vt, pos):
        sz = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
        if vt in sz:
            return pos + sz[vt]
        if vt == 8:
            slen = struct.unpack_from("<Q", peek, pos)[0]
            return pos + 8 + slen
        if vt == 9:
            at, an = struct.unpack_from("<IQ", peek, pos)
            pos += 12
            if at in sz:
                return pos + sz[at] * an
            for _ in range(an):
                pos = skip_val(at, pos)
            return pos
        raise ValueError(f"Unknown GGUF KV vtype {vt}")

    kv_items = []
    for _ in range(n_kv):
        klen = struct.unpack_from("<Q", peek, off)[0]
        kbytes = bytes(peek[off + 8 : off + 8 + klen])
        key = kbytes.decode("utf-8", errors="replace")
        vt = struct.unpack_from("<I", peek, off + 8 + klen)[0]
        vstart = off + 8 + klen + 4
        vend = skip_val(vt, vstart)
        kv_items.append((key, kbytes, vt, bytes(peek[vstart:vend])))
        off = vend

    tensor_dir_start = off
    tensors = []
    for _ in range(n_tensors):
        nlen = struct.unpack_from("<Q", peek, off)[0]
        off += 8
        name = peek[off : off + nlen].decode("utf-8", errors="replace")
        off += nlen
        nd = struct.unpack_from("<I", peek, off)[0]
        off += 4
        dims = struct.unpack_from(f"<{nd}Q", peek, off)
        off += 8 * nd
        tt, toff = struct.unpack_from("<IQ", peek, off)
        off += 12
        tensors.append({
            "name": name,
            "dims": dims,
            "ttype": tt,
            "off": toff,
        })
    tensor_dir_bytes = bytes(peek[tensor_dir_start:off])
    data_start = (off + 31) // 32 * 32
    tensors.sort(key=lambda x: x["off"])
    for i in range(len(tensors)):
        nxt = tensors[i + 1]["off"] if i + 1 < len(tensors) else (fsize - data_start)
        tensors[i]["span"] = nxt - tensors[i]["off"]
    return ver, n_tensors, kv_items, tensor_dir_bytes, data_start, fsize, tensors


def build_updated_header(ver: int, n_tensors: int, kv_items: list, tensor_dir_bytes: bytes, overrides_str: dict) -> bytearray:
    existing_keys = {k for k, _, _, _ in kv_items}
    extra_kvs = [(k, v) for k, v in overrides_str.items() if k not in existing_keys]
    out = bytearray(struct.pack("<4sIQQ", b"GGUF", ver, n_tensors, len(kv_items) + len(extra_kvs)))
    for key, kbytes, vt, raw_val in kv_items:
        out.extend(struct.pack("<Q", len(kbytes)))
        out.extend(kbytes)
        out.extend(struct.pack("<I", vt))
        if key in overrides_str and vt == 8:
            sb = overrides_str[key].encode("utf-8")
            out.extend(struct.pack("<Q", len(sb)))
            out.extend(sb)
        else:
            out.extend(raw_val)
    for key, val in extra_kvs:
        kb = key.encode("utf-8")
        vb = val.encode("utf-8")
        out.extend(struct.pack("<Q", len(kb)))
        out.extend(kb)
        out.extend(struct.pack("<I", 8))
        out.extend(struct.pack("<Q", len(vb)))
        out.extend(vb)
    out.extend(tensor_dir_bytes)
    pad = ((len(out) + 31) // 32 * 32) - len(out)
    out.extend(b"\x00" * pad)
    return out


def row_bytes_for_type(ttype: int, ncol: int) -> int:
    table = {
        0:  ncol * 4,
        2:  (ncol // 32) * 18,
        8:  (ncol // 32) * 34,
        42: (ncol // 64) * 30,
        43: (ncol // 64) * 24,
    }
    return table[ttype]


def map_gguf_to_darkstar_keys(name: str):
    if name == "token_embd.weight":
        return ["backbone.embeddings.weight"]
    m = re.match(r"blk\.(\d+)\.(.+)$", name)
    if not m:
        return None
    lid = int(m.group(1))
    suf = m.group(2)
    if suf == "ssm_out.weight":
        return [f"backbone.layers.{lid}.mixer.out_proj.weight"]
    if suf == "attn_output.weight":
        return [f"backbone.layers.{lid}.mixer.o_proj.weight"]
    if suf == "ffn_down_shexp.weight":
        return [f"backbone.layers.{lid}.mixer.shared_experts.down_proj.weight"]
    if suf == "ffn_down_exps.weight":
        return [f"backbone.layers.{lid}.mixer.experts.{e}.down_proj.weight" for e in range(128)]
    return None


def http_download_file(url: str, dst_path: str) -> None:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "llama-modchip/2.5.1"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        # Follow redirects via GET range 0-0 if Content-Length reflects redirect
        pass
    r0 = urllib.request.Request(url, headers={"Range": "bytes=0-0", "User-Agent": "llama-modchip/2.5.1"})
    with urllib.request.urlopen(r0, timeout=30) as resp:
        cr = resp.headers.get("Content-Range", "")
        total = int(cr.split("/")[-1])

    tmp_path = dst_path + ".part"
    chunk_sz = 64 * 1024 * 1024
    ranges = []
    pos = 0
    while pos < total:
        nxt = min(pos + chunk_sz, total)
        ranges.append((pos, nxt))
        pos = nxt

    fd = os.open(tmp_path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o644)
    os.ftruncate(fd, total)

    def fetch_write(rng):
        s, e = rng
        expected = e - s
        for attempt in range(1, 7):
            try:
                rq = urllib.request.Request(url, headers={"Range": f"bytes={s}-{e-1}", "User-Agent": "llama-modchip/2.5.1"})
                with urllib.request.urlopen(rq, timeout=90) as rp:
                    buf = rp.read()
                if len(buf) != expected:
                    raise RuntimeError(f"Short read {len(buf)} != {expected}")
                os.pwrite(fd, buf, s)
                return
            except Exception:
                if attempt == 6:
                    raise
                time.sleep(1.5 * attempt)

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
            list(ex.map(fetch_write, ranges))
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp_path, dst_path)


class SafetensorReader:
    def __init__(self, path: str):
        self.fd = os.open(path, os.O_RDONLY)
        self.size = os.path.getsize(path)
        self.mm = mmap.mmap(self.fd, self.size, access=mmap.ACCESS_READ)
        hlen = struct.unpack_from("<Q", self.mm, 0)[0]
        self.hdr = json.loads(self.mm[8 : 8 + hlen].decode("utf-8"))
        self.data_base = 8 + hlen
        self.base_addr = np.frombuffer(self.mm, dtype=np.uint8, count=1).ctypes.data

    def get_ptr(self, key: str):
        meta = self.hdr[key]
        assert meta["dtype"] == "BF16", f"Expected BF16 for {key}, got {meta['dtype']}"
        s, e = meta["data_offsets"]
        return meta["shape"], self.base_addr + self.data_base + s, e - s

    def close(self):
        self.mm.close()
        os.close(self.fd)


def main():
    ap = argparse.ArgumentParser(description="Build Abliterated Nemotron-3.5-30B-A3B VQ3-64 GGUF (13.30 GiB)")
    ap.add_argument("--hf-repo", default="HangGlidersRule/Darkstar-Nemotron-3.5-Lightning-30B-A3B-Abliterated-BF16")
    ap.add_argument("--hf-rev", default="cc962dcdbc1fff0519ab3dfcabd6f25ecb52423a")
    ap.add_argument("--src-gguf", required=True, help="Aligned NVIDIA-Nemotron-3.5-30B-A3B-VQ3-64.gguf")
    ap.add_argument("--out-gguf", required=True, help="Output NVIDIA-Nemotron-3.5-30B-A3B-Abliterated-VQ3-64.gguf")
    ap.add_argument("--shards-dir", required=True, help="Temporary rolling shard cache directory")
    ap.add_argument("--lib-path", required=True, help="Path to compiled libdsv4_singlepass.so")
    args = ap.parse_args()

    try:
        os.nice(19)
    except OSError:
        pass

    os.makedirs(args.shards_dir, exist_ok=True)
    hf_base = f"https://huggingface.co/{args.hf_repo}/resolve/{args.hf_rev}"

    lib = ctypes.CDLL(args.lib_path)
    lib.sp_quantize_2d.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int64, ctypes.c_int64,
    ]
    lib.sp_quantize_2d.restype = None

    t0 = time.time()
    print(f"[*] Fetching Darkstar index & report from {hf_base} ...", flush=True)
    weight_map = json.loads(urllib.request.urlopen(f"{hf_base}/model.safetensors.index.json", timeout=30).read().decode("utf-8"))["weight_map"]
    report = json.loads(urllib.request.urlopen(f"{hf_base}/abliteration_report.json", timeout=30).read().decode("utf-8"))
    edited_st_keys = {e["tensor"] for e in report["edits"] if not e["tensor"].startswith("mtp.")}

    ver, n_tensors, kv_items, tensor_dir_bytes, src_dstart, src_fsize, tensors = parse_gguf_full(args.src_gguf)
    overrides = {
        "general.name": "NVIDIA-Nemotron-3.5-30B-A3B-Abliterated-VQ3-64",
        "general.quantized_by": "Synchopate (llama-modchip)",
        "general.repo_url": "https://github.com/synchopate/llama-modchip",
        "general.base_model.0.name": "Darkstar-Nemotron-3.5-Lightning-30B-A3B-Abliterated-BF16",
        "general.base_model.0.organization": "HangGlidersRule",
        "general.base_model.0.repo_url": f"https://huggingface.co/{args.hf_repo}",
    }
    new_hdr = build_updated_header(ver, n_tensors, kv_items, tensor_dir_bytes, overrides)
    dst_dstart = len(new_hdr)
    exp_fsize = dst_dstart + (src_fsize - src_dstart)

    sf_order = []
    sf_seen = set()
    sf_refcnt = {}
    matched_st = set()
    edited_gguf_cnt = 0

    for t in tensors:
        st_keys = map_gguf_to_darkstar_keys(t["name"])
        if st_keys is not None and any(k in edited_st_keys for k in st_keys):
            assert all(k in edited_st_keys for k in st_keys), f"Partial edit on {t['name']}"
            matched_st.update(st_keys)
            sfs = []
            for k in st_keys:
                sf = weight_map[k]
                if sf not in sf_seen:
                    sf_seen.add(sf)
                    sf_order.append(sf)
                if sf not in sfs:
                    sfs.append(sf)
            for sf in sfs:
                sf_refcnt[sf] = sf_refcnt.get(sf, 0) + 1
            t["st_keys"] = st_keys
            t["sfs"] = sfs
            edited_gguf_cnt += 1
        else:
            t["st_keys"] = None
            t["sfs"] = []

    assert matched_st == edited_st_keys, f"Matched {len(matched_st)} != report {len(edited_st_keys)}"
    print(f"[Plan] Verified exact match: {len(matched_st)} Darkstar BF16 matrices -> {edited_gguf_cnt} GGUF tensors across {len(sf_order)} shards", flush=True)

    cond = threading.Condition()
    ready_set = set()
    active_on_disk = set()
    prefetch_err = []

    def prefetch_worker():
        for sf in sf_order:
            with cond:
                while len(active_on_disk) >= 3 and not prefetch_err:
                    cond.wait(timeout=1.0)
                if prefetch_err:
                    return
                active_on_disk.add(sf)
            dst_p = os.path.join(args.shards_dir, sf)
            try:
                if not os.path.isfile(dst_p):
                    t_dl = time.time()
                    http_download_file(f"{hf_base}/{sf}", dst_p)
                    print(f"  [Prefetch] Downloaded {sf} ({os.path.getsize(dst_p)/(1024**3):.2f} GiB) in {time.time()-t_dl:.1f}s", flush=True)
                with cond:
                    ready_set.add(sf)
                    cond.notify_all()
            except Exception as exc:
                with cond:
                    prefetch_err.append(str(exc))
                    cond.notify_all()
                return

    pf_thread = threading.Thread(target=prefetch_worker, daemon=True)
    pf_thread.start()

    src_fd = os.open(args.src_gguf, os.O_RDONLY)
    src_mm = mmap.mmap(src_fd, src_fsize, access=mmap.ACCESS_READ)
    dst_fd = os.open(args.out_gguf, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o644)
    os.write(dst_fd, new_hdr)
    written = len(new_hdr)

    readers = {}
    done_ed = 0

    for t in tensors:
        target_pos = dst_dstart + t["off"]
        if written < target_pos:
            os.write(dst_fd, b"\x00" * (target_pos - written))
            written = target_pos
        assert written == target_pos

        if t["st_keys"] is None:
            rem = t["span"]
            pos = src_dstart + t["off"]
            while rem > 0:
                chunk = min(rem, 64 * 1024 * 1024)
                os.write(dst_fd, src_mm[pos : pos + chunk])
                pos += chunk
                rem -= chunk
            written += t["span"]
        else:
            for sf in t["sfs"]:
                if sf not in readers:
                    with cond:
                        while sf not in ready_set and not prefetch_err:
                            cond.wait(timeout=1.0)
                        if prefetch_err:
                            raise RuntimeError(prefetch_err[0])
                    readers[sf] = SafetensorReader(os.path.join(args.shards_dir, sf))

            dims = t["dims"]
            ttype = t["ttype"]
            ncol = dims[0]
            nrow = dims[1]
            rb = row_bytes_for_type(ttype, ncol)

            if len(t["st_keys"]) == 1:
                sk = t["st_keys"][0]
                st = readers[weight_map[sk]]
                shape, w_ptr, _ = st.get_ptr(sk)
                assert shape == [nrow, ncol], f"Shape mismatch {sk}: {shape} vs [{nrow}, {ncol}]"
                nsz = nrow * rb
                out_buf = np.empty(nsz, dtype=np.uint8)
                lib.sp_quantize_2d(w_ptr, 1, 0, out_buf.ctypes.data, ttype, nrow, ncol)
            else:
                n_exp = dims[2]
                assert len(t["st_keys"]) == n_exp
                eb = nrow * rb
                nsz = n_exp * eb
                out_buf = np.empty(nsz, dtype=np.uint8)
                base_dst = out_buf.ctypes.data
                for e_idx, sk in enumerate(t["st_keys"]):
                    st = readers[weight_map[sk]]
                    shape, w_ptr, _ = st.get_ptr(sk)
                    assert shape == [nrow, ncol], f"Expert shape mismatch {sk}: {shape} vs [{nrow}, {ncol}]"
                    lib.sp_quantize_2d(w_ptr, 1, 0, base_dst + e_idx * eb, ttype, nrow, ncol)

            assert nsz <= t["span"], f"Quantized size {nsz} > span {t['span']} on {t['name']}"
            mv = memoryview(out_buf)
            pos_b = 0
            while pos_b < nsz:
                pos_b += os.write(dst_fd, mv[pos_b : pos_b + 64 * 1024 * 1024])
            written += nsz
            done_ed += 1
            if done_ed % 10 == 0 or done_ed == edited_gguf_cnt or t["name"] == "token_embd.weight":
                print(f"  [{done_ed:2d}/{edited_gguf_cnt}] Encoded {t['name']:30s} (BF16 -> type={ttype:2d}, {nsz/(1024**2):6.2f} MiB) [{time.time()-t0:.1f}s]", flush=True)

            for sf in t["sfs"]:
                sf_refcnt[sf] -= 1
                if sf_refcnt[sf] == 0:
                    readers.pop(sf).close()
                    try:
                        os.unlink(os.path.join(args.shards_dir, sf))
                    except OSError:
                        pass
                    with cond:
                        active_on_disk.discard(sf)
                        cond.notify_all()

        try:
            os.posix_fadvise(dst_fd, (target_pos // 4096) * 4096, (t["span"] // 4096) * 4096, os.POSIX_FADV_DONTNEED)
        except Exception:
            pass

    if written < exp_fsize:
        os.write(dst_fd, b"\x00" * (exp_fsize - written))
        written = exp_fsize

    assert written == exp_fsize, f"Size mismatch {written} != {exp_fsize}"
    os.fsync(dst_fd)
    os.close(dst_fd)
    src_mm.close()
    os.close(src_fd)
    pf_thread.join(timeout=5.0)
    print(f"\n[SUCCESS] Built {args.out_gguf} ({written} B / {written / (1024**3):.3f} GiB) in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
