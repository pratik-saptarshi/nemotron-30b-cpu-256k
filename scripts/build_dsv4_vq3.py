#!/usr/bin/env python3
"""
Build the DeepSeek-V4-Flash-0731-VQ3 hybrid 3-shard GGUF (56.40 GiB).

Default conservative hybrid layout (zero double-quantization of MLA attention):
  - 100% of MLA attention projections (attn_q_b, attn_output_a, attn_output_b,
    attn_kv, attn_compressor_*, indexer_compressor_*) remain untouched in
    Fractal-64 (GGML_TYPE_Q2_0 = 42, 3.75 bpw).
  - Boundary shared experts (blk.0..4, blk.39..42) and all blk.*.ffn_down_shexp
    remain untouched in Q5_K / Q6_K.
  - 68 interior shared-expert pre-SiLU projections (blk.5..38 ffn_gate_shexp
    and ffn_up_shexp) are encoded from Q5_K/Q6_K into LUT-VQ3-64
    (GGML_TYPE_VQ3_0 = 43, 3.00 bpw, 24 B / 64 weights).

Pass --aggressive to also transcode MLA expansion projections (56.07 GiB ablation).
"""

import argparse
import ctypes
import mmap
import os
import shutil
import struct
import subprocess
import time
from pathlib import Path

import numpy as np

GGML_TYPE_Q5_K = 13
GGML_TYPE_Q6_K = 14
GGML_TYPE_Q2_0 = 42
GGML_TYPE_VQ3_0 = 43

AGGRESSIVE_SUFFIXES = (
    "attn_q_b.weight",
    "attn_output_a.weight",
    "attn_compressor_gate.weight",
    "indexer.attn_q_b.weight",
    "indexer_compressor_gate.weight",
    "ffn_gate_shexp.weight",
    "ffn_up_shexp.weight",
)


def load_vq3_encoder(lib_path: str = "") -> ctypes.CDLL:
    here = Path(__file__).resolve().parent
    repo_root = here if (here / "vq3" / "vq3_enc_lib.c").is_file() else here.parent
    so_path = Path(lib_path) if lib_path else (repo_root / "vq3" / "libvq3_enc.so")
    c_src = repo_root / "vq3" / "vq3_enc_lib.c"
    if not so_path.is_file():
        if not c_src.is_file():
            raise FileNotFoundError(f"Missing encoder library {so_path} and source {c_src}")
        print(f"[*] Compiling {c_src} -> {so_path} ...", flush=True)
        subprocess.check_call([
            "gcc", "-O3", "-march=native", "-mavx512f", "-mf16c",
            "-fopenmp", "-shared", "-fPIC", str(c_src), "-lm", "-o", str(so_path),
        ])
    lib = ctypes.CDLL(str(so_path))
    lib.transcode_any_to_vq3_0_omp.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64,
    ]
    lib.transcode_any_to_vq3_0_omp.restype = None
    return lib


def should_convert_vq3(name: str, orig_type: int, aggressive: bool = False) -> bool:
    if aggressive:
        return orig_type in (GGML_TYPE_Q2_0, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K) and name.endswith(AGGRESSIVE_SUFFIXES)
    # Default shipped hybrid: never double-quantize Fractal-64 (type 42) attention tensors;
    # convert interior (blk.5 .. blk.38) shared-expert gate/up projections from Q5_K/Q6_K to VQ3_0.
    if orig_type in (GGML_TYPE_Q5_K, GGML_TYPE_Q6_K) and name.endswith(("ffn_gate_shexp.weight", "ffn_up_shexp.weight")):
        parts = name.split(".")
        if len(parts) >= 3 and parts[0] == "blk" and parts[1].isdigit():
            blk_id = int(parts[1])
            return 5 <= blk_id <= 38
    return False


def parse_header(buf: bytearray, file_size: int):
    off = 24
    magic, _version, n_tensors, n_kv = struct.unpack_from("<4sIQQ", buf, 0)
    if magic != b"GGUF":
        raise ValueError(f"Not a GGUF shard (magic={magic!r})")

    def read_str():
        nonlocal off
        (length,) = struct.unpack_from("<Q", buf, off)
        off += 8
        s = buf[off : off + length].decode("utf-8", errors="replace")
        off += length
        return s

    def skip_val(vtype):
        nonlocal off
        sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
        if vtype in sizes:
            off += sizes[vtype]
        elif vtype == 8:
            read_str()
        elif vtype == 9:
            atype, alen = struct.unpack_from("<IQ", buf, off)
            off += 12
            for _ in range(alen):
                skip_val(atype)

    for _ in range(n_kv):
        read_str()
        (vtype,) = struct.unpack_from("<I", buf, off)
        off += 4
        skip_val(vtype)

    tensors = []
    for _ in range(n_tensors):
        name = read_str()
        (ndims,) = struct.unpack_from("<I", buf, off)
        off += 4
        dims = struct.unpack_from(f"<{ndims}Q", buf, off)
        off += 8 * ndims
        pos_ttype = off
        ttype, offset = struct.unpack_from("<IQ", buf, off)
        off += 12
        tensors.append({
            "name": name,
            "dims": dims,
            "orig_type": ttype,
            "orig_off": offset,
            "pos_ttype": pos_ttype,
        })
    data_start = (off + 31) // 32 * 32
    tensors.sort(key=lambda x: x["orig_off"])
    for i in range(len(tensors)):
        nxt = tensors[i + 1]["orig_off"] if i + 1 < len(tensors) else (file_size - data_start)
        tensors[i]["orig_size"] = nxt - tensors[i]["orig_off"]
    return data_start, tensors


def transcode_shard(lib: ctypes.CDLL, src_path: str, dst_path: str, aggressive: bool = False) -> None:
    t_start = time.time()
    file_size = os.path.getsize(src_path)
    if file_size < 10 * 1024 * 1024:
        shutil.copyfile(src_path, dst_path)
        print(f"Copied metadata shard {os.path.basename(dst_path)} ({file_size} B)", flush=True)
        return

    src_fd = os.open(src_path, os.O_RDONLY)
    src_mm = mmap.mmap(src_fd, file_size, access=mmap.ACCESS_READ)

    header_peek = bytearray(src_mm[: 4 * 1024 * 1024])
    data_start, tensors = parse_header(header_peek, file_size)
    header_buf = bytearray(src_mm[:data_start])

    cur_off = 0
    vq3_total = 0
    for t in tensors:
        cur_off = (cur_off + 31) // 32 * 32
        n_el = 1
        for d in t["dims"]:
            n_el *= d
        t["n_el"] = n_el
        if should_convert_vq3(t["name"], t["orig_type"], aggressive=aggressive):
            nt = GGML_TYPE_VQ3_0
            nsz = (n_el // 64) * 24
            vq3_total += 1
        else:
            nt = t["orig_type"]
            nsz = t["orig_size"]
        t["new_type"] = nt
        t["new_off"] = cur_off
        t["new_size"] = nsz
        struct.pack_into("<IQ", header_buf, t["pos_ttype"], nt, cur_off)
        cur_off += nsz

    total_new_gib = (data_start + cur_off) / (1024 ** 3)
    print(
        f"[DSV4-VQ3] {os.path.basename(src_path)} ({file_size / (1024 ** 3):.2f} GiB) -> "
        f"{os.path.basename(dst_path)} ({total_new_gib:.2f} GiB), converting {vq3_total} tensors to VQ3_0",
        flush=True,
    )

    dst_fd = os.open(dst_path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o644)
    os.write(dst_fd, header_buf)
    written = len(header_buf)

    conv_done = 0
    for t in tensors:
        target_pos = data_start + t["new_off"]
        if written < target_pos:
            os.write(dst_fd, b"\x00" * (target_pos - written))
            written = target_pos

        ot, nt = t["orig_type"], t["new_type"]
        src_abs = data_start + t["orig_off"]
        osz, nsz = t["orig_size"], t["new_size"]

        if ot == nt:
            rem = osz
            pos = src_abs
            while rem > 0:
                chunk = min(rem, 64 * 1024 * 1024)
                os.write(dst_fd, src_mm[pos : pos + chunk])
                pos += chunk
                rem -= chunk
            written += osz
        else:
            ncol = t["dims"][0]
            nrow = t["n_el"] // ncol
            src_ptr = np.frombuffer(src_mm, dtype=np.uint8, count=osz, offset=src_abs).ctypes.data
            out_buf = np.empty(nsz, dtype=np.uint8)
            dst_ptr = out_buf.ctypes.data
            lib.transcode_any_to_vq3_0_omp(src_ptr, ot, dst_ptr, nrow, ncol)
            os.write(dst_fd, out_buf.tobytes())
            written += nsz
            conv_done += 1
            if conv_done % 15 == 0 or conv_done == vq3_total:
                print(f"  Converted {conv_done}/{vq3_total} VQ3_0 tensors in {os.path.basename(dst_path)}...", flush=True)

        try:
            os.posix_fadvise(src_fd, (src_abs // 4096) * 4096, (osz // 4096) * 4096, os.POSIX_FADV_DONTNEED)
            os.posix_fadvise(dst_fd, (target_pos // 4096) * 4096, (nsz // 4096) * 4096, os.POSIX_FADV_DONTNEED)
        except Exception:
            pass

    os.fsync(dst_fd)
    os.close(dst_fd)
    src_mm.close()
    os.close(src_fd)
    print(f"[DSV4-VQ3] Completed {os.path.basename(dst_path)} in {time.time() - t_start:.1f}s", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Build DeepSeek-V4-Flash-0731 LUT-VQ3-64 + Fractal-64 hybrid 3-shard GGUF")
    ap.add_argument(
        "--input-dir",
        default=os.environ.get("DSV4_FRACTAL_DIR", "./workspace/models"),
        help="Directory containing DeepSeek-V4-Flash-0731-Fractal-0000{1..3}-of-00003.gguf",
    )
    ap.add_argument(
        "--output-dir",
        default=os.environ.get("DSV4_VQ3_DIR", "./dsv4-vq3"),
        help="Output directory for DeepSeek-V4-Flash-0731-VQ3-0000{1..3}-of-00003.gguf",
    )
    ap.add_argument("--lib", default=os.environ.get("VQ3_ENC_LIB", ""), help="Optional path to libvq3_enc.so")
    ap.add_argument("--aggressive", action="store_true", help="Also transcode MLA expansion tensors (56.07 GiB ablation)")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    lib = load_vq3_encoder(args.lib)
    for s_id in ("00001", "00002", "00003"):
        s_in = os.path.join(args.input_dir, f"DeepSeek-V4-Flash-0731-Fractal-{s_id}-of-00003.gguf")
        s_out = os.path.join(args.output_dir, f"DeepSeek-V4-Flash-0731-VQ3-{s_id}-of-00003.gguf")
        transcode_shard(lib, s_in, s_out, aggressive=args.aggressive)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
