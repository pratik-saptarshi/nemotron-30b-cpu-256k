#!/usr/bin/env python3
"""
Build the NVIDIA-Nemotron-3.5-30B-A3B-VQ3-64 hybrid GGUF (13.30 GiB).

Default conservative hybrid layout:
  - Boundary MoE layers (blk.1, 3, 49, 51) + all 23 blk.*.ffn_down_exps.weight
    remain in Fractal-64 (GGML_TYPE_Q2_0 = 42, 3.75 bpw, 30 B / 64 weights).
  - 19 interior blk.6..47.ffn_up_exps.weight (pre-ReLU^2) are quantized directly
    from the uncompressed BF16 GGUF into LUT-VQ3-64 (GGML_TYPE_VQ3_0 = 43,
    3.00 bpw, 24 B / 64 weights).

Pass --all-up-exps to encode all 23 ffn_up_exps tensors (13.08 GiB ablation).
"""

import argparse
import ctypes
import mmap
import os
import subprocess
import struct
import time
from pathlib import Path

import numpy as np

GGML_TYPE_BF16 = 30
GGML_TYPE_VQ3_0 = 43
ANCHOR_MOE_LAYERS = (1, 3, 49, 51)


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
    lib.quantize_bf16_to_vq3_0_omp.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64,
    ]
    lib.quantize_bf16_to_vq3_0_omp.restype = None
    return lib


def parse_gguf(f):
    f.seek(0)
    magic, version, n_tensors, n_kv = struct.unpack("<4sIQQ", f.read(24))
    if magic != b"GGUF":
        raise ValueError(f"Not a GGUF file (magic={magic!r})")

    def read_str():
        length = struct.unpack("<Q", f.read(8))[0]
        return f.read(length).decode("utf-8", errors="replace")

    def skip_val(vtype):
        if vtype in (0, 1, 7):
            f.seek(1, 1)
        elif vtype in (2, 3):
            f.seek(2, 1)
        elif vtype in (4, 5, 6):
            f.seek(4, 1)
        elif vtype in (10, 11, 12):
            f.seek(8, 1)
        elif vtype == 8:
            length = struct.unpack("<Q", f.read(8))[0]
            f.seek(length, 1)
        elif vtype == 9:
            atype, alen = struct.unpack("<IQ", f.read(12))
            for _ in range(alen):
                skip_val(atype)
        else:
            raise ValueError(f"Unknown GGUF metadata type {vtype}")

    for _ in range(n_kv):
        _ = read_str()
        vtype = struct.unpack("<I", f.read(4))[0]
        skip_val(vtype)

    kv_end = f.tell()
    tensors = []
    for _ in range(n_tensors):
        name = read_str()
        n_dims = struct.unpack("<I", f.read(4))[0]
        dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(n_dims)]
        ttype, offset = struct.unpack("<IQ", f.read(12))
        tensors.append({
            "name": name,
            "n_dims": n_dims,
            "dims": dims,
            "type": ttype,
            "offset": offset,
        })
    header_end = f.tell()
    alignment = 32
    data_start = (header_end + alignment - 1) // alignment * alignment
    return version, n_tensors, n_kv, kv_end, tensors, header_end, data_start


def main() -> int:
    ap = argparse.ArgumentParser(description="Build Nemotron-3.5-30B-A3B LUT-VQ3-64 + Fractal-64 hybrid GGUF")
    ap.add_argument(
        "--fractal-gguf",
        default=os.environ.get("NEMOTRON_FRACTAL_GGUF", "/opt/models/nemotron/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Fractal-64.gguf"),
        help="Input Fractal-64 GGUF path",
    )
    ap.add_argument(
        "--bf16-gguf",
        default=os.environ.get("NEMOTRON_BF16_GGUF", "/usr/share/ollama/.ollama/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16.gguf"),
        help="Input BF16 master GGUF path",
    )
    ap.add_argument(
        "--output",
        default=os.environ.get("NEMOTRON_VQ3_OUT", "./NVIDIA-Nemotron-3.5-30B-A3B-VQ3-64.gguf"),
        help="Output VQ3-64 hybrid GGUF path",
    )
    ap.add_argument("--lib", default=os.environ.get("VQ3_ENC_LIB", ""), help="Optional path to libvq3_enc.so")
    ap.add_argument("--all-up-exps", action="store_true", help="Encode all 23 ffn_up_exps layers (13.08 GiB ablation)")
    args = ap.parse_args()

    lib = load_vq3_encoder(args.lib)
    skip_layers = () if args.all_up_exps else ANCHOR_MOE_LAYERS

    t0 = time.time()
    with open(args.fractal_gguf, "rb") as ff, open(args.bf16_gguf, "rb") as fb:
        _, _, _, kv_end, f_tensors, f_hdr_end, f_data_start = parse_gguf(ff)
        _, _, _, _, b_tensors, _, b_data_start = parse_gguf(fb)
        bf16_map = {t["name"]: t for t in b_tensors}

        sorted_ft = sorted(f_tensors, key=lambda x: x["offset"])
        ff.seek(0, os.SEEK_END)
        f_file_size = ff.tell()

        new_offset = 0
        alignment = 32
        vq3_target_total = 0
        for t in f_tensors:
            n_el = 1
            for d in t["dims"]:
                n_el *= d
            t["n_el"] = n_el
            if t["name"].endswith(".ffn_up_exps.weight") and int(t["name"].split(".")[1]) not in skip_layers:
                t["new_type"] = GGML_TYPE_VQ3_0
                t["new_bytes"] = (n_el // 64) * 24
                t["old_bytes"] = (n_el // 64) * 30
                vq3_target_total += 1
            else:
                t["new_type"] = t["type"]
                idx_s = sorted_ft.index(t)
                if idx_s + 1 < len(sorted_ft):
                    span = sorted_ft[idx_s + 1]["offset"] - t["offset"]
                else:
                    span = f_file_size - (f_data_start + t["offset"])
                t["old_bytes"] = span
                t["new_bytes"] = span
            t["new_offset"] = new_offset
            padded = (t["new_bytes"] + alignment - 1) // alignment * alignment
            new_offset += padded

        print(
            f"Parsed headers: {len(f_tensors)} tensors, converting {vq3_target_total} ffn_up_exps tensors, "
            f"output size ~{(f_data_start + new_offset) / (1024 ** 3):.2f} GiB",
            flush=True,
        )

        mm_f = mmap.mmap(ff.fileno(), 0, access=mmap.ACCESS_READ)
        mm_b = mmap.mmap(fb.fileno(), 0, access=mmap.ACCESS_READ)

        with open(args.output, "wb") as fout:
            ff.seek(0)
            fout.write(ff.read(kv_end))
            for t in f_tensors:
                nb = t["name"].encode("utf-8")
                fout.write(struct.pack("<Q", len(nb)))
                fout.write(nb)
                fout.write(struct.pack("<I", t["n_dims"]))
                for d in t["dims"]:
                    fout.write(struct.pack("<Q", d))
                fout.write(struct.pack("<IQ", t["new_type"], t["new_offset"]))
            assert fout.tell() == f_hdr_end
            if f_data_start > f_hdr_end:
                fout.write(b"\x00" * (f_data_start - f_hdr_end))

            vq3_cnt = 0
            for t in f_tensors:
                target_pos = f_data_start + t["new_offset"]
                cur_pos = fout.tell()
                if target_pos > cur_pos:
                    fout.write(b"\x00" * (target_pos - cur_pos))
                assert fout.tell() == target_pos

                if t["new_type"] == GGML_TYPE_VQ3_0:
                    bt = bf16_map[t["name"]]
                    t_type = bt["type"]
                    assert t_type == GGML_TYPE_BF16, f"Expected BF16 ({GGML_TYPE_BF16}), got {t_type}"
                    n_el = t["n_el"]
                    ncol = t["dims"][0]
                    nrow = n_el // ncol
                    b_off = b_data_start + bt["offset"]
                    src_ptr = np.frombuffer(mm_b, dtype=np.uint8, count=n_el * 2, offset=b_off).ctypes.data
                    out_buf = np.empty(t["new_bytes"], dtype=np.uint8)
                    dst_ptr = out_buf.ctypes.data
                    ts = time.time()
                    lib.quantize_bf16_to_vq3_0_omp(src_ptr, dst_ptr, nrow, ncol)
                    fout.write(out_buf.tobytes())
                    mm_b.madvise(mmap.MADV_DONTNEED, (b_off // 4096) * 4096, (n_el * 2 // 4096) * 4096)
                    vq3_cnt += 1
                    print(
                        f"[{vq3_cnt:2d}/{vq3_target_total}] Encoded {t['name']} ({n_el / 1e6:.1f}M w) "
                        f"from BF16 -> VQ3_0 in {time.time() - ts:.2f}s",
                        flush=True,
                    )
                else:
                    f_off = f_data_start + t["offset"]
                    fout.write(mm_f[f_off : f_off + t["old_bytes"]])

            rem = fout.tell() % alignment
            if rem:
                fout.write(b"\x00" * (alignment - rem))

        mm_f.close()
        mm_b.close()

    print(f"DONE in {time.time() - t0:.1f}s: {args.output} ({os.path.getsize(args.output) / (1024 ** 3):.2f} GiB)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
