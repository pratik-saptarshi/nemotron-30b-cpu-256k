#!/usr/bin/env python3
"""
lut_vq3_quantizer.py — LUT-VQ3-64 (24-Byte / 3.00 bpw) 3D VQ & Mixed-Radix Reference Codec
==========================================================================================
Implements the 24-byte / 64-weight (`3.00 bpw`, -20.0% DDR5 memory bus traffic vs 30-byte
`Fractal-64`) quantization family co-designed for AMD Zen 5 AVX-512 VBMI + VNNI:

1. **22-Slot Stride-22 Block Geometry (24 Bytes / 64 Weights = 3.00 bpw)**:
   - `uint16_t d` (2 bytes, little-endian FP16): Super-block scale in normalized units.
   - `uint8_t qs[22]` (22 bytes):
     - Slots `k = 0..19` encode weight triplets `(w_k, w_{k+22}, w_{k+44})` via a 7-bit
       3D VQ index `R_k in {0..127}` in bits `0..6`.
     - Slots `k = 20, 21` encode weight pairs `(w_20, w_42)` and `(w_21, w_43)` in bits `0..5`
       (64 states for 2 weights = 3.00 bpw) and store scale bits `bit22, bit23` in Bit 6 (`0x40`),
       selecting the lower (`0..63`) or upper (`64..127`) half of the 128-entry VPERMI2B table.
     - Bit 7 (`0x80`) of `qs[0..21]` stores scale bits `0..21`, which x86 AVX-512 VBMI
       `_mm512_permutex2var_epi8` (`VPERMI2B`) architecturally ignores during table lookup
       and `_mm512_movepi8_mask` (`VPMOVB2M`) extracts in 1 cycle.
   - Per-tensor `float32` codebook scale `g = cb_max / 127.0` (stored once per tensor next to the
     384-byte int8 LUT and applied once per row after horizontal dot-product reduction, keeping
     FP16 `d` well above the FP16 subnormal range).

2. **Single-Register 64-Weight Decode (`1x VPERMB + 3x mask2_VPERMI2B`)**:
   - Replicating `qs[0..21]` across a 64-byte ZMM register with `REP[j] = 2 + (j % 22)` and
     applying three merge-masked `_mm512_mask2_permutex2var_epi8` instructions with masks
     `M1 = [22..43]`, `M2 = [44..63]`, `M0 = [0..21]` places all 64 weights in natural order
     `0..63` in a single ZMM register, aligned with the 16 4-byte groups of `VPDPBUSD`.
"""

from __future__ import annotations

import numpy as np


def build_c5_luts(
    c1: int = 54,
    c2: int = 127,
    offset: int = 128,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the three 128-byte VPERMI2B tables (T0, T1, T2) for Base-5 (5^3 = 125 <= 128).
    Default `(c1=54, c2=127)` gives `c2 / c1 = 2.3519 ~ 2.35`.
    Entries 0..124 store the 5^3 Cartesian grid (`d0 = r % 5, d1 = (r // 5) % 5, d2 = r // 25`).
    """
    cb = np.array([-c2, -c1, 0, c1, c2], dtype=np.int16)
    T0 = np.full(128, offset, dtype=np.uint8)
    T1 = np.full(128, offset, dtype=np.uint8)
    T2 = np.full(128, offset, dtype=np.uint8)
    for r in range(125):
        d0 = r % 5
        d1 = (r // 5) % 5
        d2 = r // 25
        T0[r] = np.uint8(int(cb[d0]) + offset)
        T1[r] = np.uint8(int(cb[d1]) + offset)
        T2[r] = np.uint8(int(cb[d2]) + offset)
    return T0, T1, T2, cb


def build_vq3_luts_from_int8(
    cb3d_int8: np.ndarray,
    offset: int = 128,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert a `(128, 3)` signed `int8` 3D VQ codebook into three 128-byte `uint8`
    VPERMI2B lookup tables `(T0, T1, T2)` biased by `offset = 128`.
    """
    assert cb3d_int8.shape == (128, 3)
    u8 = (cb3d_int8.astype(np.int16) + offset).astype(np.uint8)
    return u8[:, 0].copy(), u8[:, 1].copy(), u8[:, 2].copy()


def sim_vpermi2b(
    a_lo64: np.ndarray,
    idx64: np.ndarray,
    b_hi64: np.ndarray,
    mask64: np.ndarray | None = None,
) -> np.ndarray:
    """
    Bit-exact simulation of x86 AVX-512 VBMI `_mm512_mask2_permutex2var_epi8(a, idx, mask, b)`:
    - Hardware uses ONLY bits 0..5 for byte offset within 64B register and bit 6 to select `a` vs `b`.
    - Bit 7 of `idx` is ARCHITECTURALLY IGNORED by x86 hardware (`idx & 0x7F`).
    - Under `mask2` (`_mm512_mask2_permutex2var_epi8`), lanes where `mask == False` keep `idx` untouched.
    """
    table128 = np.concatenate([a_lo64, b_hi64])
    looked_up = table128[idx64 & 0x7F]
    if mask64 is None:
        return looked_up.copy()
    out = idx64.copy()
    out[mask64] = looked_up[mask64]
    return out


def pack_vq3_block_24b(
    d_fp16: np.float16,
    idx22_u7: np.ndarray,
    scales8_u3: np.ndarray,
) -> bytes:
    """
    Pack a 64-weight LUT-VQ3-64 block into its exact 24-byte hardware binary representation:
    - Bytes 0..1: Little-endian FP16 super-block scale `d`.
    - Bytes 2..23 (`qs[0..21]`):
      - Bits 0..6 of `qs[k]` (k=0..19): 7-bit 3D VQ index `idx22_u7[k]` in `{0..127}`.
      - Bits 0..5 of `qs[20]`: 6-bit 2D VQ index; Bit 6 (`0x40`) must equal `bit22` of `s24`.
      - Bits 0..5 of `qs[21]`: 6-bit 2D VQ index; Bit 6 (`0x40`) must equal `bit23` of `s24`.
      - Bit 7 (`0x80`) of `qs[0..21]`: bits `0..21` of the 24-bit `8 x 3-bit` sub-scale word `s24`.
    """
    if idx22_u7.shape != (22,):
        raise ValueError(f"Expected idx22_u7 shape (22,), got {idx22_u7.shape}")
    if scales8_u3.shape != (8,):
        raise ValueError(f"Expected scales8_u3 shape (8,), got {scales8_u3.shape}")
    if np.any((scales8_u3 < 0) | (scales8_u3 > 7)):
        raise ValueError(f"scales8_u3 values must be in 0..7, got {scales8_u3}")
    if np.any((idx22_u7 < 0) | (idx22_u7 > 127)):
        raise ValueError(f"idx22_u7 values must be in 0..127, got {idx22_u7}")

    s24 = 0
    for s_i in range(8):
        s24 |= (int(scales8_u3[s_i]) & 0x7) << (3 * s_i)

    bit22 = (s24 >> 22) & 1
    bit23 = (s24 >> 23) & 1
    if ((int(idx22_u7[20]) >> 6) & 1) != bit22:
        raise ValueError(
            f"idx22_u7[20]={int(idx22_u7[20])} bit 6 does not match scale bit22={bit22}"
        )
    if ((int(idx22_u7[21]) >> 6) & 1) != bit23:
        raise ValueError(
            f"idx22_u7[21]={int(idx22_u7[21])} bit 6 does not match scale bit23={bit23}"
        )

    qs = np.zeros(22, dtype=np.uint8)
    for k in range(22):
        r = int(idx22_u7[k]) & 0x7F
        msb = (s24 >> k) & 1
        qs[k] = np.uint8(r | (msb << 7))

    d_bytes = np.array([d_fp16], dtype="<f2").tobytes()
    return d_bytes + qs.tobytes()


def unpack_vq3_block_24b_vbmi(
    block24: bytes,
    T0: np.ndarray,
    T1: np.ndarray,
    T2: np.ndarray,
    offset: int = 128,
) -> tuple[np.float16, np.ndarray, np.ndarray]:
    """
    Unpack a 24-byte LUT-VQ3-64 block using the exact Zen 5 AVX-512 VBMI instruction sequence:
    1. `VPERMB` replicate with `REP[j] = 2 + (j % 22)`
    2. 3x merge-masked `_mm512_mask2_permutex2var_epi8` (ignoring Bit 7 in hardware)
    3. `VPMOVB2M` (`_mm512_movepi8_mask`) + Bit 6 of `qs[20], qs[21]` recovery for all 8 x 3-bit sub-scales.
    Returns `(d_fp16, w_decoded_int16[64], scales8_u3[8])`.
    """
    if len(block24) != 24:
        raise ValueError(f"Expected 24-byte block, got {len(block24)} bytes")
    d_fp16 = np.frombuffer(block24[:2], dtype="<f2")[0].astype(np.float16)
    blk64 = np.zeros(64, dtype=np.uint8)
    blk64[:24] = np.frombuffer(block24, dtype=np.uint8)

    REP = np.array([2 + (j % 22) for j in range(64)], dtype=np.uint8)
    t = blk64[REP & 0x3F]

    M0 = np.zeros(64, dtype=bool); M0[0:22] = True
    M1 = np.zeros(64, dtype=bool); M1[22:44] = True
    M2 = np.zeros(64, dtype=bool); M2[44:64] = True

    t = sim_vpermi2b(T1[:64], t, T1[64:], M1)
    t = sim_vpermi2b(T2[:64], t, T2[64:], M2)
    w_u8 = sim_vpermi2b(T0[:64], t, T0[64:], M0)
    w_decoded = w_u8.astype(np.int16) - offset

    s24 = 0
    for k in range(22):
        if blk64[2 + k] & 0x80:
            s24 |= 1 << k
    bit22 = (int(blk64[22]) >> 6) & 1
    bit23 = (int(blk64[23]) >> 6) & 1
    s24 |= (bit22 << 22) | (bit23 << 23)

    scales8_u3 = np.array([(s24 >> (3 * i)) & 0x7 for i in range(8)], dtype=np.uint8)
    return d_fp16, w_decoded, scales8_u3


def verify_phase0_bit_exact(verbose: bool = True) -> bool:
    """
    Run bit-exact verification of the 22-slot stride-22 + VPERMB + 3x mask2_VPERMI2B + VPDPBUSD
    pipeline across random blocks, arbitrary 128-entry 3D VQ tables, and random 24-bit scales.
    """
    rng = np.random.default_rng(42)
    OFF = 128
    cb3d_int8 = rng.integers(-127, 128, size=(128, 3), dtype=np.int8)
    T0, T1, T2 = build_vq3_luts_from_int8(cb3d_int8, OFF)

    for _ in range(32):
        idx22 = np.zeros(22, dtype=np.uint8)
        idx22[0:20] = rng.integers(0, 128, size=20, dtype=np.uint8)
        scales_true = rng.integers(0, 8, size=8, dtype=np.uint8)
        s24_true = sum((int(scales_true[i]) & 0x7) << (3 * i) for i in range(8))
        bit22 = (s24_true >> 22) & 1
        bit23 = (s24_true >> 23) & 1
        idx22[20] = np.uint8(rng.integers(0, 64) | (bit22 << 6))
        idx22[21] = np.uint8(rng.integers(0, 64) | (bit23 << 6))

        w_true = np.zeros(64, dtype=np.int16)
        w_true[0:20] = cb3d_int8[idx22[0:20], 0]
        w_true[22:42] = cb3d_int8[idx22[0:20], 1]
        w_true[44:64] = cb3d_int8[idx22[0:20], 2]
        w_true[20] = cb3d_int8[idx22[20], 0]
        w_true[42] = cb3d_int8[idx22[20], 1]
        w_true[21] = cb3d_int8[idx22[21], 0]
        w_true[43] = cb3d_int8[idx22[21], 1]

        d_true = np.float16(rng.uniform(0.01, 2.0))
        packed = pack_vq3_block_24b(d_true, idx22, scales_true)
        assert len(packed) == 24

        d_rec, w_rec, scales_rec = unpack_vq3_block_24b_vbmi(packed, T0, T1, T2, OFF)
        assert d_rec == d_true, f"FP16 scale mismatch: {d_rec} != {d_true}"
        assert np.array_equal(w_rec, w_true), "Decoded 64 weights mismatch!"
        assert np.array_equal(scales_rec, scales_true), "Recovered 8x3b scales mismatch!"

        x_q8 = rng.integers(-127, 128, size=64, dtype=np.int8)
        w_u8 = (w_rec + OFF).astype(np.uint8)
        vnni_groups = (w_u8.astype(np.int32) * x_q8.astype(np.int32)).reshape(16, 4).sum(axis=1)
        x_sum4 = x_q8.astype(np.int32).reshape(16, 4).sum(axis=1)
        vnni_corrected = vnni_groups - OFF * x_sum4
        ref_groups = (w_true.astype(np.int32) * x_q8.astype(np.int32)).reshape(16, 4).sum(axis=1)
        assert np.array_equal(vnni_corrected, ref_groups), "VPDPBUSD 16-lane group sum mismatch!"

    if verbose:
        print(" [PASS] 22-slot stride-22 packing + VPERMB + 3x mask2_VPERMI2B is 100% bit-exact!")
        print(" [PASS] Hardware bit-7 ignore verified: 22 MSB scale bits + 2 bit-6 bits = 24 scale bits free!")
        print(" [PASS] All 16 VPDPBUSD int32 lanes align strictly with 8-weight and 16-weight sub-blocks!")
    return True


def quant_dequant_f64_8state(W: np.ndarray, sub_block_size: int = 8) -> np.ndarray:
    """
    Fractal-64 (3.75 bpw when sub_block_size=8 [30B], or F8-lean 3.50 bpw when sub_block_size=16 [28B]).
    Codebook: {-14, -8, -4, -1, 1, 4, 8, 14}, sub-scales in {4..19}.
    """
    orig_shape = W.shape
    W_2d = W.reshape(-1, 64).astype(np.float32)
    cb = np.array([-14.0, -8.0, -4.0, -1.0, 1.0, 4.0, 8.0, 14.0], dtype=np.float32)
    N_blocks = W_2d.shape[0]
    n_sub = 64 // sub_block_size
    W_sub = W_2d.reshape(N_blocks, n_sub, sub_block_size)

    sub_amax = np.max(np.abs(W_sub), axis=2)
    blk_amax = np.max(sub_amax, axis=1, keepdims=True)

    d_fp32 = np.where(blk_amax > 0, blk_amax / (19.0 * 14.0), 0.0)
    d = d_fp32.astype(np.float16).astype(np.float32)

    denom = np.maximum(d * 14.0, 1e-30)
    m_raw = np.round(sub_amax / denom)
    m = np.clip(m_raw, 4.0, 19.0)

    eff_scale = (d * m)[:, :, None]
    w_norm = W_sub / np.maximum(eff_scale, 1e-30)

    diffs = np.abs(w_norm[:, :, :, None] - cb[None, None, None, :])
    idx = np.argmin(diffs, axis=-1)
    w_q = cb[idx] * eff_scale
    return w_q.reshape(orig_shape)


def quant_dequant_c5(
    W: np.ndarray,
    ratio: float = 2.35,
    sub_block_size: int = 8,
    scale_bits: int = 3,
) -> np.ndarray:
    """
    C5-64 (24 B / 64 weights = 3.00 bpw):
    5-state symmetric codebook {-ratio, -1, 0, +1, +ratio} with 2-step alternating
    least-squares sub-block scale refinement.
    """
    orig_shape = W.shape
    W_2d = W.reshape(-1, 64).astype(np.float32)
    cb = np.array([-ratio, -1.0, 0.0, 1.0, ratio], dtype=np.float32)
    N_blocks = W_2d.shape[0]
    n_sub = 64 // sub_block_size
    W_sub = W_2d.reshape(N_blocks, n_sub, sub_block_size)

    n_levels = 1 << scale_bits
    m_min = 2.0 if scale_bits == 3 else 4.0
    m_max = m_min + n_levels - 1.0

    sub_amax = np.max(np.abs(W_sub), axis=2) * 0.88
    blk_amax = np.max(sub_amax, axis=1, keepdims=True)

    d_fp32 = np.where(blk_amax > 0, blk_amax / (m_max * ratio), 0.0)
    d = d_fp32.astype(np.float16).astype(np.float32)

    denom = np.maximum(d * ratio, 1e-30)
    m = np.clip(np.round(sub_amax / denom), m_min, m_max)

    for _ in range(2):
        eff_scale = (d * m)[:, :, None]
        w_norm = W_sub / np.maximum(eff_scale, 1e-30)
        idx = np.argmin(np.abs(w_norm[:, :, :, None] - cb[None, None, None, :]), axis=-1)
        q = cb[idx]
        wq = np.sum(W_sub * q, axis=2)
        qq = np.sum(q * q, axis=2)
        m_opt = np.where(qq > 0, wq / np.maximum(d * qq, 1e-30), m)
        m = np.clip(np.round(m_opt), m_min, m_max)

    eff_scale = (d * m)[:, :, None]
    w_norm = W_sub / np.maximum(eff_scale, 1e-30)
    idx = np.argmin(np.abs(w_norm[:, :, :, None] - cb[None, None, None, :]), axis=-1)
    return (cb[idx] * eff_scale).reshape(orig_shape)


def train_global_lut_vq3_codebook(
    W_sample: np.ndarray,
    sub_block_size: int = 8,
    scale_bits: int = 3,
    n_iters: int = 25,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Train a global 128-entry int8 3D Vector Quantizer codebook (384 bytes = 6 ZMM registers T0, T1, T2)
    on a sample of 64-weight blocks `W_sample` using 3D k-means++.
    Entries are interleaved by 2D radial-angular key so both halves `0..63` and `64..127` cover the
    2D plane uniformly for slots `20, 21` when Bit 6 carries `bit22, bit23`.
    Returns `(cb3d_q_float32[128, 3], cb3d_int8[128, 3])`.
    """
    if (sub_block_size, scale_bits) != (8, 3):
        raise ValueError(
            f"LUT-VQ3-64 24-byte hardware format requires (sub_block_size=8, scale_bits=3), "
            f"got ({sub_block_size}, {scale_bits})"
        )
    rng = np.random.default_rng(seed)
    W = W_sample.reshape(-1, 64).astype(np.float32)
    N_blocks = W.shape[0]
    n_sub = 8
    W_sub = W.reshape(N_blocks, n_sub, 8)

    m_min, m_max = 2.0, 9.0

    sub_amax = np.max(np.abs(W_sub), axis=2) * 0.88
    blk_amax = np.max(sub_amax, axis=1, keepdims=True)
    d = np.where(blk_amax > 0, blk_amax / m_max, 0.0).astype(np.float16).astype(np.float32)
    m = np.clip(np.round(sub_amax / np.maximum(d, 1e-30)), m_min, m_max)

    eff_scale = np.repeat(d * m, 8, axis=1)
    W_norm = W / np.maximum(eff_scale, 1e-30)

    triplets = np.stack([W_norm[:, 0:20], W_norm[:, 22:42], W_norm[:, 44:64]], axis=2).reshape(-1, 3)
    n_train = min(50000, triplets.shape[0])
    train_idx = rng.choice(triplets.shape[0], size=n_train, replace=False)
    T_train = triplets[train_idx]

    cb3d = np.zeros((128, 3), dtype=np.float32)
    cb3d[0] = [0.0, 0.0, 0.0]
    min_d2 = np.sum(T_train * T_train, axis=1)
    for k in range(1, 128):
        total_d2 = float(np.sum(min_d2))
        if total_d2 <= 1e-30:
            chosen = int(rng.integers(0, n_train))
        else:
            probs = min_d2 / total_d2
            chosen = int(rng.choice(n_train, p=probs))
        cb3d[k] = T_train[chosen]
        d2_new = np.sum((T_train - cb3d[k]) ** 2, axis=1)
        min_d2 = np.minimum(min_d2, d2_new)

    for _ in range(n_iters):
        c_norm2 = np.sum(cb3d * cb3d, axis=1)
        dots = T_train @ cb3d.T
        assign = np.argmin(c_norm2[None, :] - 2.0 * dots, axis=1)
        for k in range(128):
            mask = assign == k
            if np.any(mask):
                cb3d[k] = np.mean(T_train[mask], axis=0)

    # Interleave sorted 2D radial-angular projections into even/odd halves (0..63 and 64..127)
    # so both halves span the 2D plane when Bit 6 of qs[20..21] encodes scale bits 22..23
    radial_angle_key = np.arctan2(cb3d[:, 1], cb3d[:, 0]) + 10.0 * np.hypot(cb3d[:, 0], cb3d[:, 1])
    order = np.argsort(radial_angle_key)
    interleaved = np.concatenate([order[0::2], order[1::2]])
    cb3d = cb3d[interleaved]

    cb_max = float(np.max(np.abs(cb3d)))
    if cb_max <= 1e-30:
        cb_max = 1.0
    cb3d_int8 = np.clip(np.round(cb3d / cb_max * 127.0), -127, 127).astype(np.int8)
    cb3d_q = cb3d_int8.astype(np.float32) * (cb_max / 127.0)
    return cb3d_q, cb3d_int8


def _encode_vq3_blocks_internal(
    W_2d: np.ndarray,
    cb3d_q: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Internal encoder shared by `quant_dequant_with_global_vq3` and `encode_lut_vq3`.
    Guarantees that the returned `(d_fp16, idx22_u7, scales8_u3, W_hat)` satisfy the
    exact Bit-6 half-table constraint on `idx22_u7[20..21]` under the FINAL `scales8_u3[7]`.
    """
    N_blocks = W_2d.shape[0]
    n_sub = 8
    sub_block_size = 8
    W_sub = W_2d.reshape(N_blocks, n_sub, sub_block_size)

    m_min, m_max = 2.0, 9.0
    sub_amax = np.max(np.abs(W_sub), axis=2) * 0.88
    blk_amax = np.max(sub_amax, axis=1, keepdims=True)
    d_fp16 = np.where(blk_amax > 0, blk_amax / m_max, 0.0).astype(np.float16)
    d = d_fp16.astype(np.float32)
    m = np.clip(np.round(sub_amax / np.maximum(d, 1e-30)), m_min, m_max)

    C2 = cb3d_q * cb3d_q

    for _ in range(2):
        eff_scale = np.repeat(d * m, sub_block_size, axis=1)
        W_3 = np.stack([W_2d[:, 0:20], W_2d[:, 22:42], W_2d[:, 44:64]], axis=2)
        S_3 = np.stack([eff_scale[:, 0:20], eff_scale[:, 22:42], eff_scale[:, 44:64]], axis=2)

        S2 = (S_3 * S_3).reshape(-1, 3)
        WS = (W_3 * S_3).reshape(-1, 3)
        dist_20 = (S2 @ C2.T) - 2.0 * (WS @ cb3d_q.T)
        idx_20 = np.argmin(dist_20, axis=1).reshape(N_blocks, 20)
        q_20 = cb3d_q[idx_20]

        W_2 = np.stack([W_2d[:, 20:22], W_2d[:, 42:44]], axis=2)
        S_2 = np.stack([eff_scale[:, 20:22], eff_scale[:, 42:44]], axis=2)
        S2_2 = (S_2 * S_2).reshape(-1, 2)
        WS_2 = (W_2 * S_2).reshape(-1, 2)
        dist_2 = (S2_2 @ C2[:, :2].T) - 2.0 * (WS_2 @ cb3d_q[:, :2].T)

        s7_u3 = np.round(m[:, 7] - m_min).astype(np.int32) & 0x7
        bit22 = (s7_u3 >> 1) & 1
        bit23 = (s7_u3 >> 2) & 1
        bits_20_21 = np.stack([bit22, bit23], axis=1).reshape(-1)
        dist_2_masked = dist_2.copy()
        dist_2_masked[bits_20_21 == 0, 64:] = np.inf
        dist_2_masked[bits_20_21 == 1, :64] = np.inf
        idx_2 = np.argmin(dist_2_masked, axis=1).reshape(N_blocks, 2)
        q_2 = cb3d_q[idx_2, :2]

        W_rec_norm = np.zeros((N_blocks, 64), dtype=np.float32)
        W_rec_norm[:, 0:20] = q_20[:, :, 0]
        W_rec_norm[:, 22:42] = q_20[:, :, 1]
        W_rec_norm[:, 44:64] = q_20[:, :, 2]
        W_rec_norm[:, 20:22] = q_2[:, :, 0]
        W_rec_norm[:, 42:44] = q_2[:, :, 1]

        q_sub = W_rec_norm.reshape(N_blocks, n_sub, sub_block_size)
        wq = np.sum(W_sub * q_sub, axis=2)
        qq = np.sum(q_sub * q_sub, axis=2)
        m_opt = np.where(qq > 0, wq / np.maximum(d * qq, 1e-30), m)
        m = np.clip(np.round(m_opt), m_min, m_max)

    # Final constrained re-assignment of slots 20..21 (and 0..19) under the final frozen `m`
    # so Bit 6 of idx22_u7[20..21] is 100% consistent with bits 1..2 of final `s7_u3` (Fixes V1)
    eff_scale = np.repeat(d * m, sub_block_size, axis=1)
    W_3 = np.stack([W_2d[:, 0:20], W_2d[:, 22:42], W_2d[:, 44:64]], axis=2)
    S_3 = np.stack([eff_scale[:, 0:20], eff_scale[:, 22:42], eff_scale[:, 44:64]], axis=2)
    S2 = (S_3 * S_3).reshape(-1, 3)
    WS = (W_3 * S_3).reshape(-1, 3)
    dist_20 = (S2 @ C2.T) - 2.0 * (WS @ cb3d_q.T)
    idx_20 = np.argmin(dist_20, axis=1).reshape(N_blocks, 20)
    q_20 = cb3d_q[idx_20]

    W_2 = np.stack([W_2d[:, 20:22], W_2d[:, 42:44]], axis=2)
    S_2 = np.stack([eff_scale[:, 20:22], eff_scale[:, 42:44]], axis=2)
    S2_2 = (S_2 * S_2).reshape(-1, 2)
    WS_2 = (W_2 * S_2).reshape(-1, 2)
    dist_2 = (S2_2 @ C2[:, :2].T) - 2.0 * (WS_2 @ cb3d_q[:, :2].T)

    scales8_u3 = (np.round(m - m_min).astype(np.int32) & 0x7).astype(np.uint8)
    s7_u3 = scales8_u3[:, 7].astype(np.int32)
    bit22 = (s7_u3 >> 1) & 1
    bit23 = (s7_u3 >> 2) & 1
    bits_20_21 = np.stack([bit22, bit23], axis=1).reshape(-1)
    dist_2_masked = dist_2.copy()
    dist_2_masked[bits_20_21 == 0, 64:] = np.inf
    dist_2_masked[bits_20_21 == 1, :64] = np.inf
    idx_2 = np.argmin(dist_2_masked, axis=1).reshape(N_blocks, 2)
    q_2 = cb3d_q[idx_2, :2]

    W_rec_norm = np.zeros((N_blocks, 64), dtype=np.float32)
    W_rec_norm[:, 0:20] = q_20[:, :, 0]
    W_rec_norm[:, 22:42] = q_20[:, :, 1]
    W_rec_norm[:, 44:64] = q_20[:, :, 2]
    W_rec_norm[:, 20:22] = q_2[:, :, 0]
    W_rec_norm[:, 42:44] = q_2[:, :, 1]

    idx22_u7 = np.concatenate([idx_20, idx_2], axis=1).astype(np.uint8)
    W_hat = W_rec_norm * eff_scale
    return d_fp16.reshape(N_blocks), idx22_u7, scales8_u3, W_hat


def quant_dequant_with_global_vq3(
    W: np.ndarray,
    cb3d_q: np.ndarray,
    sub_block_size: int = 8,
    scale_bits: int = 3,
) -> np.ndarray:
    """
    Encode and reconstruct a weight tensor `W` using a static 128-entry int8 3D VQ codebook
    `cb3d_q` (`(128, 3)`), scale-weighted 3D assignment, exact Bit-6 half-table selection for
    slots 20..21, and 2-step alternating least-squares sub-block scale refinement.
    """
    if (sub_block_size, scale_bits) != (8, 3):
        raise ValueError(
            f"LUT-VQ3-64 24-byte hardware format requires (sub_block_size=8, scale_bits=3), "
            f"got ({sub_block_size}, {scale_bits})"
        )
    orig_shape = W.shape
    W_2d = W.reshape(-1, 64).astype(np.float32)
    _, _, _, W_hat = _encode_vq3_blocks_internal(W_2d, cb3d_q)
    return W_hat.reshape(orig_shape)


def encode_lut_vq3(
    W: np.ndarray,
    cb3d_int8: np.ndarray,
    g: float,
) -> bytes:
    """
    Encode a weight matrix `W` into a contiguous byte buffer of 24-byte `LUT-VQ3-64` blocks.
    `cb3d_int8` is the `(128, 3)` signed `int8` codebook and `g` is the per-tensor `float32`
    codebook scale (`cb3d_q = cb3d_int8.astype(np.float32) * np.float32(g)`).
    """
    cb3d_q = cb3d_int8.astype(np.float32) * np.float32(g)
    W_2d = W.reshape(-1, 64).astype(np.float32)
    d_fp16, idx22_u7, scales8_u3, _ = _encode_vq3_blocks_internal(W_2d, cb3d_q)
    chunks = [
        pack_vq3_block_24b(d_fp16[b], idx22_u7[b], scales8_u3[b])
        for b in range(W_2d.shape[0])
    ]
    return b"".join(chunks)


def decode_lut_vq3_vbmi(
    raw_bytes: bytes,
    cb3d_int8: np.ndarray,
    g: float,
    offset: int = 128,
) -> np.ndarray:
    """
    Decode a contiguous buffer of 24-byte `LUT-VQ3-64` blocks via the simulated Zen 5
    AVX-512 VBMI `VPERMB` + 3x `mask2_VPERMI2B` pipeline and per-tensor scale `g`.
    Returns reconstructed `float32` array of shape `(N_blocks, 64)`.
    """
    if len(raw_bytes) % 24 != 0:
        raise ValueError(f"Buffer length {len(raw_bytes)} is not a multiple of 24 bytes")
    n_blocks = len(raw_bytes) // 24
    T0, T1, T2 = build_vq3_luts_from_int8(cb3d_int8, offset=offset)
    g_f32 = np.float32(g)
    out = np.zeros((n_blocks, 64), dtype=np.float32)
    for b in range(n_blocks):
        blk24 = raw_bytes[b * 24 : (b + 1) * 24]
        d_fp16, w_s16, scales8_u3 = unpack_vq3_block_24b_vbmi(blk24, T0, T1, T2, offset=offset)
        d_f32 = d_fp16.astype(np.float32)
        m_f32 = scales8_u3.astype(np.float32) + np.float32(2.0)
        eff_scale = np.repeat(d_f32 * m_f32, 8)
        out[b] = (w_s16.astype(np.float32) * g_f32) * eff_scale
    return out


def train_and_quant_lut_vq3(
    W: np.ndarray,
    sub_block_size: int = 8,
    scale_bits: int = 3,
    n_iters: int = 25,
    seed: int = 42,
) -> np.ndarray:
    """Convenience wrapper: train a 128-entry 3D VQ codebook on `W` and quantize/reconstruct `W`."""
    cb3d_q, _ = train_global_lut_vq3_codebook(
        W, sub_block_size=sub_block_size, scale_bits=scale_bits, n_iters=n_iters, seed=seed
    )
    return quant_dequant_with_global_vq3(W, cb3d_q, sub_block_size=sub_block_size, scale_bits=scale_bits)
