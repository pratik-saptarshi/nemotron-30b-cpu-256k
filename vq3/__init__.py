"""
vq3 — LUT-VQ3-64 (24-Byte / 3.00 bpw) 3D Vector Quantization & AVX-512 VBMI Reference Codec
===========================================================================================
Provides:
- `train_global_lut_vq3_codebook`, `encode_lut_vq3`, `decode_lut_vq3_vbmi`, `quant_dequant_with_global_vq3`:
  128-entry 3D Vector Quantizer in `int8` (384 bytes = 6 ZMM registers `T0, T1, T2` + per-tensor fp32
  scale `g`) decoded via 22-slot stride-22 + `VPERMB` + 3x `_mm512_mask2_permutex2var_epi8`,
  exploiting x86 hardware Bit-7 ignore to store 22 MSB scale bits + 2 Bit-6 scale bits (`24` bits = `8 x 3-bit`).
- `quant_dequant_c5`: Scalar Base-5 (`5^3 = 125 <= 128`) mixed-radix quantizer with optimal ratio `r = 2.35`.
- `quant_dequant_f64_8state`: `Fractal-64` (`3.75 bpw`, 30B) and `F8-lean` (`3.50 bpw`, 28B) reference codecs.
- `pack_vq3_block_24b`, `unpack_vq3_block_24b_vbmi`, `verify_phase0_bit_exact`: Bit-exact 24-byte block
  packer/unpacker and verification suite.
"""

from .lut_vq3_quantizer import (
    build_c5_luts,
    build_vq3_luts_from_int8,
    decode_lut_vq3_vbmi,
    encode_lut_vq3,
    pack_vq3_block_24b,
    quant_dequant_c5,
    quant_dequant_f64_8state,
    quant_dequant_with_global_vq3,
    sim_vpermi2b,
    train_and_quant_lut_vq3,
    train_global_lut_vq3_codebook,
    unpack_vq3_block_24b_vbmi,
    verify_phase0_bit_exact,
)

__all__ = [
    "build_c5_luts",
    "build_vq3_luts_from_int8",
    "decode_lut_vq3_vbmi",
    "encode_lut_vq3",
    "pack_vq3_block_24b",
    "quant_dequant_c5",
    "quant_dequant_f64_8state",
    "quant_dequant_with_global_vq3",
    "sim_vpermi2b",
    "train_and_quant_lut_vq3",
    "train_global_lut_vq3_codebook",
    "unpack_vq3_block_24b_vbmi",
    "verify_phase0_bit_exact",
]
