/*
 * dsv4_singlepass.c — Single-Pass Safetensors (FP8_E4M3 / BF16 / MXFP4) -> GGUF Quantizer Library
 *
 * Build shared library (after building patched llama.cpp via scripts/build_zen5_llama.sh):
 *   gcc -O3 -march=native -fopenmp -fPIC -shared scripts/dsv4_singlepass.c \
 *       -I ./workspace/llama.cpp/ggml/src \
 *       -L ./workspace/llama.cpp/build/bin -Wl,-rpath,./workspace/llama.cpp/build/bin \
 *       -lggml-cpu -lggml-base -lm -o ./workspace/bin/libdsv4_singlepass.so
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <immintrin.h>
#include "vq3_codebook.h"

extern void ggml_cpu_init(void);
extern void ggml_quantize_init(int type);

extern void quantize_row_q2_0_ref(const float * x, void * y, int64_t k);
extern void quantize_row_q4_0_ref(const float * x, void * y, int64_t k);
extern void quantize_row_q8_0_ref(const float * x, void * y, int64_t k);
extern size_t quantize_q4_K(const float * src, void * dst, int64_t nrow, int64_t n_per_row, const float * quant_weights);
extern size_t quantize_q5_K(const float * src, void * dst, int64_t nrow, int64_t n_per_row, const float * quant_weights);
extern size_t quantize_q6_K(const float * src, void * dst, int64_t nrow, int64_t n_per_row, const float * quant_weights);
extern size_t quantize_iq2_xxs(const float * src, void * dst, int64_t nrow, int64_t n_per_row, const float * quant_weights);
extern size_t quantize_iq1_s(const float * src, void * dst, int64_t nrow, int64_t n_per_row, const float * quant_weights);
extern size_t quantize_iq1_m(const float * src, void * dst, int64_t nrow, int64_t n_per_row, const float * quant_weights);

extern void dequantize_row_q4_0(const void * x, float * y, int64_t k);
extern void dequantize_row_q8_0(const void * x, float * y, int64_t k);
extern void dequantize_row_q4_K(const void * x, float * y, int64_t k);
extern void dequantize_row_q5_K(const void * x, float * y, int64_t k);
extern void dequantize_row_q6_K(const void * x, float * y, int64_t k);
extern void dequantize_row_iq2_xxs(const void * x, float * y, int64_t k);
extern void dequantize_row_iq1_s(const void * x, float * y, int64_t k);
extern void dequantize_row_iq1_m(const void * x, float * y, int64_t k);
extern void dequantize_row_q2_0(const void * x, float * y, int64_t k);
extern void dequantize_row_vq3_0(const void * x, float * y, int64_t k);

#define QK_VQ3_0 64
typedef uint16_t ggml_half;
typedef struct { ggml_half d; uint8_t qs[22]; } block_vq3_0;

static inline float fp16_to_fp32(ggml_half h) { return _cvtsh_ss(h); }
static inline ggml_half fp32_to_fp16(float f) { return _cvtss_sh(f, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC); }
static inline float bf16_to_fp32(uint16_t b) {
    uint32_t u = ((uint32_t)b) << 16;
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

static float g_e8m0_f32[256];
static float g_e4m3_f32[256];
static const float g_fp4_f32[16] = {
    0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
   -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

static float g_cb_q0[128] __attribute__((aligned(64)));
static float g_cb_q1[128] __attribute__((aligned(64)));
static float g_cb_q2[128] __attribute__((aligned(64)));
static float g_cb_c20[128] __attribute__((aligned(64)));
static float g_cb_c21[128] __attribute__((aligned(64)));
static float g_cb_c22[128] __attribute__((aligned(64)));
static int g_sp_init = 0;

void sp_init(void) {
    if (g_sp_init) return;
    #pragma omp critical
    {
        if (!g_sp_init) {
            ggml_cpu_init();
            ggml_quantize_init(16);
            ggml_quantize_init(19);
            ggml_quantize_init(29);

            for (int e = 0; e < 256; ++e) {
                g_e8m0_f32[e] = (e == 255) ? 0.0f : ldexpf(1.0f, e - 127);
            }
            for (int x = 0; x < 256; ++x) {
                uint8_t abs_x = (uint8_t)(x & 0x7f);
                int sign = (x & 0x80) != 0;
                float val = 0.0f;
                if (abs_x != 0 && abs_x != 0x7f) {
                    int exp = (x >> 3) & 0x0f;
                    int man = x & 0x07;
                    val = (exp == 0) ? ldexpf((float)man, -9)
                                     : ldexpf(1.0f + (float)man / 8.0f, exp - 7);
                }
                g_e4m3_f32[x] = sign ? -val : val;
            }
            for (int c = 0; c < 128; ++c) {
                float v0 = (float)ggml_vq3_cb[c][0] * GGML_VQ3_G;
                float v1 = (float)ggml_vq3_cb[c][1] * GGML_VQ3_G;
                float v2 = (float)ggml_vq3_cb[c][2] * GGML_VQ3_G;
                g_cb_q0[c] = v0; g_cb_c20[c] = v0 * v0;
                g_cb_q1[c] = v1; g_cb_c21[c] = v1 * v1;
                g_cb_q2[c] = v2; g_cb_c22[c] = v2 * v2;
            }
            g_sp_init = 1;
        }
    }
}

static void quantize_row_vq3_0_fast(const float * restrict x, block_vq3_0 * restrict y, int64_t k) {
    const int64_t nb = k / QK_VQ3_0;
    const __m512i v_iota = _mm512_setr_epi32(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15);
    const __m512i v_step = _mm512_set1_epi32(16);
    const __m512 v_neg2  = _mm512_set1_ps(-2.0f);

    for (int64_t b = 0; b < nb; ++b) {
        const float * xb = x + b * 64;
        float sub_amax[8];
        float blk_amax = 0.0f;
        for (int s = 0; s < 8; ++s) {
            float amax = 0.0f;
            for (int j = 0; j < 8; ++j) {
                float av = fabsf(xb[s * 8 + j]);
                if (av > amax) amax = av;
            }
            sub_amax[s] = amax * 0.88f;
            if (sub_amax[s] > blk_amax) blk_amax = sub_amax[s];
        }

        ggml_half d_h = (blk_amax > 0.0f) ? fp32_to_fp16(blk_amax / 9.0f) : 0;
        float d = fp16_to_fp32(d_h);
        if (d <= 1e-30f) {
            y[b].d = 0;
            memset(y[b].qs, 0, 22);
            continue;
        }

        float m[8];
        for (int s = 0; s < 8; ++s) {
            float mr = roundf(sub_amax[s] / d);
            if (mr < 2.0f) mr = 2.0f;
            if (mr > 9.0f) mr = 9.0f;
            m[s] = mr;
        }

        uint8_t idx22[22];
        float w_rec[64];

        for (int iter = 0; iter < 2; ++iter) {
            float eff_s[64];
            for (int s = 0; s < 8; ++s) {
                float es = d * m[s];
                for (int j = 0; j < 8; ++j) eff_s[s * 8 + j] = es;
            }

            for (int j = 0; j < 20; ++j) {
                float w0 = xb[j],      s0 = eff_s[j];
                float w1 = xb[j + 22], s1 = eff_s[j + 22];
                float w2 = xb[j + 44], s2 = eff_s[j + 44];
                __m512 vs02 = _mm512_set1_ps(s0 * s0);
                __m512 vs12 = _mm512_set1_ps(s1 * s1);
                __m512 vs22 = _mm512_set1_ps(s2 * s2);
                __m512 vws0 = _mm512_mul_ps(v_neg2, _mm512_set1_ps(w0 * s0));
                __m512 vws1 = _mm512_mul_ps(v_neg2, _mm512_set1_ps(w1 * s1));
                __m512 vws2 = _mm512_mul_ps(v_neg2, _mm512_set1_ps(w2 * s2));

                __m512 best_d = _mm512_set1_ps(1e30f);
                __m512i best_i = _mm512_setzero_si512();
                __m512i cur_i = v_iota;

                for (int c = 0; c < 128; c += 16) {
                    __m512 dist = _mm512_mul_ps(vs02, _mm512_load_ps(g_cb_c20 + c));
                    dist = _mm512_fmadd_ps(vs12, _mm512_load_ps(g_cb_c21 + c), dist);
                    dist = _mm512_fmadd_ps(vs22, _mm512_load_ps(g_cb_c22 + c), dist);
                    dist = _mm512_fmadd_ps(vws0, _mm512_load_ps(g_cb_q0 + c), dist);
                    dist = _mm512_fmadd_ps(vws1, _mm512_load_ps(g_cb_q1 + c), dist);
                    dist = _mm512_fmadd_ps(vws2, _mm512_load_ps(g_cb_q2 + c), dist);
                    __mmask16 lt = _mm512_cmp_ps_mask(dist, best_d, _CMP_LT_OQ);
                    best_d = _mm512_mask_blend_ps(lt, best_d, dist);
                    best_i = _mm512_mask_blend_epi32(lt, best_i, cur_i);
                    cur_i  = _mm512_add_epi32(cur_i, v_step);
                }
                float min_val = _mm512_reduce_min_ps(best_d);
                __mmask16 eq = _mm512_cmp_ps_mask(best_d, _mm512_set1_ps(min_val), _CMP_EQ_OQ);
                int lane = __builtin_ctz((unsigned int)eq);
                int align_idx[16];
                _mm512_storeu_si512((__m512i *)align_idx, best_i);
                int best_c = align_idx[lane];
                idx22[j] = (uint8_t)best_c;
                w_rec[j]      = g_cb_q0[best_c];
                w_rec[j + 22] = g_cb_q1[best_c];
                w_rec[j + 44] = g_cb_q2[best_c];
            }

            int s7_u3 = ((int)roundf(m[7] - 2.0f)) & 0x7;
            for (int u = 0; u < 2; ++u) {
                int j = 20 + u;
                int bit_half = (s7_u3 >> (u + 1)) & 1;
                int c_start = bit_half ? 64 : 0;
                float w0 = xb[j],      s0 = eff_s[j];
                float w1 = xb[j + 22], s1 = eff_s[j + 22];
                __m512 vs02 = _mm512_set1_ps(s0 * s0);
                __m512 vs12 = _mm512_set1_ps(s1 * s1);
                __m512 vws0 = _mm512_mul_ps(v_neg2, _mm512_set1_ps(w0 * s0));
                __m512 vws1 = _mm512_mul_ps(v_neg2, _mm512_set1_ps(w1 * s1));
                __m512 best_d = _mm512_set1_ps(1e30f);
                __m512i best_i = _mm512_setzero_si512();
                __m512i cur_i = _mm512_add_epi32(v_iota, _mm512_set1_epi32(c_start));
                for (int c = c_start; c < c_start + 64; c += 16) {
                    __m512 dist = _mm512_mul_ps(vs02, _mm512_load_ps(g_cb_c20 + c));
                    dist = _mm512_fmadd_ps(vs12, _mm512_load_ps(g_cb_c21 + c), dist);
                    dist = _mm512_fmadd_ps(vws0, _mm512_load_ps(g_cb_q0 + c), dist);
                    dist = _mm512_fmadd_ps(vws1, _mm512_load_ps(g_cb_q1 + c), dist);
                    __mmask16 lt = _mm512_cmp_ps_mask(dist, best_d, _CMP_LT_OQ);
                    best_d = _mm512_mask_blend_ps(lt, best_d, dist);
                    best_i = _mm512_mask_blend_epi32(lt, best_i, cur_i);
                    cur_i  = _mm512_add_epi32(cur_i, v_step);
                }
                float min_val = _mm512_reduce_min_ps(best_d);
                __mmask16 eq = _mm512_cmp_ps_mask(best_d, _mm512_set1_ps(min_val), _CMP_EQ_OQ);
                int lane = __builtin_ctz((unsigned int)eq);
                int align_idx[16];
                _mm512_storeu_si512((__m512i *)align_idx, best_i);
                int best_c = align_idx[lane];
                idx22[j] = (uint8_t)best_c;
                w_rec[j]      = g_cb_q0[best_c];
                w_rec[j + 22] = g_cb_q1[best_c];
            }

            if (iter == 0) {
                for (int s = 0; s < 8; ++s) {
                    float wq = 0.0f, qq = 0.0f;
                    for (int j = 0; j < 8; ++j) {
                        float wv = xb[s * 8 + j];
                        float qv = w_rec[s * 8 + j];
                        wq += wv * qv;
                        qq += qv * qv;
                    }
                    if (qq > 0.0f) {
                        float mo = roundf(wq / fmaxf(d * qq, 1e-30f));
                        if (mo < 2.0f) mo = 2.0f;
                        if (mo > 9.0f) mo = 9.0f;
                        m[s] = mo;
                    }
                }
            }
        }

        uint32_t s24 = 0;
        for (int s = 0; s < 8; ++s) {
            uint32_t sv = ((uint32_t)roundf(m[s] - 2.0f)) & 0x7U;
            s24 |= (sv << (3 * s));
        }
        y[b].d = d_h;
        for (int k_i = 0; k_i < 22; ++k_i) {
            uint8_t msb = (uint8_t)((s24 >> k_i) & 1U);
            y[b].qs[k_i] = (uint8_t)((idx22[k_i] & 0x7FU) | (msb << 7));
        }
    }
}

static int64_t row_bytes_for_type(int ttype, int64_t ncol) {
    switch (ttype) {
        case 0:  return ncol * 4;
        case 2:  return (ncol / 32) * 18;
        case 8:  return (ncol / 32) * 34;
        case 12: return (ncol / 256) * 144;
        case 13: return (ncol / 256) * 176;
        case 14: return (ncol / 256) * 210;
        case 16: return (ncol / 256) * 66;
        case 19: return (ncol / 256) * 50;
        case 26: return ncol * 4;
        case 29: return (ncol / 256) * 56;
        case 30: return ncol * 2;
        case 42: return (ncol / 64) * 30;
        case 43: return (ncol / 64) * 24;
        default: return 0;
    }
}

static void decode_src_row(
    const void * src_ptr, int src_dtype, const uint8_t * scale_ptr,
    int64_t r, int64_t ncol, float * row_f32
) {
    // src_dtype: 0 = F32, 1 = BF16, 2 = F8_E4M3 (with block-128 F8_E8M0 scale_ptr)
    if (src_dtype == 0) {
        memcpy(row_f32, (const float *)src_ptr + r * ncol, (size_t)ncol * sizeof(float));
    } else if (src_dtype == 1) {
        const uint16_t * sr = (const uint16_t *)src_ptr + r * ncol;
        for (int64_t c = 0; c < ncol; ++c) row_f32[c] = bf16_to_fp32(sr[c]);
    } else if (src_dtype == 2) {
        const uint8_t * wr = (const uint8_t *)src_ptr + r * ncol;
        const int64_t scale_cols = (ncol + 127) / 128;
        const uint8_t * sr = scale_ptr + (r >> 7) * scale_cols;
        for (int64_t cb = 0; cb < scale_cols; ++cb) {
            const float s = g_e8m0_f32[sr[cb]];
            const int64_t c0 = cb * 128;
            const int64_t c1 = (c0 + 128 < ncol) ? (c0 + 128) : ncol;
            for (int64_t c = c0; c < c1; ++c) {
                row_f32[c] = g_e4m3_f32[wr[c]] * s;
            }
        }
    }
}

void sp_decode_2d_f32(
    const void * src_ptr, int src_dtype, const uint8_t * scale_ptr,
    float * dst_f32, int64_t nrow, int64_t ncol
) {
    sp_init();
    #pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < nrow; ++r) {
        decode_src_row(src_ptr, src_dtype, scale_ptr, r, ncol, dst_f32 + r * ncol);
    }
}

void sp_quantize_2d(
    const void * src_ptr, int src_dtype, const uint8_t * scale_ptr,
    void * dst_ptr, int dst_type, int64_t nrow, int64_t ncol
) {
    sp_init();
    const int64_t n_el = nrow * ncol;
    if (dst_type == 30 && src_dtype == 1) {
        memcpy(dst_ptr, src_ptr, (size_t)n_el * 2);
        return;
    }
    if (dst_type == 0 && src_dtype == 0) {
        memcpy(dst_ptr, src_ptr, (size_t)n_el * 4);
        return;
    }
    if (dst_type == 26) {
        // I64 (src_dtype == 3) or I32 (src_dtype == 4) -> I32
        int32_t * out = (int32_t *)dst_ptr;
        if (src_dtype == 3) {
            const int64_t * in64 = (const int64_t *)src_ptr;
            #pragma omp parallel for schedule(static)
            for (int64_t i = 0; i < n_el; ++i) out[i] = (int32_t)in64[i];
        } else {
            memcpy(out, src_ptr, (size_t)n_el * 4);
        }
        return;
    }
    if (dst_type == 0) {
        sp_decode_2d_f32(src_ptr, src_dtype, scale_ptr, (float *)dst_ptr, nrow, ncol);
        return;
    }

    const int64_t dst_rb = row_bytes_for_type(dst_type, ncol);
    uint8_t * dst_u8 = (uint8_t *)dst_ptr;

    #pragma omp parallel
    {
        float * row_f32 = (float *)malloc((size_t)ncol * sizeof(float));
        if (!row_f32) abort();
        #pragma omp for schedule(static)
        for (int64_t r = 0; r < nrow; ++r) {
            decode_src_row(src_ptr, src_dtype, scale_ptr, r, ncol, row_f32);
            void * drow = dst_u8 + r * dst_rb;
            if (dst_type == 2) {
                quantize_row_q4_0_ref(row_f32, drow, ncol);
            } else if (dst_type == 8) {
                quantize_row_q8_0_ref(row_f32, drow, ncol);
            } else if (dst_type == 12) {
                quantize_q4_K(row_f32, drow, 1, ncol, NULL);
            } else if (dst_type == 13) {
                quantize_q5_K(row_f32, drow, 1, ncol, NULL);
            } else if (dst_type == 14) {
                quantize_q6_K(row_f32, drow, 1, ncol, NULL);
            } else if (dst_type == 42) {
                quantize_row_q2_0_ref(row_f32, drow, ncol);
            } else if (dst_type == 43) {
                quantize_row_vq3_0_fast(row_f32, (block_vq3_0 *)drow, ncol);
            } else {
                fprintf(stderr, "FATAL: unsupported 2D dst_type=%d\n", dst_type);
                abort();
            }
        }
        free(row_f32);
    }
}

static void decode_mxfp4_expert_slice(
    const uint8_t * w_i8, const uint8_t * s_u8,
    float * out_f32, int64_t nrow, int64_t ncol
) {
    const int64_t n_blocks = ncol / 32;
    for (int64_t r = 0; r < nrow; ++r) {
        const uint8_t * sr = s_u8 + r * n_blocks;
        const uint8_t * wr = w_i8 + r * (n_blocks * 16);
        float * orow = out_f32 + r * ncol;
        for (int64_t b = 0; b < n_blocks; ++b) {
            const float s = g_e8m0_f32[sr[b]];
            const uint8_t * wb = wr + b * 16;
            float * ob = orow + b * 32;
            for (int j = 0; j < 16; ++j) {
                const uint8_t q = wb[j];
                ob[2 * j + 0] = g_fp4_f32[q & 0x0F] * s;
                ob[2 * j + 1] = g_fp4_f32[q >> 4]   * s;
            }
        }
    }
}

void sp_decode_mxfp4_expert_f32(
    const uint8_t * w_i8, const uint8_t * s_u8,
    float * out_f32, int64_t nrow, int64_t ncol
) {
    sp_init();
    decode_mxfp4_expert_slice(w_i8, s_u8, out_f32, nrow, ncol);
}

void sp_quantize_moe_3d(
    const uint8_t ** w_ptrs,
    const uint8_t ** s_ptrs,
    uint8_t * dst_ptr,
    int dst_type,
    int64_t nrow,
    int64_t ncol,
    int64_t n_exp,
    const float * imatrix_all
) {
    sp_init();
    const int64_t dst_eb = nrow * row_bytes_for_type(dst_type, ncol);
    const int64_t n_elem = nrow * ncol;

    #pragma omp parallel
    {
        float * f32 = (float *)malloc((size_t)n_elem * sizeof(float));
        if (!f32) abort();

        #pragma omp for schedule(dynamic, 1)
        for (int64_t e = 0; e < n_exp; ++e) {
            decode_mxfp4_expert_slice(w_ptrs[e], s_ptrs[e], f32, nrow, ncol);
            const float * qw_e = imatrix_all + e * ncol;
            uint8_t * dst_e = dst_ptr + e * dst_eb;
            if (dst_type == 16) {
                quantize_iq2_xxs(f32, dst_e, nrow, ncol, qw_e);
            } else if (dst_type == 19) {
                quantize_iq1_s(f32, dst_e, nrow, ncol, qw_e);
            } else if (dst_type == 29) {
                quantize_iq1_m(f32, dst_e, nrow, ncol, qw_e);
            } else {
                fprintf(stderr, "FATAL: unsupported MoE dst_type=%d\n", dst_type);
                abort();
            }
        }
        free(f32);
    }
}

void sp_dequantize_gguf_rows(
    const uint8_t * src_ptr, int ttype, float * dst_f32, int64_t nrow, int64_t ncol
) {
    sp_init();
    const int64_t rb = row_bytes_for_type(ttype, ncol);
    #pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < nrow; ++r) {
        const uint8_t * sr = src_ptr + r * rb;
        float * dr = dst_f32 + r * ncol;
        switch (ttype) {
            case 0:  memcpy(dr, sr, (size_t)ncol * 4); break;
            case 2:  dequantize_row_q4_0(sr, dr, ncol); break;
            case 8:  dequantize_row_q8_0(sr, dr, ncol); break;
            case 12: dequantize_row_q4_K(sr, dr, ncol); break;
            case 13: dequantize_row_q5_K(sr, dr, ncol); break;
            case 14: dequantize_row_q6_K(sr, dr, ncol); break;
            case 16: dequantize_row_iq2_xxs(sr, dr, ncol); break;
            case 19: dequantize_row_iq1_s(sr, dr, ncol); break;
            case 29: dequantize_row_iq1_m(sr, dr, ncol); break;
            case 30: {
                const uint16_t * b16 = (const uint16_t *)sr;
                for (int64_t c = 0; c < ncol; ++c) dr[c] = bf16_to_fp32(b16[c]);
                break;
            }
            case 42: dequantize_row_q2_0(sr, dr, ncol); break;
            case 43: dequantize_row_vq3_0(sr, dr, ncol); break;
            default: memset(dr, 0, (size_t)ncol * 4); break;
        }
    }
}
