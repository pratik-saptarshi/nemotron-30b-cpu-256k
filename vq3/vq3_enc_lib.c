/*
 * Self-contained OpenMP AVX-512 LUT-VQ3-64 (GGML_TYPE_VQ3_0 = 43) encoder & transcoder shared library.
 * Build:
 *   gcc -O3 -march=native -mavx512f -mf16c -fopenmp -shared -fPIC vq3/vq3_enc_lib.c -lm -o vq3/libvq3_enc.so
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <immintrin.h>
#include "vq3_codebook.h"

#define QK_VQ3_0 64
typedef uint16_t ggml_half;
typedef struct { ggml_half d; uint8_t qs[22]; } block_vq3_0;
typedef struct {
    ggml_half d;
    uint8_t scales[4];
    uint8_t qs[16];
    uint8_t qh[8];
} block_q2_0;

static inline float fp16_to_fp32(ggml_half h) { return _cvtsh_ss(h); }
static inline ggml_half fp32_to_fp16(float f) { return _cvtss_sh(f, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC); }
static inline float bf16_to_fp32(uint16_t b) {
    uint32_t u = ((uint32_t)b) << 16;
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

static float g_cb_q0[128] __attribute__((aligned(64)));
static float g_cb_q1[128] __attribute__((aligned(64)));
static float g_cb_q2[128] __attribute__((aligned(64)));
static float g_cb_c20[128] __attribute__((aligned(64)));
static float g_cb_c21[128] __attribute__((aligned(64)));
static float g_cb_c22[128] __attribute__((aligned(64)));
static int g_cb_init = 0;

static void init_vq3_enc_tables(void) {
    if (g_cb_init) return;
    for (int c = 0; c < 128; ++c) {
        float v0 = (float)ggml_vq3_cb[c][0] * GGML_VQ3_G;
        float v1 = (float)ggml_vq3_cb[c][1] * GGML_VQ3_G;
        float v2 = (float)ggml_vq3_cb[c][2] * GGML_VQ3_G;
        g_cb_q0[c] = v0; g_cb_c20[c] = v0 * v0;
        g_cb_q1[c] = v1; g_cb_c21[c] = v1 * v1;
        g_cb_q2[c] = v2; g_cb_c22[c] = v2 * v2;
    }
    g_cb_init = 1;
}

void quantize_row_vq3_0_fast(const float * restrict x, block_vq3_0 * restrict y, int64_t k) {
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

void quantize_bf16_to_vq3_0_omp(const uint16_t * src_bf16, void * dst_vq3, int64_t nrow, int64_t ncol) {
    init_vq3_enc_tables();
    block_vq3_0 * y = (block_vq3_0 *)dst_vq3;
    const int64_t nblk_row = ncol / 64;
    #pragma omp parallel
    {
        float * row_f32 = (float *)malloc((size_t)ncol * sizeof(float));
        if (!row_f32) { fprintf(stderr, "OOM in transcode_any_to_vq3_0_omp\n"); abort(); }
        #pragma omp for schedule(static)
        for (int64_t r = 0; r < nrow; ++r) {
            const uint16_t * sr = src_bf16 + r * ncol;
            for (int64_t c = 0; c < ncol; ++c) row_f32[c] = bf16_to_fp32(sr[c]);
            quantize_row_vq3_0_fast(row_f32, y + r * nblk_row, ncol);
        }
        free(row_f32);
    }
}

void dequantize_row_q2_0_local(const block_q2_0 * restrict x, float * restrict y, int64_t k) {
    static const int8_t kvalues_fractal3[8] = { -14, -8, -4, -1, 1, 4, 8, 14 };
    const int64_t nb = k / 64;
    for (int64_t i = 0; i < nb; i++) {
        const float d = fp16_to_fp32(x[i].d);
        float eff_s[8];
        for (int b = 0; b < 4; ++b) {
            eff_s[b]     = d * (float)(4 + (x[i].scales[b] & 0x0F));
            eff_s[b + 4] = d * (float)(4 + ((x[i].scales[b] >> 4) & 0x0F));
        }
        for (int g = 0; g < 4; ++g) {
            for (int k_idx = 0; k_idx < 16; ++k_idx) {
                const int j = g * 16 + k_idx;
                const uint8_t lo2 = (x[i].qs[k_idx] >> (2 * g)) & 0x03;
                const uint8_t hi1 = (x[i].qh[k_idx & 7] >> (2 * g + (k_idx >> 3))) & 0x01;
                const uint8_t idx = (uint8_t)(lo2 | (hi1 << 2));
                y[i * 64 + j] = eff_s[j >> 3] * (float)kvalues_fractal3[idx];
            }
        }
    }
}

void transcode_q2_0_to_vq3_0_omp(const void * src_q2_0, void * dst_vq3, int64_t nrow, int64_t ncol) {
    init_vq3_enc_tables();
    const block_q2_0 * x = (const block_q2_0 *)src_q2_0;
    block_vq3_0 * y = (block_vq3_0 *)dst_vq3;
    const int64_t nblk_row = ncol / 64;
    #pragma omp parallel
    {
        float * row_f32 = (float *)malloc((size_t)ncol * sizeof(float));
        if (!row_f32) { fprintf(stderr, "OOM in transcode_q2_0_to_vq3_0_omp\n"); abort(); }
        #pragma omp for schedule(static)
        for (int64_t r = 0; r < nrow; ++r) {
            dequantize_row_q2_0_local(x + r * nblk_row, row_f32, ncol);
            quantize_row_vq3_0_fast(row_f32, y + r * nblk_row, ncol);
        }
        free(row_f32);
    }
}

typedef struct {
    ggml_half d;
    ggml_half dmin;
    uint8_t scales[12];
    uint8_t qh[32];
    uint8_t qs[128];
} block_q5_K_local;

typedef struct {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t  scales[16];
    ggml_half d;
} block_q6_K_local;

static inline void get_scale_min_k4_local(int j, const uint8_t * restrict q, uint8_t * restrict d, uint8_t * restrict m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >>  4) | ((q[j - 0] >> 6) << 4);
    }
}

static void dequantize_row_q5_K_local(const block_q5_K_local * restrict x, float * restrict y, int64_t k) {
    const int64_t nb = k / 256;
    for (int64_t i = 0; i < nb; i++) {
        const uint8_t * ql = x[i].qs;
        const uint8_t * qh = x[i].qh;
        const float d = fp16_to_fp32(x[i].d);
        const float min = fp16_to_fp32(x[i].dmin);
        int is = 0;
        uint8_t sc, m;
        uint8_t u1 = 1, u2 = 2;
        for (int j = 0; j < 256; j += 64) {
            get_scale_min_k4_local(is + 0, x[i].scales, &sc, &m);
            const float d1 = d * sc;
            const float m1 = min * m;
            get_scale_min_k4_local(is + 1, x[i].scales, &sc, &m);
            const float d2 = d * sc;
            const float m2 = min * m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * ((ql[l]  >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
            ql += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
}

static void dequantize_row_q6_K_local(const block_q6_K_local * restrict x, float * restrict y, int64_t k) {
    const int64_t nb = k / 256;
    for (int64_t i = 0; i < nb; i++) {
        const float d = fp16_to_fp32(x[i].d);
        const uint8_t * restrict ql = x[i].ql;
        const uint8_t * restrict qh = x[i].qh;
        const int8_t  * restrict sc = x[i].scales;
        for (int n = 0; n < 256; n += 128) {
            for (int l = 0; l < 32; ++l) {
                int is = l / 16;
                const int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql[l +  0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql[l + 32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l +  0] = d * sc[is + 0] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }
            y  += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

void transcode_any_to_vq3_0_omp(const void * src, int orig_type, void * dst_vq3, int64_t nrow, int64_t ncol) {
    init_vq3_enc_tables();
    const uint8_t * x = (const uint8_t *)src;
    block_vq3_0 * y = (block_vq3_0 *)dst_vq3;
    const int64_t nblk_out = ncol / 64;
    size_t row_bytes_in = 0;
    if (orig_type == 42) row_bytes_in = (size_t)(ncol / 64) * 30;
    else if (orig_type == 13) row_bytes_in = (size_t)(ncol / 256) * 176;
    else if (orig_type == 14) row_bytes_in = (size_t)(ncol / 256) * 210;
    else { fprintf(stderr, "Unsupported orig_type=%d\n", orig_type); abort(); }

    #pragma omp parallel
    {
        float * row_f32 = (float *)malloc((size_t)ncol * sizeof(float));
        if (!row_f32) { fprintf(stderr, "OOM in transcode_any_to_vq3_0_omp\n"); abort(); }
        #pragma omp for schedule(static)
        for (int64_t r = 0; r < nrow; ++r) {
            const void * sr = x + (size_t)r * row_bytes_in;
            if (orig_type == 42) dequantize_row_q2_0_local((const block_q2_0 *)sr, row_f32, ncol);
            else if (orig_type == 13) dequantize_row_q5_K_local((const block_q5_K_local *)sr, row_f32, ncol);
            else if (orig_type == 14) dequantize_row_q6_K_local((const block_q6_K_local *)sr, row_f32, ncol);
            quantize_row_vq3_0_fast(row_f32, y + r * nblk_out, ncol);
        }
        free(row_f32);
    }
}
