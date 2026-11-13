/*
 * Standalone verification test for LUT-VQ3-64 (GGML_TYPE_VQ3_0 = 43) AVX-512 VBMI kernels.
 * Build & run:
 *   gcc -O3 -march=native -mavx512f -mavx512bw -mavx512vbmi -mf16c -mbmi2 vq3/test_vq3_ggml_kernels.c -lm -o vq3/test_vq3_ggml_kernels && ./vq3/test_vq3_ggml_kernels
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>
#include "vq3_codebook.h"

#define QK_VQ3_0 64
typedef uint16_t ggml_half;

typedef struct {
    ggml_half d;
    uint8_t qs[22];
} block_vq3_0;

typedef struct {
    ggml_half d;
    int8_t qs[32];
} block_q8_0;

static inline float fp16_to_fp32(ggml_half h) {
    return _cvtsh_ss(h);
}

static inline ggml_half fp32_to_fp16(float f) {
    return _cvtss_sh(f, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}

void quantize_row_vq3_0_ref(const float * restrict x, block_vq3_0 * restrict y, int64_t k) {
    const int64_t nb = k / QK_VQ3_0;
    float cb_q[128][3];
    float cb_c2[128][3];
    for (int c = 0; c < 128; ++c) {
        for (int j = 0; j < 3; ++j) {
            float v = (float)ggml_vq3_cb[c][j] * GGML_VQ3_G;
            cb_q[c][j] = v;
            cb_c2[c][j] = v * v;
        }
    }

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

        for (int iter = 0; iter < 3; ++iter) {
            float eff_s[64];
            for (int s = 0; s < 8; ++s) {
                float es = d * m[s];
                for (int j = 0; j < 8; ++j) eff_s[s * 8 + j] = es;
            }

            // Slots 0..19 (full 128-entry 3D search)
            for (int j = 0; j < 20; ++j) {
                float w0 = xb[j],      s0 = eff_s[j],      s02 = s0 * s0, ws0 = w0 * s0;
                float w1 = xb[j + 22], s1 = eff_s[j + 22], s12 = s1 * s1, ws1 = w1 * s1;
                float w2 = xb[j + 44], s2 = eff_s[j + 44], s22 = s2 * s2, ws2 = w2 * s2;
                float best_d = 1e30f;
                int best_c = 0;
                for (int c = 0; c < 128; ++c) {
                    float dist = (s02 * cb_c2[c][0] + s12 * cb_c2[c][1] + s22 * cb_c2[c][2])
                               - 2.0f * (ws0 * cb_q[c][0] + ws1 * cb_q[c][1] + ws2 * cb_q[c][2]);
                    if (dist < best_d) { best_d = dist; best_c = c; }
                }
                idx22[j] = (uint8_t)best_c;
                w_rec[j]      = cb_q[best_c][0];
                w_rec[j + 22] = cb_q[best_c][1];
                w_rec[j + 44] = cb_q[best_c][2];
            }

            // Constrained slots 20..21 under current m[7]
            int s7_u3 = ((int)roundf(m[7] - 2.0f)) & 0x7;
            for (int u = 0; u < 2; ++u) {
                int j = 20 + u;
                int bit_half = (s7_u3 >> (u + 1)) & 1;
                int c_start = bit_half ? 64 : 0;
                int c_end   = bit_half ? 128 : 64;
                float w0 = xb[j],      s0 = eff_s[j],      s02 = s0 * s0, ws0 = w0 * s0;
                float w1 = xb[j + 22], s1 = eff_s[j + 22], s12 = s1 * s1, ws1 = w1 * s1;
                float best_d = 1e30f;
                int best_c = c_start;
                for (int c = c_start; c < c_end; ++c) {
                    float dist = (s02 * cb_c2[c][0] + s12 * cb_c2[c][1])
                               - 2.0f * (ws0 * cb_q[c][0] + ws1 * cb_q[c][1]);
                    if (dist < best_d) { best_d = dist; best_c = c; }
                }
                idx22[j] = (uint8_t)best_c;
                w_rec[j]      = cb_q[best_c][0];
                w_rec[j + 22] = cb_q[best_c][1];
            }

            if (iter < 2) {
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

void dequantize_row_vq3_0(const block_vq3_0 * restrict x, float * restrict y, int64_t k) {
    const int64_t nb = k / QK_VQ3_0;
    for (int64_t b = 0; b < nb; ++b) {
        const float d = fp16_to_fp32(x[b].d) * GGML_VQ3_G;
        uint32_t s24 = 0;
        for (int j = 0; j < 22; ++j) {
            if (x[b].qs[j] & 0x80U) s24 |= (1U << j);
        }
        s24 |= (((uint32_t)(x[b].qs[20] >> 6) & 1U) << 22);
        s24 |= (((uint32_t)(x[b].qs[21] >> 6) & 1U) << 23);

        float eff_s[64];
        for (int s = 0; s < 8; ++s) {
            float ms = (float)(((s24 >> (3 * s)) & 0x7U) + 2U);
            for (int j = 0; j < 8; ++j) eff_s[s * 8 + j] = d * ms;
        }
        float * yb = y + b * 64;
        for (int j = 0; j < 20; ++j) {
            int c = x[b].qs[j] & 0x7F;
            yb[j]      = (float)ggml_vq3_cb[c][0] * eff_s[j];
            yb[j + 22] = (float)ggml_vq3_cb[c][1] * eff_s[j + 22];
            yb[j + 44] = (float)ggml_vq3_cb[c][2] * eff_s[j + 44];
        }
        for (int j = 20; j < 22; ++j) {
            int c = x[b].qs[j] & 0x7F;
            yb[j]      = (float)ggml_vq3_cb[c][0] * eff_s[j];
            yb[j + 22] = (float)ggml_vq3_cb[c][1] * eff_s[j + 22];
        }
    }
}

#define GGML_VQ3_0_VARS \
    const __m512i zero512      = _mm512_setzero_si512(); \
    const __m512i lut0_lo      = ggml_vq3_loadu_si512(ggml_vq3_lut0_lo); \
    const __m512i lut0_hi      = ggml_vq3_loadu_si512(ggml_vq3_lut0_hi); \
    const __m512i lut1_lo      = ggml_vq3_loadu_si512(ggml_vq3_lut1_lo); \
    const __m512i lut1_hi      = ggml_vq3_loadu_si512(ggml_vq3_lut1_hi); \
    const __m512i lut2_lo      = ggml_vq3_loadu_si512(ggml_vq3_lut2_lo); \
    const __m512i lut2_hi      = ggml_vq3_loadu_si512(ggml_vq3_lut2_hi); \
    const __m512i perm_rep64   = ggml_vq3_loadu_si512(ggml_vq3_perm_rep64); \
    const __m512i shuf_mult512 = ggml_vq3_loadu_si512(ggml_vq3_shuf_mult512);

#define GGML_VQ3_0_UNPACK_STEP(i) \
    const float d0 = fp16_to_fp32(x[i].d) * GGML_VQ3_G; \
    const __m512i raw_22 = _mm512_maskz_loadu_epi8(0x00000000003FFFFFULL, x[i].qs); \
    const uint32_t s24 = (uint32_t)_mm512_movepi8_mask(raw_22) | \
                         (((uint32_t)(x[i].qs[20] >> 6) & 1U) << 22) | \
                         (((uint32_t)(x[i].qs[21] >> 6) & 1U) << 23); \
    const uint64_t m_all = ggml_vq3_unpack_scales(s24); \
    const __m512i vmult  = _mm512_shuffle_epi8(_mm512_set1_epi64((int64_t)m_all), shuf_mult512); \
    __m512i t_idx        = _mm512_permutexvar_epi8(perm_rep64, raw_22); \
    t_idx                = _mm512_mask2_permutex2var_epi8(lut1_lo, t_idx, 0x00000FFFFFC00000ULL, lut1_hi); \
    t_idx                = _mm512_mask2_permutex2var_epi8(lut2_lo, t_idx, 0xFFFFF00000000000ULL, lut2_hi); \
    const __m512i q_s8   = _mm512_mask2_permutex2var_epi8(lut0_lo, t_idx, 0x00000000003FFFFFULL, lut0_hi); \
    const __m512i q_abs  = _mm512_abs_epi8(q_s8); \
    const __mmask64 neg_mask = _mm512_movepi8_mask(q_s8);

#define GGML_VQ3_0_DOT_COL(Y_PTR, ACC)     do {         const __m512 vd = _mm512_mask_broadcastss_ps(             _mm512_set1_ps(d0 * fp16_to_fp32((Y_PTR)[2 * i + 0].d)), 0xFF00,             _mm_set_ss(d0 * fp16_to_fp32((Y_PTR)[2 * i + 1].d)));         const __m512i q8_all = _mm512_inserti64x4(             _mm512_castsi256_si512(_mm256_loadu_si256((const __m256i *)(Y_PTR)[2 * i + 0].qs)),             _mm256_loadu_si256((const __m256i *)(Y_PTR)[2 * i + 1].qs), 1);         const __m512i q8_signed = _mm512_mask_sub_epi8(q8_all, neg_mask, zero512, q8_all);         const __m512i p = _mm512_maddubs_epi16(q_abs, q8_signed);         const __m512i s_i32 = _mm512_madd_epi16(p, vmult);         (ACC) = _mm512_fmadd_ps(vd, _mm512_cvtepi32_ps(s_i32), (ACC));     } while (0)

void vec_dot_vq3_0_q8_0(int n, float * restrict s, const block_vq3_0 * restrict x, const block_q8_0 * restrict y) {
    const int nb = n / QK_VQ3_0;
    GGML_VQ3_0_VARS;
    __m512 acc0 = _mm512_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        GGML_VQ3_0_UNPACK_STEP(i);
        GGML_VQ3_0_DOT_COL(y, acc0);
    }
    *s = _mm512_reduce_add_ps(acc0);
}

static uint64_t unpack_scales_portable_ref(uint32_t s24) {
    uint64_t m_all = 0;
    for (int b = 0; b < 8; ++b) {
        uint64_t m = 2ULL + ((uint64_t)((s24 >> (3 * b)) & 0x7U));
        m_all |= (m << (8 * b));
    }
    return m_all;
}

int main(void) {
    const int N = 4096;
    float * w = malloc(N * sizeof(float));
    float * w_rec = malloc(N * sizeof(float));
    block_vq3_0 * b_vq3 = malloc((N / 64) * sizeof(block_vq3_0));
    block_q8_0  * b_q8  = malloc((N / 32) * sizeof(block_q8_0));
    float * y_rec = malloc(N * sizeof(float));
    if (!w || !w_rec || !b_vq3 || !b_q8 || !y_rec) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }

    /* Verify ggml_vq3_unpack_scales matches exact byte-by-byte reference on all 3-bit sub-scale values */
    for (uint32_t p = 0; p < 65536U; ++p) {
        uint32_t s24 = (p * 2654435761U) & 0xFFFFFFU;
        if (ggml_vq3_unpack_scales(s24) != unpack_scales_portable_ref(s24)) {
            fprintf(stderr, "Scale unpack mismatch on s24=0x%06x\n", s24);
            return 1;
        }
    }

    for (int i = 0; i < N; ++i) {
        w[i] = sinf(i * 0.137f) * cosf(i * 0.031f) * 0.02f;
    }
    quantize_row_vq3_0_ref(w, b_vq3, N);
    dequantize_row_vq3_0(b_vq3, w_rec, N);

    double sig = 0.0, err = 0.0, sig_rec = 0.0, sig_y = 0.0;
    for (int i = 0; i < N; ++i) {
        sig += (double)w[i] * w[i];
        sig_rec += (double)w_rec[i] * w_rec[i];
        double d = (double)w[i] - w_rec[i];
        err += d * d;
    }
    printf("SQNR on test signal: %.2f dB\n", 10.0 * log10(sig / err));

    for (int b = 0; b < N / 32; ++b) {
        b_q8[b].d = fp32_to_fp16(0.015f);
        float d8 = fp16_to_fp32(b_q8[b].d);
        for (int j = 0; j < 32; ++j) {
            int8_t q = (int8_t)((((b * 32 + j) * 19) % 251) - 125);
            b_q8[b].qs[j] = q;
            y_rec[b * 32 + j] = (float)q * d8;
            sig_y += (double)y_rec[b * 32 + j] * y_rec[b * 32 + j];
        }
    }

    float dot_avx512 = 0.0f;
    vec_dot_vq3_0_q8_0(N, &dot_avx512, b_vq3, b_q8);

    double dot_ref = 0.0;
    for (int i = 0; i < N; ++i) dot_ref += (double)w_rec[i] * y_rec[i];

    /* Normalize by Cauchy-Schwarz L2 norm bound ||w_rec||*||y_rec|| to avoid near-zero cancellation bias */
    double norm_denom = fmax(sqrt(sig_rec * sig_y), 1e-12);
    double norm_err = fabs((double)dot_avx512 - dot_ref) / norm_denom;

    /* Also verify strict relative error < 1e-6 on a correlated (non-cancelling) activation vector */
    for (int b = 0; b < N / 32; ++b) {
        b_q8[b].d = fp32_to_fp16(0.02f / 127.0f);
        float d8 = fp16_to_fp32(b_q8[b].d);
        for (int j = 0; j < 32; ++j) {
            int idx = b * 32 + j;
            int iv = (int)lrintf(w[idx] / (0.02f / 127.0f));
            if (iv < -127) iv = -127;
            if (iv > 127) iv = 127;
            b_q8[b].qs[j] = (int8_t)iv;
            y_rec[idx] = (float)iv * d8;
        }
    }
    float dot_corr_avx = 0.0f;
    vec_dot_vq3_0_q8_0(N, &dot_corr_avx, b_vq3, b_q8);
    double dot_corr_ref = 0.0;
    for (int i = 0; i < N; ++i) dot_corr_ref += (double)w_rec[i] * y_rec[i];
    double rel_err = fabs((double)dot_corr_avx - dot_corr_ref) / fmax(fabs(dot_corr_ref), 1e-12);

    printf("dot_avx512 = %.8f | dot_ref = %.8f | norm_err = %.3e | corr_rel_err = %.3e\n",
           dot_avx512, dot_ref, norm_err, rel_err);

    free(w);
    free(w_rec);
    free(b_vq3);
    free(b_q8);
    free(y_rec);
    return (norm_err < 1e-7 && rel_err < 1e-6) ? 0 : 1;
}
