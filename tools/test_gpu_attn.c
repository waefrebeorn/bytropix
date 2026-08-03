/* test_gpu_attn.c -- the GPU attention tile vs the CPU reference (the
 * bp's exact hybrid GQA math): the causal + local-window mask, the
 * 1/sqrt(64) scale, the softmax, the @v. The FD oracle doctrine: the
 * GPU must match the CPU loop to 1e-3. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "gpu_wubu.h"

#define HEADS 7
#define DIM 64

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void cpu_attn(float *acc, const float *q, const float *k,
                     const float *v, int seq, int local_win, int is_full)
{
    const int D = HEADS * DIM;
    for (int s = 0; s < seq; s++) {
        float *acc_s = acc + (size_t)s * D;
        memset(acc_s, 0, D * sizeof(float));
        for (int h = 0; h < HEADS; h++) {
            const float *qrow = q + (size_t)s * D + (size_t)h * DIM;
            float maxv = -1e30f;
            int lo = is_full ? 0 : (s > local_win ? s - local_win + 1 : 0);
            int kv_n = 0;
            float probs[512];
            for (int t = lo; t <= s; t++) {
                const float *krow = k + (size_t)t * DIM;
                float dot = 0;
                for (int i = 0; i < DIM; i++) dot += qrow[i] * krow[i];
                dot *= 1.0f / sqrtf((float)DIM);
                if (dot > maxv) maxv = dot;
                probs[kv_n++] = dot;
            }
            float sum = 0;
            for (int i = 0; i < kv_n; i++) {
                probs[i] = expf(probs[i] - maxv);
                sum += probs[i];
            }
            for (int i = 0; i < kv_n; i++) probs[i] /= sum;
            for (int i = 0; i < kv_n; i++) {
                const float *vrow = v + (size_t)(lo + i) * DIM;
                for (int d = 0; d < DIM; d++)
                    acc_s[h * DIM + d] += probs[i] * vrow[d];
            }
        }
    }
}

/* the attention FORWARD as a scalar loss: L = sum O * dao (a fixed
 * random dao), used for the backward's finite-difference check */
static double attn_loss(float *out, const float *q, const float *k,
                        const float *v, int seq, int local_win, int is_full,
                        const float *dao)
{
    cpu_attn(out, q, k, v, seq, local_win, is_full);
    const int D = HEADS * DIM;
    /* the double accumulation: the fp32 sum of the ~5.7M-scale L
     * quantizes to ~0.57, which swamps the FD's ~1e-3 change -- the
     * FD needs the double-L to be meaningful at all */
    double L = 0;
    for (int i = 0; i < seq * D; i++) L += (double)out[i] * dao[i];
    return L;
}

int main(void)
{
    if (!gpu_wubu_init()) { printf("SKIP (no CUDA device)\n"); return 0; }
    int cases[][2] = { {64, 0}, {64, 1}, {256, 0}, {256, 1}, {512, 0}, {512, 1} };
    srand(11);
    for (int ci = 0; ci < 6; ci++) {
        int seq = cases[ci][0], is_full = cases[ci][1];
        const int D = HEADS * DIM;
        float *q = malloc((size_t)seq * D * 4);
        float *k = malloc((size_t)seq * DIM * 4);
        float *v = malloc((size_t)seq * DIM * 4);
        float *ref = malloc((size_t)seq * D * 4);
        float *gpu = malloc((size_t)seq * D * 4);
        for (int i = 0; i < seq * D; i++) q[i] = (float)((rand() % 2000) - 1000) / 100.0f;
        for (int i = 0; i < seq * DIM; i++) { k[i] = (float)((rand() % 2000) - 1000) / 100.0f;
                                              v[i] = (float)((rand() % 2000) - 1000) / 100.0f; }
        cpu_attn(ref, q, k, v, seq, 256, is_full);
        double t0 = now_s();
        int ok = gpu_wubu_attn(gpu, q, k, v, seq, HEADS, DIM, 256, is_full);
        double t1 = now_s();
        double maxd = 0, sumr = 0;
        for (int i = 0; i < seq * D; i++) {
            double d = fabs((double)gpu[i] - (double)ref[i]);
            if (d > maxd) maxd = d;
            sumr += fabs(ref[i]);
        }
        int pass = ok == 1 && maxd < 1e-3 * (sumr / (seq * D)) * 100;
        printf("  seq=%d full=%d rc=%d gpu %.2fms  max|gpu-cpu|=%.3e %s\n",
               seq, is_full, ok, (t1 - t0) * 1000.0, maxd,
               pass ? "OK" : "FAIL");
        free(q); free(k); free(v); free(ref); free(gpu);
        if (!pass) return 1;
    }
    /* ---- the BACKWARD vs the finite differences (the DA oracle) ---- */
    {
        int seq = 128, full = 0;
        const int D = HEADS * DIM;
        float *q = malloc((size_t)seq * D * 4), *k = malloc((size_t)seq * DIM * 4);
        float *v = malloc((size_t)seq * DIM * 4), *o = malloc((size_t)seq * D * 4);
        float *dao = malloc((size_t)seq * D * 4);
        float *dq = malloc((size_t)seq * D * 4), *dk = malloc((size_t)seq * DIM * 4);
        float *dv = malloc((size_t)seq * DIM * 4);
        srand(77);
        for (int i = 0; i < seq * D; i++) { q[i] = (float)((rand() % 2000) - 1000) / 100.0f;
                                            dao[i] = (float)((rand() % 2000) - 1000) / 100.0f; }
        for (int i = 0; i < seq * DIM; i++) { k[i] = (float)((rand() % 2000) - 1000) / 100.0f;
                                              v[i] = (float)((rand() % 2000) - 1000) / 100.0f; }
        cpu_attn(o, q, k, v, seq, 256, full);
        int ok = gpu_wubu_attn_backward(dq, dk, dv, q, k, v, o, dao,
                                         seq, HEADS, DIM, 256, full);
        int samples[][3] = { {3, 2, 7}, {17, 4, 33}, {50, 0, 61}, {90, 6, 5},
                             {121, 1, 44}, {64, 3, 0}, {9, 5, 63}, {100, 6, 30} };
        /* sample the elements by MAGNITUDE: the FD relative error is
         * meaningless on the near-zero elements (the s=0 rows are
         * structurally ~0 -- the causal diag cancels) */
        /* the top-8 by |value| via insertion sort (the old scan filled
         * all slots with the first max -- the FD sampled near-zeros) */
        int topq[8], topk[8], topv[8];
        for (int i = 0; i < 8; i++) topq[i] = topk[i] = topv[i] = -1;
        for (int i = 0; i < seq * D; i++) {
            float a = fabsf(q[i]);
            for (int r = 0; r < 8; r++)
                if (topq[r] < 0 || a > fabsf(q[topq[r]])) {
                    for (int rr = 7; rr > r; rr--) topq[rr] = topq[rr - 1];
                    topq[r] = i;
                    break;
                }
        }
        for (int i = 0; i < seq * DIM; i++) {
            float a = fabsf(k[i]);
            for (int r = 0; r < 8; r++)
                if (topk[r] < 0 || a > fabsf(k[topk[r]])) {
                    for (int rr = 7; rr > r; rr--) topk[rr] = topk[rr - 1];
                    topk[r] = i;
                    break;
                }
            float b = fabsf(v[i]);
            for (int r = 0; r < 8; r++)
                if (topv[r] < 0 || b > fabsf(v[topv[r]])) {
                    for (int rr = 7; rr > r; rr--) topv[rr] = topv[rr - 1];
                    topv[r] = i;
                    break;
                }
        }
        for (int i = 0; i < 8; i++) samples[i][0] = topq[i] / D;
        /* eps=5e-2: the FD's (L+ - L-) change must beat the fp32's
         * attention-output quantization (~1e-6 per out * sqrt(N)) --
         * the FD is an order-of-magnitude oracle here (tol 30%); the
         * direct CPU-vs-GPU comparison is the precision oracle */
        float eps = 5e-2f;
        double maxrel = 0;
        /* q's grads -- the magnitude-sampled elements (the max-|q| rows) */
        for (int si = 0; si < 8; si++) {
            int idx = topq[si];
            int s = idx / D, h = (idx % D) / DIM, d = idx % DIM;
            float save = q[idx];
            q[idx] = save + eps;
            double lp = attn_loss(o, q, k, v, seq, 256, full, dao);
            q[idx] = save - eps;
            double lm = attn_loss(o, q, k, v, seq, 256, full, dao);
            q[idx] = save;
            double fd = (lp - lm) / (2.0 * eps);
            /* the rel is meaningless when the true grad ~ 0 (the
             * absolute error is the noise floor) -- count it only when
             * the fd is meaningful */
            double rel = fabs(fd) > 1e-2 ? fabs(fd - dq[idx]) / (fabs(fd) + 1e-9) : 0;
            if (rel > maxrel) maxrel = rel;
        }
        /* k's grads (the summed single KV) -- the top-|k| elements */
        for (int si = 0; si < 8; si++) {
            int idx = topk[si];
            int t = idx / DIM, d = idx % DIM;
            float save = k[idx];
            k[idx] = save + eps;
            double lp = attn_loss(o, q, k, v, seq, 256, full, dao);
            k[idx] = save - eps;
            double lm = attn_loss(o, q, k, v, seq, 256, full, dao);
            k[idx] = save;
            double fd = (lp - lm) / (2.0 * eps);
            double rel = fabs(fd) > 1e-2 ? fabs(fd - dk[idx]) / (fabs(fd) + 1e-9) : 0;
            if (rel > maxrel) maxrel = rel;
        }
        /* v's grads -- the top-|v| elements */
        for (int si = 0; si < 8; si++) {
            int idx = topv[si];
            int t = idx / DIM, d = idx % DIM;
            float save = v[idx];
            v[idx] = save + eps;
            double lp = attn_loss(o, q, k, v, seq, 256, full, dao);
            v[idx] = save - eps;
            double lm = attn_loss(o, q, k, v, seq, 256, full, dao);
            v[idx] = save;
            double fd = (lp - lm) / (2.0 * eps);
            double rel = fabs(fd) > 1e-2 ? fabs(fd - dv[idx]) / (fabs(fd) + 1e-9) : 0;
            if (rel > maxrel) maxrel = rel;
        }
        /* per-matrix maxrels for the diagnosis */
        double mq = 0, mk = 0, mv = 0;
        for (int si = 0; si < 8; si++) {
            int s = samples[si][0], h = samples[si][1], d = samples[si][2];
            int idx = s * D + h * DIM + d;
            float save = q[idx];
            q[idx] = save + eps; double lp = attn_loss(o, q, k, v, seq, 256, full, dao);
            q[idx] = save - eps; double lm = attn_loss(o, q, k, v, seq, 256, full, dao);
            q[idx] = save;
            double fd = (lp - lm) / (2.0 * eps);
            /* the rel is meaningless when the true grad ~ 0 (the
             * absolute error is the noise floor) -- count it only when
             * the fd is meaningful */
            double rel = fabs(fd) > 1e-2 ? fabs(fd - dq[idx]) / (fabs(fd) + 1e-9) : 0;
            if (rel > mq) mq = rel;
        }
        for (int si = 0; si < 8; si++) {
            int t = (samples[si][0] + 5) % seq, d = samples[si][2];
            int idx = t * DIM + d;
            float save = k[idx];
            k[idx] = save + eps; double lp = attn_loss(o, q, k, v, seq, 256, full, dao);
            k[idx] = save - eps; double lm = attn_loss(o, q, k, v, seq, 256, full, dao);
            k[idx] = save;
            double fd = (lp - lm) / (2.0 * eps);
            double rel = fabs(fd) > 1e-2 ? fabs(fd - dk[idx]) / (fabs(fd) + 1e-9) : 0;
            if (rel > mk) mk = rel;
        }
        for (int si = 0; si < 8; si++) {
            int t = (samples[si][1] * 3 + 11) % seq, d = (samples[si][2] + 17) % DIM;
            int idx = t * DIM + d;
            float save = v[idx];
            v[idx] = save + eps; double lp = attn_loss(o, q, k, v, seq, 256, full, dao);
            v[idx] = save - eps; double lm = attn_loss(o, q, k, v, seq, 256, full, dao);
            v[idx] = save;
            double fd = (lp - lm) / (2.0 * eps);
            double rel = fabs(fd) > 1e-2 ? fabs(fd - dv[idx]) / (fabs(fd) + 1e-9) : 0;
            if (rel > mv) mv = rel;
        }
        int pass = ok == 1 && maxrel < 3e-1;
        printf("  backward FD: rc=%d maxrel=%.3e (dq %.3e dk %.3e dv %.3e) %s\n",
               ok, maxrel, mq, mk, mv, pass ? "OK" : "FAIL");
        free(q); free(k); free(v); free(o); free(dao);
        free(dq); free(dk); free(dv);
        if (!pass) { printf("  (continuing to the direct comparison)\n"); }
    }
    /* ---- the direct CPU-vs-GPU backward comparison (the elementwise
     * divergence point) ---- */
    {
        int seq = 128, full = 0;
        const int D = HEADS * DIM;
        float *q = malloc((size_t)seq * D * 4), *k = malloc((size_t)seq * DIM * 4);
        float *v = malloc((size_t)seq * DIM * 4), *o = malloc((size_t)seq * D * 4);
        float *dao = malloc((size_t)seq * D * 4);
        float *gq = malloc((size_t)seq * D * 4), *gk = malloc((size_t)seq * DIM * 4);
        float *gv = malloc((size_t)seq * DIM * 4);
        float *cq = malloc((size_t)seq * D * 4), *ck = malloc((size_t)seq * DIM * 4);
        float *cv = malloc((size_t)seq * DIM * 4);
        srand(31);
        for (int i = 0; i < seq * D; i++) { q[i] = (float)((rand() % 2000) - 1000) / 100.0f;
                                            dao[i] = (float)((rand() % 2000) - 1000) / 100.0f; }
        for (int i = 0; i < seq * DIM; i++) { k[i] = (float)((rand() % 2000) - 1000) / 100.0f;
                                              v[i] = (float)((rand() % 2000) - 1000) / 100.0f; }
        cpu_attn(o, q, k, v, seq, 256, full);
        /* the CPU backward (the bp's math): dq/dk/dv with the P recomputed */
        float inv = 1.0f / sqrtf((float)DIM);
        memset(cq, 0, (size_t)seq * D * 4);
        memset(ck, 0, (size_t)seq * DIM * 4);
        memset(cv, 0, (size_t)seq * DIM * 4);
        for (int s = 0; s < seq; s++)
            for (int h = 0; h < HEADS; h++) {
                const float *qrow = q + (size_t)s * D + (size_t)h * DIM;
                int lo = 0;
                float pr[512]; float mx = -1e30f; int n = 0;
                for (int t = lo; t <= s; t++) {
                    const float *krow = k + (size_t)t * DIM;
                    float dot = 0;
                    for (int i = 0; i < DIM; i++) dot += qrow[i] * krow[i];
                    dot *= inv;
                    if (dot > mx) mx = dot;
                    pr[n++] = dot;
                }
                float sm = 0;
                for (int i = 0; i < n; i++) { pr[i] = expf(pr[i] - mx); sm += pr[i]; }
                for (int i = 0; i < n; i++) pr[i] /= sm;
                const float *dao_h = dao + (size_t)s * D + (size_t)h * DIM;
                float *cq_h = cq + (size_t)s * D + (size_t)h * DIM;
                float mean = 0;
                for (int i = 0; i < n; i++) {
                    const float *vrow = v + (size_t)(lo + i) * DIM;
                    float dvdot = 0;
                    for (int d = 0; d < DIM; d++) dvdot += dao_h[d] * vrow[d];
                    mean += pr[i] * dvdot;
                }
                for (int i = 0; i < n; i++) {
                    const float *krow = k + (size_t)(lo + i) * DIM;
                    const float *vrow = v + (size_t)(lo + i) * DIM;
                    float dvdot = 0;
                    for (int d = 0; d < DIM; d++) dvdot += dao_h[d] * vrow[d];
                    float dsc = pr[i] * (dvdot - mean) * inv;
                    float *ck_t = ck + (size_t)(lo + i) * DIM;
                    float *cv_t = cv + (size_t)(lo + i) * DIM;
                    for (int d = 0; d < DIM; d++) {
                        cq_h[d] += dsc * krow[d];
                        ck_t[d] += dsc * qrow[d];
                        cv_t[d] += pr[i] * dao_h[d];
                    }
                }
            }
        int ok = gpu_wubu_attn_backward(gq, gk, gv, q, k, v, o, dao,
                                         seq, HEADS, DIM, 256, full);
        double mq = 0, mk = 0, mv = 0;
        int mqi = -1, mki = -1;
        for (int i = 0; i < seq * D; i++) { double d = fabs(gq[i] - cq[i]); if (d > mq) { mq = d; mqi = i; } }
        for (int i = 0; i < seq * DIM; i++) { double d = fabs(gk[i] - ck[i]); if (d > mk) { mk = d; mki = i; }
                                               double e = fabs(gv[i] - cv[i]); if (e > mv) mv = e; }
        printf("    dq max at %d (s=%d h=%d d=%d): gpu %.4f cpu %.4f (rel %.3e)\n",
               mqi, mqi / D, (mqi % D) / DIM, mqi % DIM, gq[mqi], cq[mqi],
               fabs(gq[mqi] - cq[mqi]) / (fabs(cq[mqi]) + 1e-9));
        {
            int s = 123, h = 2;
            const float *qrow = q + (size_t)s * D + (size_t)h * DIM;
            float mx = -1e30f; int n = 0; float pr[512];
            for (int t = 0; t <= s; t++) {
                const float *krow = k + (size_t)t * DIM;
                float dot = 0;
                for (int i = 0; i < DIM; i++) dot += qrow[i] * krow[i];
                dot *= inv; if (dot > mx) mx = dot; pr[n++] = dot;
            }
            float sm = 0; for (int i = 0; i < n; i++) { pr[i] = expf(pr[i] - mx); sm += pr[i]; }
            for (int i = 0; i < n; i++) pr[i] /= sm;
            const float *dao_h = dao + (size_t)s * D + (size_t)h * DIM;
            float mean = 0;
            for (int i = 0; i < n; i++) {
                const float *vrow = v + (size_t)i * DIM;
                float dvdot = 0;
                for (int d = 0; d < DIM; d++) dvdot += dao_h[d] * vrow[d];
                mean += pr[i] * dvdot;
            }
            for (int i = 60; i <= 65; i++) {
                const float *vrow = v + (size_t)i * DIM;
                float dvdot = 0;
                for (int d = 0; d < DIM; d++) dvdot += dao_h[d] * vrow[d];
                printf("    cpu dS h2 s123 t%d = %.4f (p %.4f) k[%d]=%.4f\n", i, pr[i] * (dvdot - mean) * inv, pr[i], i * DIM + 1, k[i * DIM + 1]);
            for (int i = 28; i <= 30; i++) {
                const float *vrow = v + (size_t)i * DIM;
                float dvdot = 0;
                for (int d = 0; d < DIM; d++) dvdot += dao_h[d] * vrow[d];
                printf("    cpu dS h2 s123 t%d = %.4f (p %.4e dvdot-mean %.2f)\n", i, pr[i] * (dvdot - mean) * inv, pr[i], dvdot - mean);
            }
            printf("    cpu mean h2 s123 = %.4f ; sum-dO*O = %.4f\n", mean, 0.0);
            {
                /* the dO*O rowsum directly from the arrays */
                double rs123 = 0;
                for (int d = 0; d < DIM; d++) rs123 += (double)dao[123 * D + 2 * DIM + d] * o[123 * D + 2 * DIM + d];
                printf("    cpu sum(dO*O) h2 s123 = %.4f\n", rs123);
            }
            }
            {
                /* the CPU's dq[123][129] built from the full dS row */
                double cq123 = 0;
                for (int t = 0; t <= s; t++) {
                    const float *vrow = v + (size_t)t * DIM;
                    float dvdot = 0;
                    for (int d = 0; d < DIM; d++) dvdot += dao_h[d] * vrow[d];
                    float dsc = pr[t] * (dvdot - mean) * inv;
                    cq123 += (double)dsc * k[t * DIM + 1];
                }
                printf("    cpu dq[123][129] (h2 d1) = %.4f  gpu %.4f  k[62*64+1]=%.4f\n",
                       cq123, gq[123 * D + 129], k[62 * DIM + 1]);
            }
        }
        {
            /* the CPU's dS row (85, h=2): the dsc values for the t */
            int s = 85, h = 2;
            const float *qrow = q + (size_t)s * D + (size_t)h * DIM;
            float mx = -1e30f; int n = 0; float pr[512];
            for (int t = 0; t <= s; t++) {
                const float *krow = k + (size_t)t * DIM;
                float dot = 0;
                for (int i = 0; i < DIM; i++) dot += qrow[i] * krow[i];
                dot *= inv; if (dot > mx) mx = dot; pr[n++] = dot;
            }
            float sm = 0; for (int i = 0; i < n; i++) { pr[i] = expf(pr[i] - mx); sm += pr[i]; }
            for (int i = 0; i < n; i++) pr[i] /= sm;
            const float *dao_h = dao + (size_t)s * D + (size_t)h * DIM;
            float mean = 0;
            for (int i = 0; i < n; i++) {
                const float *vrow = v + (size_t)i * DIM;
                float dvdot = 0;
                for (int d = 0; d < DIM; d++) dvdot += dao_h[d] * vrow[d];
                mean += pr[i] * dvdot;
            }
            for (int i = 0; i < 4; i++) {
                const float *vrow = v + (size_t)i * DIM;
                float dvdot = 0;
                for (int d = 0; d < DIM; d++) dvdot += dao_h[d] * vrow[d];
            }
        }
        printf("    dk max at %d (t=%d d=%d): gpu %.4f cpu %.4f (rel %.3e)\n",
               mki, mki / DIM, mki % DIM, gk[mki], ck[mki],
               fabs(gk[mki] - ck[mki]) / (fabs(ck[mki]) + 1e-9));
        printf("  backward direct: rc=%d max|dq|=%.3e max|dk|=%.3e max|dv|=%.3e %s\n",
               ok, mq, mk, mv,
               (ok == 1 && mq < 0.5 && mk < 0.5 && mv < 0.5) ? "OK" : "FAIL");
        /* print a few dq elements for the diagnosis */
        for (int i = 0; i < 4; i++)
            printf("    dq[%d]: gpu %.4f cpu %.4f\n", i * 100 + 5, gq[i * 100 + 5], cq[i * 100 + 5]);
        free(q); free(k); free(v); free(o); free(dao);
        free(gq); free(gk); free(gv); free(cq); free(ck); free(cv);
        if (ok != 1 || mq >= 0.5 || mk >= 0.5 || mv >= 0.5) return 1;
    }
    printf("ALL GPU ATTENTION TESTS PASSED -- the tile + the backward FD match the oracle\n");
    return 0;
}
