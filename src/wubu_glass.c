/*
 * wubu_glass.c -- 64-D signature. Layout (each block is 8 values):
 *   [ 0.. 8) bitplane occupation: fraction of pixels with bit k set
 *   [ 8..16) gradient orientation histogram (8 bins, normalised)
 *   [16..24) intensity moments: mean, std, skew, kurtosis, p10, p50, p90, entropy
 *   [24..32) 2D Haar detail ratios at three scales (+ 5 zeros)
 *   [32..40) RESERVED: worldtube causality (no 3D pipeline yet)
 *   [40..64) 4x6 spatial density grid (mean/255 per cell)
 */
#include "wubu_glass.h"
#include <math.h>
#include <string.h>

static float clamp01(double v)
{
    if (v < 0.0) return 0.0f;
    if (v > 1.0) return 1.0f;
    return (float)v;
}

int wubu_glassbox64(const wubu_img *im, float out[WUBU_GLASS_DIMS])
{
    if (!im || !im->data || !out) return -1;
    memset(out, 0, WUBU_GLASS_DIMS * sizeof *out);

    int w = im->w, h = im->h;
    double n = (double)w * (double)h;

    /* [0..8) bitplane occupation. */
    double bits[8] = {0};
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            unsigned v = im->data[(size_t)y * im->stride + x];
            for (int k = 0; k < 8; k++) if (v & (1u << k)) bits[k]++;
        }
    for (int k = 0; k < 8; k++) out[k] = clamp01(bits[k] / n);

    /* [8..16) gradient orientation histogram. */
    double hist[8] = {0};
    double gsum = 0.0;
    for (int y = 1; y < h - 1; y++) {
        for (int x = 1; x < w - 1; x++) {
            double gx = (double)wubu_img_at(im, x + 1, y) - (double)wubu_img_at(im, x - 1, y);
            double gy = (double)wubu_img_at(im, x, y + 1) - (double)wubu_img_at(im, x, y - 1);
            double mag = sqrt(gx * gx + gy * gy);
            if (mag <= 0.0) continue;
            double ang = atan2(gy, gx);
            if (ang < 0.0) ang += 2.0 * M_PI;
            int bin = (int)(ang / (2.0 * M_PI) * 8.0) & 7;
            hist[bin] += mag;
            gsum += mag;
        }
    }
    for (int k = 0; k < 8; k++) out[8 + k] = gsum > 0.0 ? clamp01(hist[k] / gsum) : 0.0f;

    /* [16..24) moments. */
    double mean = 0.0;
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) mean += (double)im->data[(size_t)y * im->stride + x];
    mean /= n;

    double m2 = 0.0, m3 = 0.0, m4 = 0.0;
    uint64_t histo[256] = {0};
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            double d = (double)im->data[(size_t)y * im->stride + x] - mean;
            m2 += d * d; m3 += d * d * d; m4 += d * d * d * d;
            histo[im->data[(size_t)y * im->stride + x]]++;
        }
    m2 /= n; m3 /= n; m4 /= n;
    double var = m2 > 1e-12 ? m2 : 1e-12;
    double sd = sqrt(var);
    double skew = m3 / (var * sd);
    double kurt = m4 / (var * var);

    int p10 = 0, p50 = 0, p90 = 0;
    uint64_t cum = 0;
    for (int i = 0; i < 256; i++) {
        cum += histo[i];
        if (!p10 && cum >= (uint64_t)(0.10 * n)) p10 = i;
        if (!p50 && cum >= (uint64_t)(0.50 * n)) p50 = i;
        if (!p90 && cum >= (uint64_t)(0.90 * n)) p90 = i;
    }
    double ent = 0.0;
    for (int i = 0; i < 256; i++) {
        if (!histo[i]) continue;
        double p = (double)histo[i] / n;
        ent -= p * log2(p);
    }

    out[16] = clamp01(mean / 255.0);
    out[17] = clamp01(sd / 128.0);
    out[18] = clamp01((skew + 4.0) / 8.0);
    out[19] = clamp01(kurt / 16.0);
    out[20] = (float)p10 / 255.0f;
    out[21] = (float)p50 / 255.0f;
    out[22] = (float)p90 / 255.0f;
    out[23] = clamp01(ent / 8.0);

    /* [24..32) 2D Haar detail ratios for three scales (substitute for the
     * crate's 3D Schur-Feshbach terms; documented deviation). */
    for (int s = 0; s < 3; s++) {
        int step = 1 << (s + 1);
        double detail = 0.0, approx = 0.0;
        for (int y = 0; y + step < h; y += step)
            for (int x = 0; x + step < w; x += step) {
                double a = wubu_img_at(im, x, y);
                double b = wubu_img_at(im, x + step / 2, y);
                double c = wubu_img_at(im, x, y + step / 2);
                double d = wubu_img_at(im, x + step / 2, y + step / 2);
                detail += fabs((a + d) - (b + c));
                approx += fabs((a + b + c + d) * 0.25);
            }
        out[24 + s] = (detail + approx) > 0.0
                    ? clamp01(detail / (detail + approx)) : 0.0f;
    }

    /* [32..40) reserved: worldtube causality. */

    /* [40..64) 4x6 density grid. */
    for (int gy = 0; gy < 4; gy++) {
        for (int gx = 0; gx < 6; gx++) {
            int x0 = gx * w / 6, x1 = (gx + 1) * w / 6;
            int y0 = gy * h / 4, y1 = (gy + 1) * h / 4;
            double s = 0.0;
            int cnt = 0;
            for (int y = y0; y < y1; y++)
                for (int x = x0; x < x1; x++) { s += im->data[(size_t)y * im->stride + x]; cnt++; }
            out[40 + gy * 6 + gx] = cnt ? clamp01(s / (double)cnt / 255.0) : 0.0f;
        }
    }
    return 0;
}

static uint8_t d4_px(const wubu_img *im, int x, int y, int t)
{
    int W = im->w - 1, H = im->h - 1, sx = x, sy = y;
    switch (t) {
        case 0: break;
        case 1: sx = W - y; sy = x; break;
        case 2: sx = W - x; sy = H - y; break;
        case 3: sx = y; sy = H - x; break;
        case 4: sx = W - x; sy = y; break;
        case 5: sx = x; sy = H - y; break;
        case 6: sx = y; sy = x; break;
        default: sx = W - y; sy = H - x; break;
    }
    return im->data[(size_t)sy * im->stride + (size_t)sx];
}

uint64_t wubu_glassbox_hash(const wubu_img *im)
{
    if (!im || !im->data) return 0;

    uint64_t best = ~(uint64_t)0;
    wubu_img t = {0};

    for (int k = 0; k < 8; k++) {
        if (wubu_img_alloc(&t, im->w, im->h) != 0) return 0;
        for (int y = 0; y < im->h; y++)
            for (int x = 0; x < im->w; x++)
                t.data[(size_t)y * t.stride + x] = d4_px(im, x, y, k);

        float sig[WUBU_GLASS_DIMS];
        if (wubu_glassbox64(&t, sig) != 0) { wubu_img_free(&t); return 0; }

        uint64_t h = 1469598103934665603ULL;
        for (int i = 0; i < WUBU_GLASS_DIMS; i++) {
            unsigned q = (unsigned)(sig[i] * 255.0f + 0.5f);
            if (q > 255u) q = 255u;
            h ^= (uint64_t)(q & 0xFF);
            h *= 1099511628211ULL;
        }
        if (h < best) best = h;
        wubu_img_free(&t);
    }
    return best;
}