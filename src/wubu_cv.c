/*
 * wubu_cv.c -- core image + imgproc. See wubu_cv.h.
 */
#include "wubu_cv.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

int wubu_img_alloc(wubu_img *im, int w, int h)
{
    if (!im || w <= 0 || h <= 0) return -1;
    im->data = (uint8_t *)calloc((size_t)w * (size_t)h, 1);
    if (!im->data) return -1;
    im->w = w; im->h = h; im->stride = w;
    return 0;
}

void wubu_img_free(wubu_img *im)
{
    if (!im) return;
    free(im->data);
    im->data = NULL; im->w = im->h = im->stride = 0;
}

int wubu_img_clone(const wubu_img *src, wubu_img *dst)
{
    if (!src || !dst || !src->data) return -1;
    if (wubu_img_alloc(dst, src->w, src->h) != 0) return -1;
    for (int y = 0; y < src->h; y++)
        memcpy(dst->data + (size_t)y * dst->stride,
               src->data + (size_t)y * src->stride, (size_t)src->w);
    return 0;
}

void wubu_img_fill(wubu_img *im, uint8_t v)
{
    if (!im || !im->data) return;
    for (int y = 0; y < im->h; y++) memset(im->data + (size_t)y * im->stride, v, (size_t)im->w);
}

uint8_t wubu_img_at(const wubu_img *im, int x, int y)
{
    if (!im || !im->data) return 0;
    if (x < 0) x = 0;
    if (x >= im->w) x = im->w - 1;
    if (y < 0) y = 0;
    if (y >= im->h) y = im->h - 1;
    return im->data[(size_t)y * im->stride + (size_t)x];
}

int wubu_integral_u64(const wubu_img *im, uint64_t **sum, uint64_t **sumsq)
{
    if (!im || !im->data || !sum || !sumsq) return -1;
    int w = im->w, h = im->h, iw = w + 1;
    uint64_t *s = (uint64_t *)calloc((size_t)iw * (size_t)(h + 1), sizeof *s);
    uint64_t *q = (uint64_t *)calloc((size_t)iw * (size_t)(h + 1), sizeof *q);
    if (!s || !q) { free(s); free(q); return -1; }

    for (int y = 0; y < h; y++) {
        uint64_t rowsum = 0, rowsq = 0;
        for (int x = 0; x < w; x++) {
            uint64_t v = im->data[(size_t)y * im->stride + (size_t)x];
            rowsum += v;
            rowsq += v * v;
            s[(size_t)(y + 1) * iw + (x + 1)] = s[(size_t)y * iw + (x + 1)] + rowsum;
            q[(size_t)(y + 1) * iw + (x + 1)] = q[(size_t)y * iw + (x + 1)] + rowsq;
        }
    }
    *sum = s; *sumsq = q;
    return 0;
}

static uint64_t rect_sum(const uint64_t *ii, int iw, int x, int y, int bw, int bh)
{
    int x1 = x, y1 = y, x2 = x + bw, y2 = y + bh;
    return ii[(size_t)y2 * iw + x2] - ii[(size_t)y1 * iw + x2]
         - ii[(size_t)y2 * iw + x1] + ii[(size_t)y1 * iw + x1];
}

int wubu_box_stats(const uint64_t *sum, const uint64_t *sumsq, int w, int h,
                   int x, int y, int bw, int bh, double *mean, double *var)
{
    if (!sum || !sumsq || !mean || !var) return -1;
    if (bw <= 0 || bh <= 0 || x < 0 || y < 0 || x + bw > w || y + bh > h) return -1;

    int iw = w + 1;
    double n = (double)bw * (double)bh;
    double s = (double)rect_sum(sum, iw, x, y, bw, bh);
    double q = (double)rect_sum(sumsq, iw, x, y, bw, bh);
    *mean = s / n;
    double v = q / n - (*mean) * (*mean);
    *var = v > 0.0 ? v : 0.0;
    return 0;
}

int wubu_sobel3(const wubu_img *im, int16_t *gx, int16_t *gy)
{
    if (!im || !im->data || !gx || !gy) return -1;

    for (int y = 0; y < im->h; y++) {
        for (int x = 0; x < im->w; x++) {
            int a = wubu_img_at(im, x - 1, y - 1), b = wubu_img_at(im, x, y - 1), c = wubu_img_at(im, x + 1, y - 1);
            int d = wubu_img_at(im, x - 1, y),                              f = wubu_img_at(im, x + 1, y);
            int g = wubu_img_at(im, x - 1, y + 1), hh = wubu_img_at(im, x, y + 1), i = wubu_img_at(im, x + 1, y + 1);
            int sx = (c + 2 * f + i) - (a + 2 * d + g);
            int sy = (g + 2 * hh + i) - (a + 2 * b + c);
            gx[(size_t)y * im->w + x] = (int16_t)sx;
            gy[(size_t)y * im->w + x] = (int16_t)sy;
        }
    }
    return 0;
}

int wubu_otsu(const wubu_img *im)
{
    if (!im || !im->data) return -1;
    uint64_t hist[256] = {0};
    uint64_t total = (uint64_t)im->w * (uint64_t)im->h;
    for (int y = 0; y < im->h; y++)
        for (int x = 0; x < im->w; x++)
            hist[im->data[(size_t)y * im->stride + x]]++;

    double sum_all = 0.0;
    for (int i = 0; i < 256; i++) sum_all += (double)i * (double)hist[i];

    double sum_b = 0.0, best_var = -1.0;
    uint64_t w_b = 0;
    int best = 0;
    for (int t = 0; t < 256; t++) {
        w_b += hist[t];
        if (w_b == 0) continue;
        uint64_t w_f = total - w_b;
        if (w_f == 0) break;
        sum_b += (double)t * (double)hist[t];
        double m_b = sum_b / (double)w_b;
        double m_f = (sum_all - sum_b) / (double)w_f;
        double between = (double)w_b * (double)w_f * (m_b - m_f) * (m_b - m_f);
        if (between > best_var) { best_var = between; best = t; }
    }
    return best;
}

int wubu_gauss3(const wubu_img *in, wubu_img *out)
{
    if (!in || !in->data || !out || !out->data) return -1;
    if (in->w != out->w || in->h != out->h) return -1;

    const int k[3] = {1, 2, 1};
    uint16_t *tmp = (uint16_t *)malloc((size_t)in->w * (size_t)in->h * sizeof *tmp);
    if (!tmp) return -1;

    for (int y = 0; y < in->h; y++) {
        for (int x = 0; x < in->w; x++) {
            unsigned acc = 0;
            for (int i = -1; i <= 1; i++) acc += (unsigned)wubu_img_at(in, x + i, y) * k[i + 1];
            tmp[(size_t)y * in->w + x] = (uint16_t)((acc + 2) >> 2);
        }
    }
    for (int y = 0; y < in->h; y++) {
        for (int x = 0; x < in->w; x++) {
            unsigned acc = 0;
            for (int j = -1; j <= 1; j++) {
                int yy = y + j;
                if (yy < 0) yy = 0;
                if (yy >= in->h) yy = in->h - 1;
                acc += (unsigned)tmp[(size_t)yy * in->w + x] * k[j + 1];
            }
            out->data[(size_t)y * out->stride + x] = (uint8_t)((acc + 2) >> 2);
        }
    }
    free(tmp);
    return 0;
}