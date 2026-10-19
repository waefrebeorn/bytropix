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
/* ---------------- Canny + contours (wave 2) ---------------- */

int wubu_canny(const wubu_img *in, int low, int high, wubu_img *edges)
{
    if (!in || !in->data || !edges || !edges->data) return -1;
    if (low < 0 || high < low) return -1;
    if (edges->w != in->w || edges->h != in->h) return -1;

    int w = in->w, h = in->h;
    int16_t *gx = (int16_t *)malloc((size_t)w * h * sizeof *gx);
    int16_t *gy = (int16_t *)malloc((size_t)w * h * sizeof *gy);
    int32_t *mag = (int32_t *)malloc((size_t)w * h * sizeof *mag);
    uint8_t *state = (uint8_t *)calloc((size_t)w * h, 1); /* 0 none, 1 weak, 2 edge */
    int *stack = (int *)malloc((size_t)w * h * sizeof *stack);
    if (!gx || !gy || !mag || !state || !stack) {
        free(gx); free(gy); free(mag); free(state); free(stack);
        return -1;
    }

    wubu_sobel3(in, gx, gy);

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int32_t a = gx[(size_t)y * w + x], b = gy[(size_t)y * w + x];
            mag[(size_t)y * w + x] = a * a + b * b;
        }
    }

    /* Non-maximum suppression along the quantised gradient direction. */
    for (int y = 1; y < h - 1; y++) {
        for (int x = 1; x < w - 1; x++) {
            size_t i = (size_t)y * w + x;
            int32_t a = gx[i], b = gy[i];
            int32_t m = mag[i];
            int32_t n1, n2;
            int aa = a < 0 ? -a : a, bb = b < 0 ? -b : b;

            if (aa > 2 * bb) {                    /* horizontal gradient */
                n1 = mag[i - 1]; n2 = mag[i + 1];
            } else if (bb > 2 * aa) {             /* vertical */
                n1 = mag[i - w]; n2 = mag[i + w];
            } else if ((a > 0) == (b > 0)) {      /* main diagonal */
                n1 = mag[i - w - 1]; n2 = mag[i + w + 1];
            } else {                              /* anti-diagonal */
                n1 = mag[i - w + 1]; n2 = mag[i + w - 1];
            }
            if (m < n1 || m < n2) mag[i] = 0;
        }
    }

    /* Threshold in magnitude units (the caller's numbers, not squared ones). */
    int32_t lo2 = (int32_t)low * (int32_t)low;
    int32_t hi2 = (int32_t)high * (int32_t)high;
    int sp = 0;

    for (size_t i = 0; i < (size_t)w * h; i++) {
        if (mag[i] >= hi2) { state[i] = 2; stack[sp++] = (int)i; }
        else if (mag[i] >= lo2) state[i] = 1;
    }

    while (sp > 0) {
        int i = stack[--sp];
        int x = i % w, y = i / w;
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int nx = x + dx, ny = y + dy;
                if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
                size_t j = (size_t)ny * w + nx;
                if (state[j] == 1) { state[j] = 2; stack[sp++] = (int)j; }
            }
        }
    }

    for (size_t i = 0; i < (size_t)w * h; i++)
        edges->data[i] = state[i] == 2 ? 255 : 0;

    free(gx); free(gy); free(mag); free(state); free(stack);
    return 0;
}

int wubu_find_contours(const wubu_img *edges, wubu_contour *out,
                       int max_contours, int min_area)
{
    if (!edges || !edges->data || !out || max_contours <= 0) return -1;
    int w = edges->w, h = edges->h;
    uint8_t *seen = (uint8_t *)calloc((size_t)w * h, 1);
    int *stack = (int *)malloc((size_t)w * h * sizeof *stack);
    if (!seen || !stack) { free(seen); free(stack); return -1; }

    int found = 0;
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            size_t start = (size_t)y * w + x;
            if (!edges->data[start] || seen[start]) continue;

            int sp = 0, area = 0;
            int minx = x, maxx = x, miny = y, maxy = y;
            stack[sp++] = (int)start;
            seen[start] = 1;

            while (sp > 0) {
                int i = stack[--sp];
                int cx = i % w, cy = i / w;
                area++;
                if (cx < minx) minx = cx;
                if (cx > maxx) maxx = cx;
                if (cy < miny) miny = cy;
                if (cy > maxy) maxy = cy;
                for (int dy = -1; dy <= 1; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        int nx = cx + dx, ny = cy + dy;
                        if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
                        size_t j = (size_t)ny * w + nx;
                        if (edges->data[j] && !seen[j]) { seen[j] = 1; stack[sp++] = (int)j; }
                    }
                }
            }

            if (area >= min_area && found < max_contours) {
                out[found].x = minx;
                out[found].y = miny;
                out[found].w = maxx - minx + 1;
                out[found].h = maxy - miny + 1;
                out[found].area = area;
                found++;
            }
        }
    }

    free(seen); free(stack);
    return found;
}

int wubu_morph3(const wubu_img *in, wubu_img *out, int mode)
{
    if (!in || !in->data || !out || !out->data) return -1;
    if (mode != 0 && mode != 1) return -1;
    if (in->w != out->w || in->h != out->h) return -1;

    for (int y = 0; y < in->h; y++) {
        for (int x = 0; x < in->w; x++) {
            int v = mode ? 0 : 255;
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    int p = wubu_img_at(in, x + dx, y + dy);
                    if (mode) { if (p > v) v = p; }
                    else      { if (p < v) v = p; }
                }
            }
            out->data[(size_t)y * out->stride + x] = (uint8_t)v;
        }
    }
    return 0;
}

int wubu_foveated_rois(const wubu_img *im, int tile, int k, int nms_radius,
                       wubu_roi *out, int max_rois)
{
    if (!im || !im->data || !out || tile <= 0 || k <= 0 || max_rois <= 0) return -1;
    if (nms_radius < 0) return -1;

    int w = im->w, h = im->h;
    int gw = (w + tile - 1) / tile, gh = (h + tile - 1) / tile;
    size_t nt = (size_t)gw * (size_t)gh;

    int16_t *gx = (int16_t *)malloc((size_t)w * h * sizeof *gx);
    int16_t *gy = (int16_t *)malloc((size_t)w * h * sizeof *gy);
    double *score = (double *)calloc(nt, sizeof *score);
    uint8_t *used = (uint8_t *)calloc(nt, 1);
    if (!gx || !gy || !score || !used) {
        free(gx); free(gy); free(score); free(used);
        return -1;
    }

    wubu_sobel3(im, gx, gy);
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            size_t i = (size_t)y * w + x;
            int a = gx[i], b = gy[i];
            if (a < 0) a = -a;
            if (b < 0) b = -b;
            score[(size_t)(y / tile) * gw + (x / tile)] += (double)(a + b);
        }
    }

    int found = 0;
    while (found < k && found < max_rois) {
        size_t best = nt;
        double bestv = 0.0;
        for (size_t i = 0; i < nt; i++)
            if (!used[i] && score[i] > bestv) { bestv = score[i]; best = i; }
        if (best == nt) break;   /* nothing left with energy */

        int bx = (int)(best % (size_t)gw), by = (int)(best / (size_t)gw);
        for (int dy = -nms_radius; dy <= nms_radius; dy++)
            for (int dx = -nms_radius; dx <= nms_radius; dx++) {
                int nx = bx + dx, ny = by + dy;
                if (nx < 0 || ny < 0 || nx >= gw || ny >= gh) continue;
                used[(size_t)ny * gw + nx] = 1;
            }

        int x0 = bx * tile, y0 = by * tile;
        out[found].x = x0;
        out[found].y = y0;
        out[found].w = (x0 + tile <= w) ? tile : (w - x0);
        out[found].h = (y0 + tile <= h) ? tile : (h - y0);
        out[found].score = bestv;
        found++;
    }

    free(gx); free(gy); free(score); free(used);
    return found;
}


int wubu_contour_trace(const wubu_img *edges, int sx, int sy,
                       int *xs, int *ys, int max_points)
{
    if (!edges || !edges->data || !xs || !ys || max_points <= 0) return -1;
    if (sx < 0 || sy < 0 || sx >= edges->w || sy >= edges->h) return -1;
    if (!edges->data[(size_t)sy * edges->stride + sx]) return 0;

    int w = edges->w, h = edges->h;
    size_t n = (size_t)w * h;
    uint8_t *seen = (uint8_t *)calloc(n, 1);
    int *stack = (int *)malloc(n * sizeof *stack);
    int *comp = (int *)malloc(n * sizeof *comp);
    if (!seen || !stack || !comp) { free(seen); free(stack); free(comp); return -1; }

    int sp = 0, cn = 0;
    int start = sy * w + sx;
    seen[start] = 1;
    stack[sp++] = start;

    double cx = 0, cy = 0;
    while (sp > 0) {
        int i = stack[--sp];
        comp[cn++] = i;
        int x = i % w, y = i / w;
        cx += x; cy += y;
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int nx = x + dx, ny = y + dy;
                if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
                size_t j = (size_t)ny * w + nx;
                if (edges->data[j] && !seen[j]) { seen[j] = 1; stack[sp++] = (int)j; }
            }
        }
    }
    cx /= (double)cn;
    cy /= (double)cn;

    /* Insertion sort by angle: components here are small (tens of points). */
    for (int i = 1; i < cn; i++) {
        int key = comp[i];
        double ka = atan2((double)(key / w) - cy, (double)(key % w) - cx);
        int j = i - 1;
        while (j >= 0) {
            double ja = atan2((double)(comp[j] / w) - cy, (double)(comp[j] % w) - cx);
            if (ja <= ka) break;
            comp[j + 1] = comp[j];
            j--;
        }
        comp[j + 1] = key;
    }

    int out_n = cn < max_points ? cn : max_points;
    for (int i = 0; i < out_n; i++) {
        xs[i] = comp[i] % w;
        ys[i] = comp[i] / w;
    }

    free(seen); free(stack); free(comp);
    return out_n;
}
