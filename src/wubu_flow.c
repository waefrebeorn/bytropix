/*
 * wubu_flow.c -- sparse Lucas-Kanade. See wubu_flow.h.
 */
#include "wubu_flow.h"
#include "wubu_uplift.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

static float sample_bilinear(const wubu_img *im, float x, float y)
{
    int x0 = (int)floorf(x), y0 = (int)floorf(y);
    float fx = x - (float)x0, fy = y - (float)y0;

    float a = (float)wubu_img_at(im, x0, y0);
    float b = (float)wubu_img_at(im, x0 + 1, y0);
    float c = (float)wubu_img_at(im, x0, y0 + 1);
    float d = (float)wubu_img_at(im, x0 + 1, y0 + 1);

    float top = a * (1.0f - fx) + b * fx;
    float bot = c * (1.0f - fx) + d * fx;
    return top * (1.0f - fy) + bot * fy;
}

static void lk_refine(const wubu_img *prev, const wubu_img *curr,
                      int px, int py, int win, int iters, float *u, float *v);

int wubu_lk_flow(const wubu_img *prev, const wubu_img *curr,
                 const wubu_keypoint *kp, int n, int win, int iters,
                 float *dx, float *dy)
{
    if (!prev || !curr || !kp || !dx || !dy) return -1;
    if (prev->w != curr->w || prev->h != curr->h) return -1;
    if (win < 3 || (win % 2) == 0 || iters <= 0 || n <= 0) return -1;

    int r = win / 2;

    for (int k = 0; k < n; k++) {
        int px = kp[k].x, py = kp[k].y;
        if (px < r + 1 || py < r + 1 || px >= prev->w - r - 1 || py >= prev->h - r - 1) {
            dx[k] = dy[k] = 0.0f;
            continue;
        }

        float u = 0.0f, v = 0.0f;
        lk_refine(prev, curr, px, py, win, iters, &u, &v);
        dx[k] = u;
        dy[k] = v;
    }
    return 0;
}

static void lk_refine(const wubu_img *prev, const wubu_img *curr,
                      int px, int py, int win, int iters, float *u, float *v)
{
    int r = win / 2;
    float uu = *u, vv = *v;

    for (int it = 0; it < iters; it++) {
        double A11 = 0, A12 = 0, A22 = 0, b1 = 0, b2 = 0;

        for (int oy = -r; oy <= r; oy++) {
            for (int ox = -r; ox <= r; ox++) {
                int x = px + ox, y = py + oy;
                float Ix = ((float)wubu_img_at(prev, x + 1, y)
                          - (float)wubu_img_at(prev, x - 1, y)) * 0.5f;
                float Iy = ((float)wubu_img_at(prev, x, y + 1)
                          - (float)wubu_img_at(prev, x, y - 1)) * 0.5f;
                float It = sample_bilinear(curr, (float)x + uu, (float)y + vv)
                         - (float)wubu_img_at(prev, x, y);
                A11 += (double)Ix * Ix;
                A12 += (double)Ix * Iy;
                A22 += (double)Iy * Iy;
                b1  -= (double)Ix * It;
                b2  -= (double)Iy * It;
            }
        }

        double det = A11 * A22 - A12 * A12;
        if (fabs(det) < 1e-6) break;

        double du = (A22 * b1 - A12 * b2) / det;
        double dv = (A11 * b2 - A12 * b1) / det;

        if (du > 2.0) du = 2.0;
        if (du < -2.0) du = -2.0;
        if (dv > 2.0) dv = 2.0;
        if (dv < -2.0) dv = -2.0;

        uu += (float)du;
        vv += (float)dv;

        if (fabs(du) < 1e-3 && fabs(dv) < 1e-3) break;
    }
    *u = uu;
    *v = vv;
}

int wubu_lk_flow_pyr(const wubu_img *prev, const wubu_img *curr,
                     const wubu_keypoint *kp, int n, int levels,
                     int win, int iters, float *dx, float *dy)
{
    if (!prev || !curr || !kp || !dx || !dy) return -1;
    if (prev->w != curr->w || prev->h != curr->h) return -1;
    if (levels < 1 || levels > 5 || win < 3 || (win % 2) == 0 || iters <= 0 || n <= 0)
        return -1;

    wubu_img p[5] = {{0}}, c[5] = {{0}};
    if (wubu_img_clone(prev, &p[0]) != 0) return -1;
    if (wubu_img_clone(curr, &c[0]) != 0) { wubu_img_free(&p[0]); return -1; }

    int built = 1;
    for (int L = 1; L < levels; L++) {
        if (wubu_down2x_box(&p[L - 1], &p[L]) != 0) break;
        if (wubu_down2x_box(&c[L - 1], &c[L]) != 0) break;
        built++;
    }

    for (int k = 0; k < n; k++) {
        float u = 0.0f, v = 0.0f;
        int ok = 1;

        for (int L = built - 1; L >= 0; L--) {
            int px = kp[k].x >> L, py = kp[k].y >> L;
            int r = win / 2;
            if (px < r + 1 || py < r + 1 || px >= p[L].w - r - 1 || py >= p[L].h - r - 1) {
                ok = 0;
                break;
            }
            lk_refine(&p[L], &c[L], px, py, win, iters, &u, &v);
            if (L > 0) { u *= 2.0f; v *= 2.0f; }
        }
        dx[k] = ok ? u : 0.0f;
        dy[k] = ok ? v : 0.0f;
    }

    for (int L = 0; L < built; L++) {
        wubu_img_free(&p[L]);
        wubu_img_free(&c[L]);
    }
    return 0;
}


int wubu_block_flow(const wubu_img *prev, const wubu_img *curr,
                    int block, int search, int16_t *dx, int16_t *dy)
{
    if (!prev || !curr || !dx || !dy) return -1;
    if (prev->w != curr->w || prev->h != curr->h) return -1;
    if (block < 4 || block % 2 || search < 0 || search > 16) return -1;

    int w = prev->w, h = prev->h;
    int gw = (w + block - 1) / block, gh = (h + block - 1) / block;
    int n = 0;

    for (int by = 0; by < gh; by++) {
        for (int bx = 0; bx < gw; bx++) {
            int cx = bx * block + block / 2;
            int cy = by * block + block / 2;
            int bestx = 0, besty = 0;
            long best = -1;

            for (int sy = -search; sy <= search; sy++) {
                for (int sx = -search; sx <= search; sx++) {
                    long sad = 0;
                    int valid = 1;
                    for (int oy = -block / 2; oy < block / 2 && valid; oy += 2) {
                        int py = cy + oy;
                        int qy = py + sy;
                        if (py < 0 || py >= h || qy < 0 || qy >= h) { valid = 0; break; }
                        for (int ox = -block / 2; ox < block / 2; ox += 2) {
                            int px = cx + ox, qx = px + sx;
                            if (px < 0 || px >= w || qx < 0 || qx >= w) { valid = 0; break; }
                            int a = prev->data[(size_t)py * prev->stride + px];
                            int b = curr->data[(size_t)qy * curr->stride + qx];
                            sad += a > b ? a - b : b - a;
                        }
                    }
                    if (!valid) continue;
                    if (best < 0 || sad < best) { best = sad; bestx = sx; besty = sy; }
                }
            }
            dx[n] = (int16_t)bestx;
            dy[n] = (int16_t)besty;
            n++;
        }
    }
    return n;
}


static long block_sad(const wubu_img *a, const wubu_img *b, int cx, int cy,
                      int block, int sx, int sy)
{
    int w = a->w, h = a->h;
    long sad = 0;
    for (int oy = -block / 2; oy < block / 2; oy += 2) {
        int py = cy + oy, qy = py + sy;
        if (py < 0 || py >= h || qy < 0 || qy >= h) return -1;
        for (int ox = -block / 2; ox < block / 2; ox += 2) {
            int px = cx + ox, qx = px + sx;
            if (px < 0 || px >= w || qx < 0 || qx >= w) return -1;
            int va = a->data[(size_t)py * a->stride + px];
            int vb = b->data[(size_t)qy * b->stride + qx];
            sad += va > vb ? va - vb : vb - va;
        }
    }
    return sad;
}

static float parabola(float sm, float s0, float sp)
{
    float den = sm - 2.0f * s0 + sp;
    if (fabsf(den) < 1e-6f) return 0.0f;
    float d = 0.5f * (sm - sp) / den;
    if (d > 0.5f) d = 0.5f;
    if (d < -0.5f) d = -0.5f;
    return d;
}

int wubu_block_flow_subpix(const wubu_img *prev, const wubu_img *curr,
                           int block, int search, float *dx, float *dy)
{
    if (!prev || !curr || !dx || !dy) return -1;
    if (prev->w != curr->w || prev->h != curr->h) return -1;
    if (block < 4 || block % 2 || search < 0 || search > 16) return -1;

    int w = prev->w, h = prev->h;
    int gw = (w + block - 1) / block, gh = (h + block - 1) / block;
    int n = 0;

    for (int by = 0; by < gh; by++) {
        for (int bx = 0; bx < gw; bx++) {
            int cx = bx * block + block / 2;
            int cy = by * block + block / 2;
            int bestx = 0, besty = 0;
            long best = -1;

            for (int sy = -search; sy <= search; sy++) {
                for (int sx = -search; sx <= search; sx++) {
                    long sad = block_sad(prev, curr, cx, cy, block, sx, sy);
                    if (sad < 0) continue;
                    if (best < 0 || sad < best) { best = sad; bestx = sx; besty = sy; }
                }
            }

            float fx = (float)bestx, fy = (float)besty;
            if (best >= 0) {
                long l = block_sad(prev, curr, cx, cy, block, bestx - 1, besty);
                long r = block_sad(prev, curr, cx, cy, block, bestx + 1, besty);
                if (l >= 0 && r >= 0)
                    fx += parabola((float)l, (float)best, (float)r);

                long u = block_sad(prev, curr, cx, cy, block, bestx, besty - 1);
                long d = block_sad(prev, curr, cx, cy, block, bestx, besty + 1);
                if (u >= 0 && d >= 0)
                    fy += parabola((float)u, (float)best, (float)d);
            }

            dx[n] = fx;
            dy[n] = fy;
            n++;
        }
    }
    return n;
}


int wubu_horn_schunck(const wubu_img *prev, const wubu_img *curr,
                      int iters, float alpha, float *dx, float *dy)
{
    if (!prev || !curr || !dx || !dy) return -1;
    if (prev->w != curr->w || prev->h != curr->h) return -1;
    if (iters <= 0 || alpha <= 0.0f) return -1;

    int w = prev->w, h = prev->h;
    size_t n = (size_t)w * h;

    float *ix = (float *)calloc(n, sizeof *ix);
    float *iy = (float *)calloc(n, sizeof *iy);
    float *it = (float *)calloc(n, sizeof *it);
    float *ub = (float *)calloc(n, sizeof *ub);
    float *vb = (float *)calloc(n, sizeof *vb);
    if (!ix || !iy || !it || !ub || !vb) {
        free(ix); free(iy); free(it); free(ub); free(vb);
        return -1;
    }

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            size_t i = (size_t)y * w + x;
            float a = (float)wubu_img_at(prev, x, y);
            float b = (float)wubu_img_at(curr, x, y);
            float ax = 0.25f * ((float)wubu_img_at(prev, x + 1, y) - a
                              + (float)wubu_img_at(curr, x + 1, y) - b
                              + (float)wubu_img_at(prev, x + 1, y + 1) - (float)wubu_img_at(prev, x, y + 1)
                              + (float)wubu_img_at(curr, x + 1, y + 1) - (float)wubu_img_at(curr, x, y + 1));
            float ay = 0.25f * ((float)wubu_img_at(prev, x, y + 1) - a
                              + (float)wubu_img_at(curr, x, y + 1) - b
                              + (float)wubu_img_at(prev, x + 1, y + 1) - (float)wubu_img_at(prev, x + 1, y)
                              + (float)wubu_img_at(curr, x + 1, y + 1) - (float)wubu_img_at(curr, x + 1, y));
            ix[i] = ax;
            iy[i] = ay;
            it[i] = b - a;
        }
    }

    memset(dx, 0, n * sizeof *dx);
    memset(dy, 0, n * sizeof *dy);

    float a2 = alpha * alpha;
    for (int k = 0; k < iters; k++) {
        /* neighbour averages */
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                size_t i = (size_t)y * w + x;
                float su = 0, sv = 0;
                int c = 0;
                if (x > 0)     { su += dx[i - 1]; sv += dy[i - 1]; c++; }
                if (x + 1 < w) { su += dx[i + 1]; sv += dy[i + 1]; c++; }
                if (y > 0)     { su += dx[i - w]; sv += dy[i - w]; c++; }
                if (y + 1 < h) { su += dx[i + w]; sv += dy[i + w]; c++; }
                ub[i] = c ? su / (float)c : 0.0f;
                vb[i] = c ? sv / (float)c : 0.0f;
            }
        }
        for (size_t i = 0; i < n; i++) {
            float den = a2 + ix[i] * ix[i] + iy[i] * iy[i];
            float num = ix[i] * ub[i] + iy[i] * vb[i] + it[i];
            dx[i] = ub[i] - ix[i] * num / den;
            dy[i] = vb[i] - iy[i] * num / den;
        }
    }

    free(ix); free(iy); free(it); free(ub); free(vb);
    return 0;
}
