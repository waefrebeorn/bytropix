/*
 * wubu_flow.c -- sparse Lucas-Kanade. See wubu_flow.h.
 */
#include "wubu_flow.h"
#include "wubu_uplift.h"
#include <math.h>

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
