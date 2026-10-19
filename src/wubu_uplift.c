/*
 * wubu_uplift.c -- mass-conserving 2x zoom + D4 canonical hash.
 */
#include "wubu_uplift.h"
#include <math.h>
#include <stdlib.h>

int wubu_uplift2x(const wubu_img *in, wubu_img *out)
{
    if (!in || !in->data || !out) return -1;
    if (wubu_img_alloc(out, in->w * 2, in->h * 2) != 0) return -1;

    for (int y = 0; y < in->h; y++) {
        for (int x = 0; x < in->w; x++) {
            uint8_t v = in->data[(size_t)y * in->stride + x];
            size_t o = (size_t)(2 * y) * out->stride + (size_t)(2 * x);
            out->data[o] = v;
            out->data[o + 1] = v;
            out->data[o + out->stride] = v;
            out->data[o + out->stride + 1] = v;
        }
    }
    return 0;
}

int wubu_down2x_box(const wubu_img *in, wubu_img *out)
{
    if (!in || !in->data || !out) return -1;
    int ow = in->w / 2, oh = in->h / 2;
    if (ow <= 0 || oh <= 0) return -1;
    if (wubu_img_alloc(out, ow, oh) != 0) return -1;

    for (int y = 0; y < oh; y++) {
        for (int x = 0; x < ow; x++) {
            unsigned s = 0;
            s += wubu_img_at(in, 2 * x, 2 * y);
            s += wubu_img_at(in, 2 * x + 1, 2 * y);
            s += wubu_img_at(in, 2 * x, 2 * y + 1);
            s += wubu_img_at(in, 2 * x + 1, 2 * y + 1);
            out->data[(size_t)y * out->stride + x] = (uint8_t)((s + 2) >> 2);
        }
    }
    return 0;
}

static uint8_t d4_pixel(const wubu_img *im, int x, int y, int t)
{
    int W = im->w - 1, H = im->h - 1;
    int sx = x, sy = y;
    switch (t) {
        case 0: sx = x;         sy = y;         break;  /* identity */
        case 1: sx = W - y;     sy = x;         break;  /* rot 90 */
        case 2: sx = W - x;     sy = H - y;     break;  /* rot 180 */
        case 3: sx = y;         sy = H - x;     break;  /* rot 270 */
        case 4: sx = W - x;     sy = y;         break;  /* flip H */
        case 5: sx = x;         sy = H - y;     break;  /* flip V */
        case 6: sx = y;         sy = x;         break;  /* transpose */
        default: sx = W - y;    sy = H - x;     break;  /* anti-transpose */
    }
    return im->data[(size_t)sy * im->stride + (size_t)sx];
}

uint64_t wubu_d4_hash(const wubu_img *im)
{
    if (!im || !im->data) return 0;

    uint64_t best = ~(uint64_t)0;
    for (int t = 0; t < 8; t++) {
        uint64_t h = 1469598103934665603ULL;
        for (int y = 0; y < im->h; y++) {
            for (int x = 0; x < im->w; x++) {
                h ^= d4_pixel(im, x, y, t);
                h *= 1099511628211ULL;
            }
        }
        if (h < best) best = h;
    }
    return best;
}

static float sample_bi(const wubu_img *im, float x, float y, int *inside)
{
    int x0 = (int)floorf(x), y0 = (int)floorf(y);
    if (x0 < 0 || y0 < 0 || x0 + 1 >= im->w || y0 + 1 >= im->h) { *inside = 0; return 0.0f; }
    *inside = 1;
    float fx = x - (float)x0, fy = y - (float)y0;
    float a = (float)wubu_img_at(im, x0, y0);
    float b = (float)wubu_img_at(im, x0 + 1, y0);
    float c = (float)wubu_img_at(im, x0, y0 + 1);
    float d = (float)wubu_img_at(im, x0 + 1, y0 + 1);
    float top = a * (1.0f - fx) + b * fx;
    float bot = c * (1.0f - fx) + d * fx;
    return top * (1.0f - fy) + bot * fy;
}

int wubu_burst_super_resolve(const wubu_img *const *frames,
                             const float *dx, const float *dy,
                             int n, int factor, wubu_img *out)
{
    if (!frames || !frames[0] || !dx || !dy || !out || n <= 0) return -1;
    if (factor < 1 || factor > 4) return -1;

    int w = frames[0]->w, h = frames[0]->h;
    if (wubu_img_alloc(out, w * factor, h * factor) != 0) return -1;

    for (int Y = 0; Y < out->h; Y++) {
        for (int X = 0; X < out->w; X++) {
            double acc = 0.0;
            int used = 0;

            for (int i = 0; i < n; i++) {
                if (!frames[i]) continue;
                float x = (float)X / (float)factor - dx[i];
                float y = (float)Y / (float)factor - dy[i];
                int inside = 0;
                float v = sample_bi(frames[i], x, y, &inside);
                if (!inside) continue;
                acc += v;
                used++;
            }
            out->data[(size_t)Y * out->stride + X] =
                used ? (uint8_t)((acc / used) + 0.5) : 0;
        }
    }
    return 0;
}
