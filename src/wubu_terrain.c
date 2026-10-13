/* wubu_terrain.c -- WORLDSIM #259 terrain heightfield + erosion.
 * See wubu_terrain.h. Particle-based hydraulic erosion, deterministic LCG. */
#include "wubu_terrain.h"
#include <math.h>

static uint64_t lcg(uint64_t *s) {
    *s = *s * 6364136223846793005ULL + 1442695040888963407ULL;
    return *s;
}
static double rng_unit(uint64_t *s) {
    return (double)(lcg(s) >> 11) / (double)(1ULL << 53);
}

int wubu_terrain_init(wubu_terrain_t *t, int w, int h, double base) {
    if (!t || w <= 0 || h <= 0 || w > WUBU_TERRAIN_MAX || h > WUBU_TERRAIN_MAX) return -1;
    t->w = w; t->h = h;
    t->rng_state = 0x123456789ABCDEFULL;   /* fixed seed -> deterministic */
    t->sediment = 0.0;
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
            t->height[y][x] = base;
    return 0;
}

double wubu_terrain_get(const wubu_terrain_t *t, int x, int y) {
    if (!t || x < 0 || y < 0 || x >= t->w || y >= t->h) return 0.0;
    return t->height[y][x];
}

int wubu_terrain_set(wubu_terrain_t *t, int x, int y, double h) {
    if (!t || x < 0 || y < 0 || x >= t->w || y >= t->h) return -1;
    t->height[y][x] = h;
    return 0;
}

int wubu_terrain_bump(wubu_terrain_t *t, int cx, int cy, double r, double amp) {
    if (!t || r <= 0.0) return -1;
    double r2 = r * r;
    for (int y = 0; y < t->h; y++) {
        for (int x = 0; x < t->w; x++) {
            double dx = x - cx, dy = y - cy;
            double d2 = dx*dx + dy*dy;
            if (d2 < r2) {
                double falloff = exp(-3.0 * d2 / r2);   /* gaussian */
                t->height[y][x] += amp * falloff;
            }
        }
    }
    return 0;
}

int wubu_terrain_erode_drop(wubu_terrain_t *t, int max_steps) {
    if (!t || max_steps <= 0) return 0;
    int x = (int)(rng_unit(&t->rng_state) * t->w);
    int y = (int)(rng_unit(&t->rng_state) * t->h);
    if (x >= t->w) x = t->w - 1;
    if (y >= t->h) y = t->h - 1;
    double carry = 0.0;
    int visited = 0;
    const double erode_rate = 0.3, deposit_rate = 0.3, capacity_k = 4.0;

    for (int step = 0; step < max_steps; step++) {
        visited++;
        /* steepest descent among 4 neighbours */
        double h_here = t->height[y][x];
        int bx = x, by = y;
        double lowest = h_here;
        static const int dxs[4] = {1,-1,0,0}, dys[4] = {0,0,1,-1};
        for (int i = 0; i < 4; i++) {
            int nx = x + dxs[i], ny = y + dys[i];
            if (nx < 0 || ny < 0 || nx >= t->w || ny >= t->h) continue;
            if (t->height[ny][nx] < lowest) { lowest = t->height[ny][nx]; bx = nx; by = ny; }
        }
        if (bx == x && by == y) {
            /* local minimum (pit): deposit all carried sediment */
            t->height[y][x] += carry;
            t->sediment += fabs(carry);
            break;
        }
        double slope = h_here - t->height[by][bx];
        /* capacity grows with slope and current load */
        double capacity = capacity_k * slope;
        if (carry < capacity) {
            /* erode: pick up sediment */
            double take = erode_rate * (capacity - carry);
            t->height[y][x] -= take;
            carry += take;
            t->sediment += fabs(take);
        } else {
            /* deposit: drop excess */
            double drop = deposit_rate * (carry - capacity);
            t->height[y][x] += drop;
            carry -= drop;
            t->sediment += fabs(drop);
        }
        x = bx; y = by;
    }
    return visited;
}

int wubu_terrain_erode(wubu_terrain_t *t, int n_drops, int max_steps) {
    if (!t || n_drops < 0) return -1;
    for (int i = 0; i < n_drops; i++) wubu_terrain_erode_drop(t, max_steps);
    return 0;
}

double wubu_terrain_sediment(const wubu_terrain_t *t) {
    return t ? t->sediment : 0.0;
}

double wubu_terrain_roughness(const wubu_terrain_t *t) {
    if (!t) return 0.0;
    /* mean absolute height (a rough proxy for terrain relief) */
    double sum = 0.0;
    int n = 0;
    for (int y = 0; y < t->h; y++)
        for (int x = 0; x < t->w; x++) { sum += fabs(t->height[y][x]); n++; }
    return n ? sum / n : 0.0;
}
