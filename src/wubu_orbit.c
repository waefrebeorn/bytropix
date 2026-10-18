/*
 * wubu_orbit.c -- experts as gravitating bodies. See wubu_orbit.h.
 */
#include "wubu_orbit.h"
#include <math.h>
#include <stddef.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

double wubu_orbit_radius(const wubu_orbit_body *b)
{
    if (!b) return -1.0;
    return sqrt(b->x * b->x + b->y * b->y);
}

double wubu_orbit_poincare_dist(double r1, double t1, double r2, double t2)
{
    if (r1 < 0.0 || r2 < 0.0 || r1 >= 1.0 || r2 >= 1.0) return -1.0;
    double x1 = r1 * cos(t1), y1 = r1 * sin(t1);
    double x2 = r2 * cos(t2), y2 = r2 * sin(t2);
    double dx = x1 - x2, dy = y1 - y2;
    double num = dx * dx + dy * dy;
    double den = (1.0 - r1 * r1) * (1.0 - r2 * r2);
    if (den <= 0.0) return -1.0;
    double arg = 1.0 + 2.0 * num / den;
    if (arg < 1.0) arg = 1.0;
    return acosh(arg);
}

double wubu_orbit_potential(const wubu_orbit_body *b, double central_mass,
                            double g, double softening)
{
    if (!b || g < 0.0 || central_mass < 0.0 || softening < 0.0) return 0.0;
    double r = wubu_orbit_radius(b);
    return -g * central_mass * b->mass / (r + softening);
}

double wubu_orbit_period(const wubu_orbit_body *b, double central_mass, double g)
{
    if (!b || g <= 0.0 || central_mass <= 0.0) return -1.0;
    double r = wubu_orbit_radius(b);
    if (r <= 0.0) return 0.0;
    return 2.0 * M_PI * sqrt((r * r * r) / (g * central_mass));
}

int wubu_orbit_step(wubu_orbit_body *bodies, int n, const wubu_orbit_sim *s)
{
    if (!bodies || n <= 0 || n > WUBU_ORBIT_MAX || !s || s->dt <= 0.0) return -1;
    if (s->g < 0.0 || s->central_mass < 0.0) return -1;

    double soft2 = s->softening * s->softening;
    double *ax = (double[WUBU_ORBIT_MAX]){0};
    double *ay = (double[WUBU_ORBIT_MAX]){0};

    /* Central mass. */
    for (int i = 0; i < n; i++) {
        double r2 = bodies[i].x * bodies[i].x + bodies[i].y * bodies[i].y;
        double den = r2 + soft2;
        double inv = 1.0 / (den * sqrt(den));
        ax[i] -= s->g * s->central_mass * bodies[i].x * inv;
        ay[i] -= s->g * s->central_mass * bodies[i].y * inv;
    }

    /* Pairwise attraction between bodies. */
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            double dx = bodies[j].x - bodies[i].x;
            double dy = bodies[j].y - bodies[i].y;
            double den = dx * dx + dy * dy + soft2;
            double inv = 1.0 / (den * sqrt(den));
            double fi = s->g * bodies[j].mass * inv;
            double fj = s->g * bodies[i].mass * inv;
            ax[i] += fi * dx; ay[i] += fi * dy;
            ax[j] -= fj * dx; ay[j] -= fj * dy;
        }
    }

    int escaped = 0;
    for (int i = 0; i < n; i++) {
        bodies[i].vx += ax[i] * s->dt;
        bodies[i].vy += ay[i] * s->dt;
        bodies[i].x += bodies[i].vx * s->dt;
        bodies[i].y += bodies[i].vy * s->dt;

        double r = wubu_orbit_radius(&bodies[i]);
        if (r > s->disk_limit) {
            double k = s->disk_limit / r;
            bodies[i].x *= k; bodies[i].y *= k;
            escaped++;
        }
    }
    return escaped;
}

int wubu_orbit_route(const wubu_orbit_body *bodies, int n, double tx, double ty,
                     double temperature, double g, double softening,
                     double *weights)
{
    if (!bodies || !weights || n <= 0 || n > WUBU_ORBIT_MAX) return -1;
    if (temperature <= 0.0 || g < 0.0 || softening < 0.0) return -1;

    double phi[WUBU_ORBIT_MAX];
    double deepest = 1e300;
    for (int i = 0; i < n; i++) {
        double dx = bodies[i].x - tx, dy = bodies[i].y - ty;
        double d = sqrt(dx * dx + dy * dy + softening * softening);
        phi[i] = -g * bodies[i].mass / d;
        if (phi[i] < deepest) deepest = phi[i];
    }

    /* Boltzmann in potential energy: weight ~ exp(-(phi - phi_min)/T), so the
     * deepest well has weight 1 and every shallower body decays from it. */
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        weights[i] = exp(-(phi[i] - deepest) / temperature);
        sum += weights[i];
    }
    if (sum <= 0.0) return -1;
    for (int i = 0; i < n; i++) weights[i] /= sum;
    return 0;
}

int wubu_orbit_resonance(const wubu_orbit_body *a, const wubu_orbit_body *b,
                         double central_mass, double g, int p, int q,
                         double tol)
{
    if (!a || !b || p <= 0 || q <= 0 || tol < 0.0) return 0;
    double pa = wubu_orbit_period(a, central_mass, g);
    double pb = wubu_orbit_period(b, central_mass, g);
    if (pa <= 0.0 || pb <= 0.0) return 0;
    double ratio = pb / pa;
    double target = (double)p / (double)q;
    return fabs(ratio - target) <= tol;
}

int wubu_orbit_cool(wubu_orbit_body *bodies, int n, double rate, double dt,
                    double floor_r)
{
    if (!bodies || n <= 0 || n > WUBU_ORBIT_MAX) return 0;
    if (rate < 0.0 || dt < 0.0 || floor_r < 0.0) return 0;

    int moved = 0;
    double k = 1.0 - rate * dt;
    if (k < 0.0) k = 0.0;

    for (int i = 0; i < n; i++) {
        double r = wubu_orbit_radius(&bodies[i]);
        if (r <= floor_r) continue;
        double nr = r * k;
        if (nr < floor_r) nr = floor_r;
        if (r > 0.0) {
            bodies[i].x *= nr / r;
            bodies[i].y *= nr / r;
            bodies[i].vx *= nr / r;
            bodies[i].vy *= nr / r;
            moved++;
        }
    }
    return moved;
}