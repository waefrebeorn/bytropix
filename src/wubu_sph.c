/* wubu_sph.c -- WORLDSIM #265 fluid simulation (SPH).
 * See wubu_sph.h. */
#include "wubu_sph.h"
#include <math.h>

int wubu_sph_init(wubu_sph_t *s, double h, double mass) {
    if (!s || h <= 0.0 || mass <= 0.0) return -1;
    s->n = 0;
    s->h = h;
    s->mass = mass;
    return 0;
}

int wubu_sph_add(wubu_sph_t *s, double x, double y) {
    if (!s || s->n >= WUBU_SPH_MAX) return -1;
    wubu_sph_particle_t *p = &s->p[s->n];
    p->x = x; p->y = y; p->vx = 0; p->vy = 0;
    p->density = 0; p->pressure = 0;
    return s->n++;
}

/* Poly6 kernel (2D): W(r) = 4/(pi h^8) (h^2-r^2)^3 for r<h. */
double wubu_sph_kernel(double r, double h) {
    if (r < 0 || r >= h) return 0.0;
    double h2 = h * h, r2 = r * r;
    double diff = h2 - r2;
    double norm = 4.0 / (M_PI * pow(h, 8));
    return norm * diff * diff * diff;
}

int wubu_sph_compute_density(wubu_sph_t *s) {
    if (!s) return -1;
    for (int i = 0; i < s->n; i++) {
        double dens = 0.0;
        for (int j = 0; j < s->n; j++) {
            double dx = s->p[i].x - s->p[j].x, dy = s->p[i].y - s->p[j].y;
            double r = sqrt(dx*dx + dy*dy);
            dens += s->mass * wubu_sph_kernel(r, s->h);
        }
        s->p[i].density = dens;
        /* Tait equation of state: P = k*(rho - rho0) */
        s->p[i].pressure = WUBU_SPH_STIFFNESS * (dens - WUBU_SPH_REST_DENSITY);
        if (s->p[i].pressure < 0) s->p[i].pressure = 0;   /* no tension */
    }
    return 0;
}

int wubu_sph_step(wubu_sph_t *s, double dt) {
    if (!s || dt <= 0.0) return -1;
    wubu_sph_compute_density(s);
    /* pressure + viscosity forces */
    for (int i = 0; i < s->n; i++) {
        double fx = 0, fy = 0;
        for (int j = 0; j < s->n; j++) {
            if (i == j) continue;
            double dx = s->p[j].x - s->p[i].x, dy = s->p[j].y - s->p[i].y;
            double r = sqrt(dx*dx + dy*dy);
            if (r >= s->h || r < 1e-9) continue;
            /* pressure force: push from high to low pressure */
            double pavg = 0.5 * (s->p[i].pressure + s->p[j].pressure);
            double ux = dx / r, uy = dy / r;
            double pf = -pavg / (s->p[i].density > 0 ? s->p[i].density : 1.0);
            fx += pf * ux; fy += pf * uy;
            /* viscosity: damp relative velocity */
            double rvx = s->p[j].vx - s->p[i].vx, rvy = s->p[j].vy - s->p[i].vy;
            double vf = WUBU_SPH_VISCOSITY / (s->p[i].density > 0 ? s->p[i].density : 1.0);
            fx += vf * rvx; fy += vf * rvy;
        }
        s->p[i].vx += fx * dt;
        s->p[i].vy += fy * dt;
    }
    /* integrate positions */
    for (int i = 0; i < s->n; i++) {
        s->p[i].x += s->p[i].vx * dt;
        s->p[i].y += s->p[i].vy * dt;
    }
    return 0;
}

double wubu_sph_ke(const wubu_sph_t *s) {
    if (!s) return 0.0;
    double ke = 0.0;
    for (int i = 0; i < s->n; i++)
        ke += 0.5 * (s->p[i].vx*s->p[i].vx + s->p[i].vy*s->p[i].vy);
    return ke;
}
