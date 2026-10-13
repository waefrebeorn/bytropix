/* wubu_ns_world.c -- WORLDSIM #262 worldsim state at /n/world.
 * See wubu_ns_world.h. Renders live world state as 9P file content. */
#include "wubu_ns_world.h"
#include <stdio.h>
#include <string.h>

int wubu_ns_world_read(const wubu_world_t *w, const char *relpath,
                       char *out, size_t outsz) {
    if (!w || !relpath || !out || outsz == 0) return -1;

    if (strcmp(relpath, "steps") == 0)
        return snprintf(out, outsz, "%llu\n", (unsigned long long)w->steps);
    if (strcmp(relpath, "bodies") == 0)
        return snprintf(out, outsz, "%d\n", w->n_bodies);
    if (strcmp(relpath, "ke") == 0)
        return snprintf(out, outsz, "%.6f\n", wubu_world_ke(w));
    if (strcmp(relpath, "dt") == 0)
        return snprintf(out, outsz, "%.9f\n", w->clock.dt);
    if (strcmp(relpath, "restitution") == 0)
        return snprintf(out, outsz, "%.6f\n", w->restitution);
    if (strcmp(relpath, "gravity") == 0)
        return snprintf(out, outsz, "%.6f\n", w->forces.gravity);

    /* per-body: body/{i}/{field} */
    if (strncmp(relpath, "body/", 5) == 0) {
        const char *p = relpath + 5;
        char *slash = strchr(p, '/');
        if (!slash) return -1;
        /* parse index */
        int idx = 0;
        for (const char *q = p; q < slash; q++) {
            if (*q < '0' || *q > '9') return -1;
            idx = idx * 10 + (*q - '0');
        }
        const wubu_body_t *b = wubu_world_body(w, idx);
        if (!b) return -1;
        const char *field = slash + 1;
        if (strcmp(field, "x") == 0)        return snprintf(out, outsz, "%.6f\n", b->x);
        if (strcmp(field, "y") == 0)        return snprintf(out, outsz, "%.6f\n", b->y);
        if (strcmp(field, "vx") == 0)       return snprintf(out, outsz, "%.6f\n", b->vx);
        if (strcmp(field, "vy") == 0)       return snprintf(out, outsz, "%.6f\n", b->vy);
        if (strcmp(field, "mass") == 0)     return snprintf(out, outsz, "%.6f\n", b->mass);
        if (strcmp(field, "radius") == 0)   return snprintf(out, outsz, "%.6f\n", b->radius);
        return -1;
    }
    return -1;   /* unknown /n/world file */
}
