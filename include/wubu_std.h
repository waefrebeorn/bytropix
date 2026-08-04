/*
 * wubu_std.h -- the self-contained C11 compatibility surface.
 *
 * The user's directive (2026-08-04): we keep having to add
 * `_GNU_SOURCE` to get M_PI / strdup / CPU macros — a GNU feature-test
 * macro that imports a dependency surface (and its license framing)
 * we don't want. The endgame is our own compiler (wubuos HolyC) where
 * WE define the feature surface. Until then, this header provides the
 * tiny set of helpers WITHOUT _GNU_SOURCE:
 *
 *   - M_PI / M_PI_2 (a constant — just define it, no feature macro)
 *   - wubu_strdup (4 lines, no GNU dependency)
 *   - wubu_fmaxf etc. if ever needed
 *
 * Include THIS instead of reaching for _GNU_SOURCE. The only places
 * that still legitimately need _GNU_SOURCE are Linux-kernel-API calls
 * (CPU_ZERO/CPU_SET affinity) — those are scheduling features, not
 * license hooks, and they stay localized.
 */
#ifndef WUBU_STD_H
#define WUBU_STD_H

#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif
#ifndef M_PI_2
#define M_PI_2 1.57079632679489661923132169163975144
#endif
#ifndef M_E
#define M_E 2.71828182845904523536028747135266250
#endif

/* strdup without _GNU_SOURCE (POSIX strdup is behind feature macros;
 * this is 4 lines and dependency-free). */
static inline char *wubu_strdup(const char *s)
{
    if (!s) return NULL;
    size_t n = strlen(s) + 1;
    char *c = (char *)malloc(n);
    if (c) memcpy(c, s, n);
    return c;
}

#endif /* WUBU_STD_H */
