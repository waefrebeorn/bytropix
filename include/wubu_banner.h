/* wubu_banner.h — Consistent visual identity for all WuBu CLI tools.
 *
 * Every WuBu CLI tool prints the same styled banner so the user
 * always knows which tool they're looking at. The banner uses
 * the WuBu green accent (#00C853) in terminal ANSI when supported,
 * falling back to plain Unicode box-drawing.
 *
 * C11, no third-party deps, no god headers.
 */
#ifndef WUBU_BANNER_H
#define WUBU_BANNER_H

#include <stdio.h>
#include <stdarg.h>     /* wubu_print_stat() variadic */

/* Centralize visual identity: wubu_tokens.h is the single source of truth
 * for colors, spacing, and layout tokens shared across CLI + GUI + themes.
 * (Adopts research 066-J3: design token file, Theme J rank-3.) */
#include "wubu_tokens.h"

/* Tool name + version string — set at compile time via Makefile. */
#ifndef WUBU_TOOL_NAME
#define WUBU_TOOL_NAME "wubu"
#endif
#ifndef WUBU_TOOL_VERSION
#define WUBU_TOOL_VERSION "dev"
#endif

static inline void wubu_print_banner(const char *tag, const char *extra)
{
    /* Tag is e.g. "Local Inference API Server" or "CLI Runner".
     * Green accent from wubu_tokens.h (single source of truth, J3). */
    printf("\n");
    printf("  %s╔══════════════════════════════════════════════════════╗%s\n", WUBU_TOKEN_GREEN_ANSI, WUBU_TOKEN_RESET);
    printf("  %s║  %-52s%s║\n", WUBU_TOKEN_GREEN_ANSI, WUBU_TOOL_NAME " " WUBU_TOOL_VERSION, WUBU_TOKEN_RESET);
    printf("  ║  %-52s║\n", tag);
    if (extra && extra[0])
        printf("  ║  %-52s║\n", extra);
    printf("  ╚══════════════════════════════════════════════════════╝\n");
}

/* Print a section header used in stats blocks. */
static inline void wubu_print_section(const char *title)
{
    printf("\n── %s ──\n", title);
}

/* Print a key:value stat line with consistent alignment. */
static inline void wubu_print_stat(const char *key, const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    printf("  %-18s ", key);
    vprintf(fmt, ap);
    printf("\n");
    va_end(ap);
}

#endif /* WUBU_BANNER_H */