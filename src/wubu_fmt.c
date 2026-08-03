/* wubu_fmt.c -- the format-constraint reward checker. */
#include <stdio.h>
#include <string.h>
#include "wubu_fmt.h"

static int json_ok(const char *out)
{
    /* a light JSON validator: the output must start with { or [ and the
     * braces/brackets must balance (the strings skipped) */
    if (!out) return 0;
    const char *q = out;
    while (*q == ' ' || *q == '\t' || *q == '\n' || *q == '\r') q++;
    if (*q != '{' && *q != '[') return 0;   /* must START with the bracket */
    int depth = 0, in_str = 0, esc = 0, seen = 0;
    for (const char *p = out; *p; p++) {
        if (in_str) {
            if (esc) esc = 0;
            else if (*p == '\\') esc = 1;
            else if (*p == '"') in_str = 0;
            continue;
        }
        if (*p == '"') { in_str = 1; seen = 1; continue; }
        if (*p == '{' || *p == '[') { depth++; seen = 1; }
        else if (*p == '}' || *p == ']') { depth--; seen = 1; }
        if (depth < 0) return 0;
    }
    return seen && depth == 0 && !in_str;
}

int wubu_fmt_check(int type, const char *out, int limit, const char *extra)
{
    if (!out) return 0;
    switch (type) {
    case WUBU_FMT_JSON:
        return json_ok(out);
    case WUBU_FMT_THINK: {
        const char *o = strstr(out, "<think>");
        const char *c = strstr(out, "</think>");
        if (!o || !c || c < o) return 0;
        return strstr(c + 8, "</think>") == NULL;   /* exactly one close */
    }
    case WUBU_FMT_LEN_MAX:
        return (int)strlen(out) <= limit;
    case WUBU_FMT_LEN_MIN:
        return (int)strlen(out) >= limit;
    case WUBU_FMT_PREFIX:
        return extra && strncmp(out, extra, strlen(extra)) == 0;
    }
    return 0;
}

float wubu_fmt_reward(const int *types, int n, const char *out,
                      const int *limits, const char **extras)
{
    if (!types || n < 1 || !out) return 0;
    int held = 0;
    for (int i = 0; i < n; i++)
        if (wubu_fmt_check(types[i], out, limits ? limits[i] : 0,
                           extras ? extras[i] : NULL)) held++;
    return (float)held / (float)n;
}
