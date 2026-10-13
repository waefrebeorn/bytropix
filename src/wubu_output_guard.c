/* wubu_output_guard.c -- AGIEDR #306 LLM-output-as-untrusted validation.
 * See wubu_output_guard.h. */
#include "wubu_output_guard.h"
#include <string.h>

/* Hard dangers -> REJECT (never usable). Soft dangers -> SANITIZE. */
static int has_hard(const char *s) {
    /* command substitution / exec / traversal */
    return strstr(s, "$(") || strstr(s, "`") || strstr(s, "&&") ||
           strstr(s, "||") || strstr(s, ";rm ") || strstr(s, "../") ||
           strstr(s, "/etc/passwd") || strstr(s, "/etc/shadow");
}
static int count_soft(const char *s) {
    int n = 0;
    for (const char *p = s; *p; p++) {
        if (*p == '|' || *p == ';' || *p == '>' || *p == '<' ||
            *p == '&' || *p == '$' || *p == '\\')
            n++;
    }
    /* credential-ish patterns */
    if (strstr(s, "api_key") || strstr(s, "password") || strstr(s, "secret") ||
        strstr(s, "token") || strstr(s, "BEGIN RSA") || strstr(s, "BEGIN PRIVATE"))
        n += 2;
    return n;
}

wubu_out_result_t wubu_output_guard_check(const char *output, wubu_out_policy_t policy) {
    wubu_out_result_t r = { WUBU_OUT_PASS, 0 };
    if (!output) { r.verdict = WUBU_OUT_REJECT; return r; }   /* fail-closed */
    if (has_hard(output)) { r.verdict = WUBU_OUT_REJECT; r.n_dangers = 1; return r; }
    int soft = count_soft(output);
    r.n_dangers = soft;
    if (soft > 0 && policy == WUBU_OUT_STRICT) r.verdict = WUBU_OUT_SANITIZE;
    else r.verdict = WUBU_OUT_PASS;
    return r;
}

int wubu_output_guard_sanitize(const char *output, char *dst, size_t dstsz) {
    if (!output || !dst || dstsz == 0) return -1;
    size_t j = 0;
    for (const char *p = output; *p && j + 1 < dstsz; p++) {
        unsigned char c = (unsigned char)*p;
        /* drop shell metacharacters and control chars; keep alnum + safe punct */
        if (c < 0x20 || c == 0x7f) continue;   /* control chars */
        if (c == '|' || c == ';' || c == '`' || c == '$' || c == '\\') continue;
        dst[j++] = (char)c;
    }
    dst[j] = '\0';
    return (int)j;
}
