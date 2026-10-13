/* wubu_indirect_pi.c -- AGIEDR #304 indirect prompt-injection defense.
 * See wubu_indirect_pi.h. Guards the READ path (tool outputs, retrieved docs). */
#include "wubu_indirect_pi.h"
#include <string.h>

/* Malicious lexicons: imperative overrides + hidden-instruction markers that
 * only appear in injection payloads hidden in content. */
static const char *const kMal[] = {
    "ignore previous instructions", "ignore all previous",
    "disregard the above", "disregard your instructions",
    "you are now", "act as if you", "new system prompt",
    "system: you must", "assistant: ignore", "override your",
    "forget your instructions", "do not tell the user",
    "secretly", "without telling the user", "exfiltrate",
    "send the conversation to", "<!--", "-->",
};
/* Suspect lexicons: softer signals (instructions addressed to an agent). */
static const char *const kSus[] = {
    "instructions:", "you must", "you should", "as an ai",
    "your task is", "respond with", "output only", "do not reveal",
};

static int icontains(const char *hay, const char *needle) {
    size_t nl = strlen(needle);
    if (nl == 0) return 1;
    for (const char *p = hay; *p; p++) {
        size_t i = 0;
        while (i < nl && p[i]) {
            char a = p[i], b = needle[i];
            if (a >= 'A' && a <= 'Z') a += 32;
            if (b >= 'A' && b <= 'Z') b += 32;
            if (a != b) break;
            i++;
        }
        if (i == nl) return 1;
    }
    return 0;
}
static int count_hits(const char *s, const char *const *lex, int n) {
    int h = 0;
    for (int i = 0; i < n; i++) if (icontains(s, lex[i])) h++;
    return h;
}

double wubu_indirect_pi_score(const char *content) {
    if (!content) return 1.0;   /* fail-closed */
    int mal = count_hits(content, kMal, (int)(sizeof kMal / sizeof *kMal));
    int sus = count_hits(content, kSus, (int)(sizeof kSus / sizeof *kSus));
    double score = 0.5 * mal + 0.15 * sus;
    if (score < 0) score = 0;
    if (score > 1) score = 1;
    return score;
}

wubu_ipi_verdict_t wubu_indirect_pi_scan(const char *content) {
    if (!content) return WUBU_IPI_MALICIOUS;   /* fail-closed */
    int mal = count_hits(content, kMal, (int)(sizeof kMal / sizeof *kMal));
    int sus = count_hits(content, kSus, (int)(sizeof kSus / sizeof *kSus));
    if (mal > 0) return WUBU_IPI_MALICIOUS;
    if (sus >= 2) return WUBU_IPI_SUSPECT;
    return WUBU_IPI_CLEAN;
}

int wubu_indirect_pi_block(const char *content) {
    return wubu_indirect_pi_scan(content) == WUBU_IPI_MALICIOUS ? 1 : 0;
}
