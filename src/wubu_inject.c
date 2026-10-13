/* wubu_inject.c -- AGI-EDR prompt-injection runtime detector (AGIEDR #296).
 * See wubu_inject.h. Deterministic signal scoring, fail-closed verdict. */
#include "wubu_inject.h"
#include <string.h>
#include <ctype.h>

/* Signal lexicons. Each is a lowercase substring; matching is case-insensitive.
 * Weighted by category. Keep auditable and small. */
static const char *const kOverride[] = {
    "ignore previous", "ignore all previous", "ignore the above",
    "disregard the above", "disregard previous", "forget your instructions",
    "forget everything", "override your instructions", "new instructions",
    "you are now", "act as if", "pretend you are", "from now on you",
    "system:", "system prompt", "### system", "[system]", "<system",
};
static const char *const kExfil[] = {
    "send to", "post to", "curl ", "wget ", "upload to", "exfiltrate",
    "http://", "https://", "base64", "api_key", "api key", "password",
    "secret", "token", "ssh-rsa", "-----begin",
};
static const char *const kEscape[] = {
    "```", "{{", "}}", "}}}", "<|", "|>", "</", "\\n\\n", "---end",
    "===end", "ignore",  /* bare 'ignore' near instructions */
};

/* Case-insensitive substring search. Returns 1 if needle in hay. */
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

static int count_hits(const char *prompt, const char *const *lex, int n) {
    int hits = 0;
    for (int i = 0; i < n; i++) if (icontains(prompt, lex[i])) hits++;
    return hits;
}

double wubu_inject_score(const char *prompt) {
    if (!prompt) return 1.0;   /* fail-closed */
    size_t len = strlen(prompt);
    if (len == 0) return 0.0;  /* empty is benign */

    int ov = count_hits(prompt, kOverride, (int)(sizeof kOverride / sizeof *kOverride));
    int ex = count_hits(prompt, kExfil,    (int)(sizeof kExfil    / sizeof *kExfil));
    int es = count_hits(prompt, kEscape,   (int)(sizeof kEscape   / sizeof *kEscape));

    /* Weighted sum, normalised by a saturating denominator. Override phrases
     * are the strongest signal. */
    double w = 0.45 * ov + 0.30 * ex + 0.15 * es;
    double score = w / 3.0;           /* 3 categories worth of weight */
    /* Density boost: a short prompt dense with signals is more suspicious. */
    if (len < 200 && (ov + ex + es) >= 2) score += 0.15;

    if (score < 0.0) score = 0.0;
    if (score > 1.0) score = 1.0;
    return score;
}

wubu_inject_verdict_t wubu_inject_classify(const char *prompt) {
    wubu_inject_verdict_t v;
    if (!prompt) { v.score = 1.0; v.n_hits = 0; v.is_injection = 1; return v; }
    v.score = wubu_inject_score(prompt);
    v.n_hits = count_hits(prompt, kOverride, (int)(sizeof kOverride / sizeof *kOverride))
             + count_hits(prompt, kExfil,    (int)(sizeof kExfil    / sizeof *kExfil))
             + count_hits(prompt, kEscape,   (int)(sizeof kEscape   / sizeof *kEscape));
    v.is_injection = (v.score >= WUBU_INJECT_THRESHOLD) ? 1 : 0;
    return v;
}

int wubu_inject_is_injection(const char *prompt) {
    return wubu_inject_classify(prompt).is_injection;
}
