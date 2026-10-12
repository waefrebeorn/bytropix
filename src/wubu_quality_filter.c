/* wubu_quality_filter.c — document quality filter (BRAIN #245).
 * C11, self-contained. Heuristic score in [0,1]: alphabetic ratio, word-count
 * adequacy, mean word length sanity, vocabulary diversity (unique/total words).
 */
#include "wubu_quality_filter.h"
#include <ctype.h>
#include <string.h>
#include <math.h>

static double clamp01(double x) { return x < 0 ? 0 : (x > 1 ? 1 : x); }

double wubu_quality_score(const char *text) {
    if (!text || !*text) return 0.0;

    long alpha = 0, alnum = 0, chars = 0, words = 0, vowels = 0;
    int in_word = 0;
    /* Track unique words crudely via a small hash set of first-3-chars. */
    unsigned char seen[256]; memset(seen, 0, sizeof(seen));
    long unique = 0;

    for (const char *p = text; *p; p++) {
        unsigned char c = (unsigned char)*p;
        chars++;
        if (isalpha(c)) { alpha++; if (strchr("aeiouAEIOU", c)) vowels++; }
        if (isalnum(c)) alnum++;
        if (isspace(c)) {
            in_word = 0;
        } else if (!in_word) {
            in_word = 1;
            words++;
            /* crude diversity key: first char folded */
            unsigned char k = (unsigned char)tolower(c);
            if (!seen[k]) { seen[k] = 1; unique++; }
        }
    }
    if (words == 0 || chars == 0) return 0.0;

    double alpha_ratio = (double)alpha / (double)chars;          /* ~0.7 for prose */
    double word_adeq    = clamp01((double)words / 20.0);          /* want >=20 words */
    double mean_wl      = (double)alnum / (double)words;          /* ~5 for prose */
    double wl_sanity    = clamp01(1.0 - fabs(mean_wl - 5.0) / 10.0);
    double vowel_ratio  = (double)vowels / (double)(alpha ? alpha : 1); /* ~0.38 */
    double vowel_ok     = clamp01(1.0 - fabs(vowel_ratio - 0.38) / 0.5);
    double diversity    = clamp01((double)unique / (double)(words > 26 ? 26 : words));

    /* Weighted blend. Real prose scores high; gibberish/boilerplate low. */
    double score = 0.30 * alpha_ratio
                 + 0.20 * word_adeq
                 + 0.20 * wl_sanity
                 + 0.15 * vowel_ok
                 + 0.15 * diversity;
    return clamp01(score);
}

int wubu_quality_accept(const char *text, double min_score) {
    return wubu_quality_score(text) >= min_score ? 1 : 0;
}
