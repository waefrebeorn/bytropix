#ifndef WUBU_QUALITY_FILTER_H
#define WUBU_QUALITY_FILTER_H
/* wubu_quality_filter.h — document quality filter (BRAIN #245).
 *
 * Scores a text document on cheap heuristics and accepts/rejects it against a
 * minimum score, the front line of corpus curation (rejects boilerplate,
 * gibberish, over-numeric spam). Score in [0,1] blends: alphabetic ratio,
 * word-count adequacy, mean-word-length sanity, and vocabulary diversity.
 * The forward (wubu_quality_score + wubu_quality_accept) is covered by
 * test_quality_filter with measured assertions.
 */
#include <stddef.h>

/* Compute a quality score in [0,1] for a NUL-terminated text. */
double wubu_quality_score(const char *text);

/* Accept a document if its score >= min_score. Returns 1 accept / 0 reject. */
int wubu_quality_accept(const char *text, double min_score);

#endif
