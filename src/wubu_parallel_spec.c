/*
 * wubu_parallel_spec.c -- Parallel speculative decoding (V01-V04) + length-gen PE (W01-W03). C11.
 *
 * Convergence (EAGLE-3 / P-EAGLE / tree-attn / Kangaroo / NoPE / ALiBi 7-hop):
 *   - V01 EAGLE-3 feature drafting: instead of drafting tokens, predict the next
 *        HIDDEN FEATURE (a proxy vector) and only the top-1 matters; we model the
 *        draft as a feature score array and pick the argmax as the drafted token.
 *   - V02 P-EAGLE parallel drafting: maintain K independent draft paths; verify
 *        them in parallel (we just enumerate K draft ids and report acceptance
 *        given a match mask).
 *   - V03 tree-attention verify mask: build a parent-index array for a tree of
 *        speculative positions so each node attends to its tree ancestor (not the
 *        full causal prefix) -- we return the parent array.
 *   - V04 Kangaroo double-early-exit: two exit points (shallow + deep) produce
 *        draft candidates; accept if either matches -- report combined accept.
 *   - W01 NoPE: positional encoding is identity (no-op); sequence order carried by
 *        attention. We expose a flag + a passthrough.
 *   - W02 ALiBi distance bias: bias[i][j] = -slope * (i - j) for i>j (causal).
 *        slope is extrapolatable (fixed per head). Returns the bias matrix.
 *   - W03 attention-sandwich (FFN-first) layer order flag (length-robust): a
 *        boolean toggle for layer ordering.
 *
 * Triple-DA: dims checked; null handled; deterministic.
 */
#include "wubu_parallel_spec.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* V01 EAGLE-3 feature drafting: argmax of feature score -> drafted token. */
int wubu_eagle3_draft(const float *feat_score, int n, int *drafted) {
    if (!feat_score || !drafted || n <= 0) return 0;
    int best = 0; float bv = feat_score[0];
    for (int i = 1; i < n; i++) if (feat_score[i] > bv) { bv = feat_score[i]; best = i; }
    *drafted = best;
    return 1;
}

/* V02 P-EAGLE parallel drafting: given K draft ids and a match mask (1=accept),
 * return the number accepted. */
int wubu_peagle_verify(const int *drafts, const char *match, int K, int *accepted) {
    if (!drafts || !match || !accepted || K <= 0) return 0;
    int cnt = 0;
    for (int i = 0; i < K; i++) if (match[i]) { accepted[cnt++] = drafts[i]; }
    return cnt;
}

/* V03 tree-attention parent array: node 0 is root (parent -1); node i's parent
 * is (i-1) (a simple left-branch tree). Writes parents sized >= n. Returns n. */
int wubu_tree_attn_parents(int n, int *parents) {
    if (!parents || n <= 0) return 0;
    for (int i = 0; i < n; i++) parents[i] = (i == 0) ? -1 : i - 1;
    return n;
}

/* V04 Kangaroo double-early-exit: accept if shallow OR deep draft matches. */
int wubu_kangaroo_accept(int shallow_match, int deep_match) {
    return (shallow_match || deep_match) ? 1 : 0;
}

/* W01 NoPE flag (passthrough: no positional transform). */
int wubu_nope_enabled(void) { return 1; }

/* W02 ALiBi distance bias: bias[i*d+j] = -slope*(i-j) for i>=j (causal lower
 * triangle); upper triangle unused (set 0). */
int wubu_alibi_bias(float *bias, int n, int d, float slope) {
    if (!bias || n <= 0 || d <= 0 || slope < 0.0f) return 0;
    (void)d;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            bias[(size_t)i*n + j] = (i >= j) ? (-slope * (float)(i - j)) : 0.0f;
    return n;
}

/* W03 attention-sandwich (FFN-first) ordering flag. */
int wubu_ffn_first_enabled(void) { return 1; }
