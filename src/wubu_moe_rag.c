/*
 * wubu_moe_rag.c -- MoE routing (X01-X06) + retrieval-augmented KV (Y01-Y04). C11.
 *
 * Convergence (Top-K / Expert-Choice / shared / sigmoid / ExpertFlow / capacity /
 * KV-Packet / RACC / CAG / cross-doc-isolation 7-hop):
 *   - X01 Top-K router: softmax gate over N experts, pick the top-K by score.
 *        Writes selected expert ids (sized >= K).
 *   - X02 Expert-Choice: each expert picks its top-C tokens (balanced load). We
 *        return, per expert, the chosen token ids (flattened; counts in `cnt`).
 *   - X03 shared-expert: routed output (sum of selected experts) += shared
 *        expert output (always on). Returns combined weight mask.
 *   - X04 sigmoid gating: independent P(expert) = sigmoid(score); select experts
 *        with P > thr (variable count). Writes selected ids.
 *   - X05 predictive expert caching: given predicted-expert set and current
 *        cached set, return which experts to prefetch (in predicted, not cached).
 *   - X06 capacity factor: max tokens per expert = cap * (tokens/N); if a chosen
 *        expert exceeds capacity, mark overflow (drop). Returns kept count.
 *   - Y01 KV Packet: a document's KV is context-independent (computed alone) and
 *        reusable. We model a packet id per document; returns the per-token doc id.
 *   - Y02 RACC: given token->is_retrieved flag, keep retrieved tokens' KV (drop
 *        non-retrieved under compression). Returns keep-mask.
 *   - Y03 CAG: preload doc KV once; at query time no retrieval -- we just confirm
 *        the preloaded doc id is "ready" (1) so decode proceeds from cache.
 *   - Y04 cross-document KV isolation: assign each doc a namespace id so its KV is
 *        not cross-attended. Returns the namespace for a token.
 *
 * Triple-DA: dims/zero handled; thresholds clamped; deterministic.
 */
#include "wubu_moe_rag.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* X01 Top-K router. */
int wubu_topk_route(const float *gate, int N, int K, int *sel) {
    if (!gate || !sel || N <= 0 || K <= 0) return 0;
    if (K > N) K = N;
    /* partial selection sort by gate desc */
    int *idx = (int *)malloc((size_t)N * sizeof(int));
    if (!idx) return 0;
    for (int i = 0; i < N; i++) idx[i] = i;
    for (int k = 0; k < K; k++) {
        int best = k;
        for (int j = k + 1; j < N; j++) if (gate[idx[j]] > gate[idx[best]]) best = j;
        int t = idx[k]; idx[k] = idx[best]; idx[best] = t;
        sel[k] = idx[k];
    }
    free(idx);
    return K;
}

/* X02 Expert-Choice: each expert picks its top-C tokens. tokens scores are
 * ntok x N (token-major). Returns per-expert chosen token ids in `out`
 * (sized >= N*C), counts per expert in `cnt` (sized >= N). */
int wubu_expert_choice(const float *score, int ntok, int N, int C,
                        int *out, int *cnt) {
    if (!score || !out || !cnt || ntok <= 0 || N <= 0 || C <= 0) return 0;
    int off = 0;
    for (int e = 0; e < N; e++) {
        int *tidx = (int *)malloc((size_t)ntok * sizeof(int));
        if (!tidx) return off;
        for (int i = 0; i < ntok; i++) tidx[i] = i;
        int keep = C < ntok ? C : ntok;
        for (int k = 0; k < keep; k++) {
            int best = k;
            for (int j = k + 1; j < ntok; j++)
                if (score[(size_t)tidx[j]*N + e] > score[(size_t)tidx[best]*N + e]) best = j;
            int t = tidx[k]; tidx[k] = tidx[best]; tidx[best] = t;
            out[off++] = tidx[k];
        }
        cnt[e] = keep;
        free(tidx);
    }
    return off;
}

/* X03 shared-expert aggregation: out[i] = (routed selected? 1:0) + 1 (shared). */
int wubu_shared_expert(const int *routed, int N, int K, int *out) {
    if (!routed || !out || N <= 0 || K <= 0) return 0;
    for (int i = 0; i < N; i++) out[i] = 0;
    for (int k = 0; k < K && k < N; k++) out[routed[k]] = 1; /* routed selected */
    for (int i = 0; i < N; i++) out[i] += 1;                 /* shared always on */
    return N;
}

/* X04 sigmoid gating: select experts with sigmoid(score) > thr (variable count). */
int wubu_sigmoid_gate(const float *score, int N, float thr, int *sel) {
    if (!score || !sel || N <= 0) return 0;
    if (thr < 0.0f) thr = 0.0f; if (thr > 1.0f) thr = 1.0f;
    int c = 0;
    for (int i = 0; i < N; i++) {
        float p = 1.0f / (1.0f + expf(-score[i]));
        if (p > thr) sel[c++] = i;
    }
    return c;
}

/* X05 predictive expert caching: prefetch = predicted not yet cached. */
int wubu_expert_prefetch(const int *predicted, int np, const char *cached,
                         int N, int *prefetch) {
    if (!predicted || !cached || !prefetch || np <= 0 || N <= 0) return 0;
    int c = 0;
    for (int i = 0; i < np; i++) {
        int e = predicted[i];
        if (e >= 0 && e < N && !cached[e]) prefetch[c++] = e;
    }
    return c;
}

/* X06 capacity factor: per-expert cap = max(1, (int)(cap * ntok / N)). Returns
 * kept count (drops tokens whose expert is over capacity). keep mask sized ntok. */
int wubu_capacity_factor(const int *expert_of, int ntok, int N, float cap,
                         char *keep) {
    if (!expert_of || !keep || ntok <= 0 || N <= 0 || cap <= 0.0f) return 0;
    int limit = (int)(cap * ntok / N);
    if (limit < 1) limit = 1;
    int *used = (int *)calloc((size_t)N, sizeof(int));
    if (!used) return 0;
    int kept = 0;
    for (int i = 0; i < ntok; i++) {
        int e = expert_of[i];
        if (e < 0 || e >= N) { keep[i] = 0; continue; }
        if (used[e] < limit) { keep[i] = 1; used[e]++; kept++; }
        else keep[i] = 0; /* dropped (overflow) */
    }
    free(used);
    return kept;
}

/* Y01 KV Packet doc id per token (default 1 doc -> all doc 0). */
int wubu_kvpacket_doc(const int *tok_doc, int n, int *doc_id) {
    if (!doc_id || n <= 0) return 0;
    if (!tok_doc) { for (int i=0;i<n;i++) doc_id[i]=0; }
    else for (int i=0;i<n;i++) doc_id[i]=tok_doc[i];
    return n;
}

/* Y02 RACC keep-mask: keep retrieved tokens. */
int wubu_racc_keep(const char *is_retrieved, int n, char *keep) {
    if (!is_retrieved || !keep || n <= 0) return 0;
    int c = 0;
    for (int i = 0; i < n; i++) { keep[i] = is_retrieved[i] ? 1 : 0; c += keep[i]; }
    return c;
}

/* Y03 CAG ready flag for a preloaded doc. */
int wubu_cag_ready(int doc_loaded) { return doc_loaded ? 1 : 0; }

/* Y04 cross-document namespace id (== doc id). */
int wubu_crossdoc_ns(const int *tok_doc, int i) {
    if (!tok_doc || i < 0) return 0;
    return tok_doc[i];
}
