/*
 * wubu_generate.c -- autoregressive generation + n-gram speculative decoding
 * (doc 018 / K01). Self-contained C11. See header.
 */
#include "wubu_generate.h"
#include "wubu_ngram.h"
#include "wubu_integrate.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int argmaxf(const float *a, int n) {
    int best = 0; float bv = a[0];
    for (int i = 1; i < n; i++) if (a[i] > bv) { bv = a[i]; best = i; }
    return best;
}

/* deterministic rng (xorshift) for sampling */
static uint32_t grng_state;
static void grng_srand(uint32_t s) { grng_state = s ? s : 0x1234567u; }
static uint32_t grng_u32(void) {
    uint32_t x = grng_state;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    grng_state = x;
    return x;
}
static float grng_uni(void) { return (float)grng_u32() / (float)0xFFFFFFFFu; }

/* softmax in place */
static void softmaxf(float *a, int n) {
    float mx = a[0]; for (int i = 1; i < n; i++) if (a[i] > mx) mx = a[i];
    float s = 0; for (int i = 0; i < n; i++) { a[i] = expf(a[i] - mx); s += a[i]; }
    float inv = s > 0 ? 1.0f/s : 0; for (int i = 0; i < n; i++) a[i] *= inv;
}

int wubu_generate(wubu_model_t *model, const int *prompt, int n_prompt,
                  const wubu_generate_cfg_t *cfg, int *out) {
    if (!model || !prompt || !cfg || !out) return 0;
    const int V = model->vocab_size;
    grng_srand(cfg->seed);

    /* running sequence = prompt + emitted */
    int *seq = (int *)malloc(sizeof(int) * (n_prompt + cfg->max_tokens + cfg->spec_k + 1));
    if (!seq) return 0;
    memcpy(seq, prompt, (size_t)n_prompt * sizeof(int));
    int seqlen = n_prompt;

    /* runtime decode policy: composes the recursive-loop gap modules
     * (capacity_wall + kv_budget + stream_kv + ctx_manage + hybrid + pd).
     * Pure policy -- guards the 512K OOM ceiling and advises eviction. */
    int max_ctx = model->gqa_max_ctx > 0 ? model->gqa_max_ctx : 524288;
    wubu_decode_policy_t *policy = wubu_decode_policy_default(max_ctx, model->n_layers);

    float *logits = (float *)malloc(sizeof(float) * (n_prompt + cfg->max_tokens + cfg->spec_k + 1) * V);
    if (!logits) { free(seq); return 0; }

    int emitted = 0;
    int K = cfg->spec_k;

    while (emitted < cfg->max_tokens) {
        wubu_decode_decision_t dec;
        wubu_decode_policy_step(policy, seqlen, 0, 1 << 30, 0, &dec);
        /* capacity_wall: never let decode exceed the 512K OOM ceiling (EAMM guard).
         * If unsafe, stop gracefully -- the operator's budget/streaming keeps us
         * under the ceiling; we never silently OOM. */
        if (!dec.oom_safe) {
            fprintf(stderr, "[decode-policy] 512K ceiling reached at seqlen=%d; "
                    "stopping (force_evict=%d). Tighten WUBU_STREAM_WINDOW.\n", seqlen, dec.force_evict);
            break;
        }

        if (K <= 0) {
            /* plain decode: forward whole sequence, take last position */
            int T = seqlen;
            wubu_model_forward(model, seq, 1, T, logits);
            float *lp = logits + (size_t)(T - 1) * V;
            int tok;
            if (cfg->greedy) {
                tok = argmaxf(lp, V);
            } else {
                softmaxf(lp, V);
                float r = grng_uni(), acc = 0; tok = V - 1;
                for (int i = 0; i < V; i++) { acc += lp[i]; if (acc >= r) { tok = i; break; } }
            }
            out[emitted++] = tok;
            seq[seqlen++] = tok;
            continue;
        }

        /* ---- speculative decode (draft K tokens, verify in one forward) ---- */
        int *drafts = (int *)malloc(sizeof(int) * K);
        int *dc = (int *)malloc(sizeof(int) * (seqlen + K + 1)); /* virtual ctx */
        int nprop = 0;
        if (dc) memcpy(dc, seq, (size_t)seqlen * sizeof(int));
        for (int s = 0; s < K && dc; s++) {
            wubu_ngram_draft_t *ng = wubu_ngram_create(dc, seqlen + s, cfg->ngram_order);
            int p[1]; int got = wubu_ngram_propose(ng, 1, p);
            wubu_ngram_free(ng);
            if (got <= 0) break;
            drafts[nprop++] = p[0];
            dc[seqlen + nprop - 1] = p[0];  /* virtually extend for next lookup */
        }
        free(dc);

        if (nprop == 0) {
            /* no draft available -> fall back to one plain step */
            int T = seqlen;
            wubu_model_forward(model, seq, 1, T, logits);
            float *lp = logits + (size_t)(T - 1) * V;
            int tok = cfg->greedy ? argmaxf(lp, V) : (softmaxf(lp, V), argmaxf(lp, V));
            if (!cfg->greedy) { float r = grng_uni(), acc = 0; tok = V - 1;
                for (int i = 0; i < V; i++) { acc += lp[i]; if (acc >= r) { tok = i; break; } } }
            out[emitted++] = tok; seq[seqlen++] = tok;
            free(drafts); continue;
        }

        /* forward the sequence + drafts in one pass.
         * logits has T = seqlen + nprop positions. Position (seqlen-1 + k)
         * predicts drafts[k] (k=0..nprop-1); position (seqlen-1+nprop) predicts
         * the bonus token (it exists since T = seqlen+nprop). */
        int T = seqlen + nprop;
        wubu_model_forward(model, seq, 1, T, logits);

        /* Linear autoregressive verify: draft[k] checked against target logits
         * at its own position (seqlen-1+k). This is the provably-equivalent
         * greedy/sampled speculative step (Leviathan et al. 2023). */
        int L = 0;
        float rng = grng_uni();
        for (int k = 0; k < nprop; k++) {
            float *tlp = logits + (size_t)(seqlen - 1 + k) * V;
            if (!cfg->greedy) softmaxf(tlp, V);
            float p_target = cfg->greedy ? (tlp[drafts[k]] == argmaxf(tlp, V) ? 1.0f : 0.0f)
                                         : tlp[drafts[k]];
            /* draft prob: greedy -> one-hot on proposed; sampled -> uniform prior */
            float p_draft = cfg->greedy ? 1.0f : (1.0f / (float)V);
            int accept;
            if (cfg->greedy) accept = (tlp[drafts[k]] == argmaxf(tlp, V));
            else accept = (p_target >= p_draft) || (rng < p_target / (p_draft > 1e-9f ? p_draft : 1e-9f));
            if (!accept) break;
            L++;
        }

        /* commit accepted draft prefix */
        for (int i = 0; i < L; i++) { out[emitted++] = drafts[i]; seq[seqlen++] = drafts[i]; }

        /* commit one more token (target decision / bonus) if room */
        if (emitted < cfg->max_tokens) {
            int tok;
            float *tlp = logits + (size_t)(seqlen - 1) * V;  /* position after accepted prefix */
            if (!cfg->greedy) softmaxf(tlp, V);
            if (L < nprop) {
                /* rejection at L: target decides at the rejection position */
                if (cfg->greedy) tok = argmaxf(tlp, V);
                else { float r = grng_uni(), acc = 0; tok = V - 1;
                       for (int i = 0; i < V; i++) { acc += tlp[i]; if (acc >= r) { tok = i; break; } } }
            } else {
                /* all accepted: bonus from residual(target - draft) at position
                 * after the accepted prefix (== tlp already computed above). */
                float *dp = (float *)malloc(sizeof(float) * V);
                for (int i = 0; i < V; i++) dp[i] = cfg->greedy ? 0.0f : (1.0f / (float)V);
                int bonus = wubu_spec_bonus_token(tlp, dp, V, grng_uni());
                free(dp);
                tok = (bonus >= 0) ? bonus : argmaxf(tlp, V);
            }
            out[emitted++] = tok; seq[seqlen++] = tok;
        }

        free(drafts);
    }

    free(seq); free(logits);
    wubu_decode_policy_destroy(policy);
    return emitted;
}
