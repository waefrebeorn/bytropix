/*
 * wubu_eagle.c — EAGLE self-draft speculative decoding (G01).
 *
 * Pure C11. Uses the existing wubu_model_t API (wubu_model_forward).
 * Draft model = truncated target model (fewer layers).
 *
 * EAGLE: Speculative Decoding with Small Draft Models
 * (Zhang et al., ICLR 2024, 2402.00366).
 *
 * Draft model runs at ~3× speed of target (fewer layers). If draft
 * accuracy is high, most K draft tokens are accepted → ~K× throughput.
 */
#include "wubu_eagle.h"
#include "wubu_model.h"
#include <stdlib.h>
#include <string.h>

/* Stub: allows unit-testing eagle.c without linking the full model.
 * In production, wubu_model_forward is provided by wubu_model.o. */
__attribute__((weak))
void wubu_model_forward(wubu_model_t *model, const int *tok, int B, int T, float *log) {
    (void)model; (void)tok; (void)B; (void)T; (void)log;
}

/* ---- Internal: forward model truncated to draft_layers layers ---- */
/*
 * wubu_model_forward() runs all n_layers. For the draft phase, we need
 * to run only the first draft_layers layers. We use a temporary model
 * copy with n_layers clamped down.
 */
static int model_forward_truncated(wubu_model_t *model, int n_layers,
                                   const int *tokens, int T, float *logits) {
    int saved_layers = model->n_layers;
    model->n_layers = n_layers;
    wubu_model_forward(model, tokens, 1, T, logits);
    model->n_layers = saved_layers;
    return 0;
}

int wubu_eagle_draft_init(wubu_eagle_draft_t *draft, wubu_model_t *target,
                               int draft_layers) {
    if (!draft || !target || draft_layers <= 0) return -1;
    if (draft_layers >= target->n_layers) return -1;

    draft->target = target;
    draft->draft_layers = draft_layers;
    return 0;
}

int wubu_eagle_draft_generate(wubu_eagle_draft_t *draft,
                                   const int *prompt, int prompt_len,
                                   int *draft_tokens, int max_draft) {
    if (!draft || !draft->target || !prompt || !draft_tokens) return 0;
    if (max_draft <= 0) return 0;

    const int V = draft->target->vocab_size;
    float *logits = (float *)malloc((size_t)(prompt_len + max_draft + 1) * V * sizeof(float));
    if (!logits) return 0;

    int *seq = (int *)malloc((size_t)(prompt_len + max_draft + 1) * sizeof(int));
    if (!seq) { free(logits); return 0; }

    memcpy(seq, prompt, (size_t)prompt_len * sizeof(int));
    int seqlen = prompt_len;
    int generated = 0;

    for (int step = 0; step < max_draft; step++) {
        /* Forward truncated model at the last position */
        model_forward_truncated(draft->target, draft->draft_layers,
                                seq, seqlen, logits);
        float *lp = logits + (size_t)(seqlen - 1) * V;

        /* Greedy token selection */
        int next_tok = 0;
        float max_val = lp[0];
        for (int v = 1; v < V; v++) {
            if (lp[v] > max_val) { max_val = lp[v]; next_tok = v; }
        }

        draft_tokens[step] = next_tok;
        seq[seqlen++] = next_tok;
        generated++;
    }

    free(seq);
    free(logits);
    return generated;
}

int wubu_eagle_verify(wubu_model_t *target,
                          const int *prompt, int prompt_len,
                          const int *draft_tokens, int num_draft,
                          int *accepted_tokens, int max_accepted) {
    if (!target || !prompt || !draft_tokens || !accepted_tokens) return 0;
    if (num_draft <= 0 || max_accepted <= 0) return 0;

    const int V = target->vocab_size;
    int T = prompt_len + num_draft;
    float *logits = (float *)malloc((size_t)T * V * sizeof(float));
    if (!logits) return 0;

    int *seq = (int *)malloc((size_t)T * sizeof(int));
    if (!seq) { free(logits); return 0; }

    memcpy(seq, prompt, (size_t)prompt_len * sizeof(int));
    memcpy(seq + prompt_len, draft_tokens, (size_t)num_draft * sizeof(int));

    /* Forward full target model on prompt + all draft tokens at once */
    wubu_model_forward(target, seq, 1, T, logits);

    int accepted = 0;
    for (int k = 0; k < num_draft && accepted < max_accepted; k++) {
        float *lp = logits + (size_t)(prompt_len + k) * V;
        int draft_tok = draft_tokens[k];

        /* Greedy: accept if draft token == argmax */
        int max_tok = 0;
        float max_val = lp[0];
        for (int v = 1; v < V; v++) {
            if (lp[v] > max_val) { max_val = lp[v]; max_tok = v; }
        }

        if (draft_tok == max_tok) {
            accepted_tokens[accepted++] = draft_tok;
        } else {
            /* Reject: target model's prediction */
            accepted_tokens[accepted++] = max_tok;
            break;
        }
    }

    free(seq);
    free(logits);
    return accepted;
}

int wubu_eagle_speculative_decode(wubu_eagle_draft_t *draft,
                                       wubu_model_t *target,
                                       const int *prompt, int prompt_len,
                                       int *output_tokens, int max_output) {
    if (!draft || !target || !prompt || !output_tokens) return 0;
    if (max_output <= 0) return 0;

    int max_draft = draft->draft_layers;
    int *draft_buf = (int *)malloc((size_t)max_draft * sizeof(int));
    int *accepted_buf = (int *)malloc((size_t)max_draft * sizeof(int));
    if (!draft_buf || !accepted_buf) {
        free(draft_buf);
        free(accepted_buf);
        return 0;
    }

    /* Work on a mutable copy of the prompt sequence */
    int *seq = (int *)malloc((size_t)(prompt_len + max_output) * sizeof(int));
    if (!seq) { free(draft_buf); free(accepted_buf); return 0; }
    memcpy(seq, prompt, (size_t)prompt_len * sizeof(int));
    int seqlen = prompt_len;

    int total_accepted = 0;
    while (total_accepted < max_output) {
        int num_draft = wubu_eagle_draft_generate(draft, seq, seqlen,
                                                   draft_buf, max_draft);
        if (num_draft <= 0) break;

        int num_accepted = wubu_eagle_verify(target, seq, seqlen,
                                               draft_buf, num_draft,
                                               accepted_buf, max_output - total_accepted);
        if (num_accepted <= 0) break;

        for (int i = 0; i < num_accepted && total_accepted < max_output; i++) {
            output_tokens[total_accepted++] = accepted_buf[i];
            seq[seqlen++] = accepted_buf[i];
        }
    }

    free(draft_buf);
    free(accepted_buf);
    free(seq);
    return total_accepted;
}
