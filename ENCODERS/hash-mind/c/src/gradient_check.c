/**
 * gradient_check.c — Compare analytical backward pass against numerical gradients
 */
#include "tokenizer.h"
#include "hashmind_model.h"
#include "rolling_hash.h"
#include <stdio.h>
#include <math.h>

static float eps = 1e-3f;  /* larger ε for noisy gradients */

int main() {
    Tokenizer tok;
    tokenizer_init(&tok);

    HashMindModel model;
    HashMindGrad grad;
    hashmind_model_init(&model);
    hashmind_zero_grad(&grad);

    const char* text = "hello world, this is a test";
    int tokens[32];
    int n = tokenizer_encode(&tok, text, tokens, 32);
    uint32_t hashes[32];
    int nh;
    rolling_hash_all(tokens, n, 3, hashes, &nh);

    int ctx_len = n - 1;
    if (ctx_len > CONTEXT_LEN) ctx_len = CONTEXT_LEN;

    float logits[VOCAB_SIZE];
    BlockActs acts;
    hashmind_forward(&model, hashes, ctx_len, tokens, ctx_len, logits, &acts);
    int target = tokens[ctx_len];

    printf("Test: context='");
    for (int i = 0; i < ctx_len; i++) printf("%c", tok.idx_to_char[tokens[i]]);
    printf("', target='%c' (idx=%d)\n", tok.idx_to_char[target], target);

    float loss = nn_cross_entropy_loss(logits, target, VOCAB_SIZE);
    printf("Loss: %.6f\n", loss);

    /* Analytical backward */
    float dlogits[VOCAB_SIZE];
    nn_cross_entropy_grad(logits, target, VOCAB_SIZE, dlogits);
    hashmind_backward(&model, &grad, dlogits, hashes, ctx_len, tokens, ctx_len, &acts);

    float* gp = (float*)&grad;
    float* mp = (float*)&model;
    long nparams = hashmind_param_count();

    /* Check scattered weights */
    int check_pos[] = {
        0, 1, 2,                                         /* token_embed[0][0..2] */
        D_MODEL, D_MODEL+1,                               /* token_embed[1][0..1] */
        VOCAB_SIZE * D_MODEL,                             /* hash_projector[0] */
        VOCAB_SIZE * D_MODEL + D_MODEL + 10,              /* block 0, qkv_w[0][10] */
        VOCAB_SIZE * D_MODEL + D_MODEL + 100 + D_MODEL*D_MODEL*3,  /* block 0, ffn1_w */
        VOCAB_SIZE * D_MODEL + D_MODEL +                    /* end of block 0 */
            100 + D_MODEL*D_MODEL*3 + D_MODEL*D_FF + D_FF*D_MODEL + D_MODEL*4,
        nparams - 1,                                       /* last param */
    };
    int n_check = sizeof(check_pos) / sizeof(check_pos[0]);

    printf("\n%-8s %-15s %-15s %-8s\n", "Idx", "Analytical", "Numerical", "Status");
    int pass = 1;
    for (int ci = 0; ci < n_check; ci++) {
        int idx = check_pos[ci];
        if (idx < 0 || idx >= nparams) continue;

        float analytical = gp[idx];
        float orig = mp[idx];

        mp[idx] = orig + eps;
        BlockActs ap; float logits_p[VOCAB];
        hashmind_forward(&model, hashes, ctx_len, tokens, ctx_len, logits_p, &ap);
        float loss_p = nn_cross_entropy_loss(logits_p, target, VOCAB);

        mp[idx] = orig - eps;
        BlockActs am; float logits_m[VOCAB];
        hashmind_forward(&model, hashes, ctx_len, tokens, ctx_len, logits_m, &am);
        float loss_m = nn_cross_entropy_loss(logits_m, target, VOCAB);

        mp[idx] = orig;

        float numerical = (loss_p - loss_m) / (2.0f * eps);
        float diff = fabsf(analytical - numerical);
        float max_val = fmaxf(fabsf(analytical), fabsf(numerical));
        float ratio = (max_val > 1e-8f) ? diff / max_val : diff;
        int ok = ratio < 0.1f;

        printf("%-8d %-15.6f %-15.6f %s\n", idx, analytical, numerical,
               ok ? "OK" : "FAIL");
        if (!ok) { pass = 0; printf("  diff=%.6f ratio=%.6f\n", diff, ratio); }
    }

    printf("\nOverall: %s\n", pass ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED");
    return pass ? 0 : 1;
}
