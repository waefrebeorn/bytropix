/* test_specdec.c -- BRAIN #126 speculative decoding draft/verify.
 * Proves: rejection-sampling acceptance (min(1,p/q)), all-accept -> bonus
 * token, acceptance rate, deterministic with a fixed seed, fail-closed. */
#include "wubu_specdec.h"
#include <stdio.h>
#include <string.h>

static int fails = 0, checks = 0;
#define CHECK(cond, msg) do { checks++; if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } else { printf("  [PASS] %s\n", msg); } } while (0)

/* small vocab helper: make a one-hot-ish distribution over a few tokens */
static void set_dist(float *row, int vocab_size, int hot, float p) {
    for (int v = 0; v < vocab_size; v++) row[v] = (v == hot) ? p : (1.0f - p) / (vocab_size - 1);
}

int main(void) {
    const int K = 4;
    const int V = WUBU_SPECDEC_VOCAB;

    /* ---- case 1: target agrees with draft (p_target high on draft token) ----
     * every draft token has p_target >= p_draft -> always accepted (ratio>=1) */
    wubu_specdec_t sd;
    memset(&sd, 0, sizeof sd);
    sd.draft_len = K;
    for (int i = 0; i < K; i++) sd.draft_tokens[i] = 100 + i;
    for (int i = 0; i < K; i++) {
        set_dist(sd.draft_probs[i], V, 100 + i, 0.5f);
        set_dist(sd.target_probs[i], V, 100 + i, 0.9f);  /* target higher -> accept */
    }
    unsigned seed = 12345;
    int produced = wubu_specdec_verify(&sd, &seed);
    CHECK(produced == K + 1, "all-accept: K draft + 1 bonus token produced");
    CHECK(sd.n_accepted == K, "all K accepted");
    for (int i = 0; i < K; i++) CHECK(sd.accepted[i] == 1, "each draft accepted");
    CHECK(wubu_specdec_rate(&sd) == 1.0f, "acceptance rate = 1.0");
    /* bonus token is drawn from the target distribution over the vocab */
    CHECK(sd.bonus_token >= 0 && sd.bonus_token < V, "bonus token in vocab range");

    /* ---- case 2: deterministic with a fixed seed (same seed -> same result) */
    unsigned seed2 = 12345;
    wubu_specdec_t sd2 = sd;
    memset(sd2.accepted, 0, sizeof sd2.accepted); sd2.n_accepted = 0; sd2.bonus_token = -1;
    int produced2 = wubu_specdec_verify(&sd2, &seed2);
    CHECK(produced2 == produced, "same seed -> same produced count (deterministic)");
    CHECK(sd2.bonus_token == sd.bonus_token, "same seed -> same bonus token");

    /* ---- case 3: target strongly disagrees (p_target ~0 on draft token) ----
     * ratio ~0 -> rejected; only the resampled token is produced */
    wubu_specdec_t sd3;
    memset(&sd3, 0, sizeof sd3);
    sd3.draft_len = K;
    for (int i = 0; i < K; i++) sd3.draft_tokens[i] = 100 + i;
    for (int i = 0; i < K; i++) {
        set_dist(sd3.draft_probs[i], V, 100 + i, 0.5f);
        /* target puts ~0 on the draft token -> accept_p ~0 -> reject */
        set_dist(sd3.target_probs[i], V, 500 + i, 0.99f);
    }
    unsigned seed3 = 777;
    int produced3 = wubu_specdec_verify(&sd3, &seed3);
    CHECK(sd3.n_accepted == 0, "disagreeing target: 0 accepted");
    CHECK(produced3 == 1, "disagreeing: 0 accepted + 1 resampled = 1 token");
    CHECK(sd3.accepted[0] == 0, "first draft rejected");
    CHECK(wubu_specdec_rate(&sd3) == 0.0f, "acceptance rate = 0");

    /* ---- case 4: fail-closed ---- */
    unsigned s = 1;
    CHECK(wubu_specdec_verify(NULL, &s) == -1, "NULL sd -> -1");
    CHECK(wubu_specdec_verify(&sd, NULL) == -1, "NULL seed -> -1");
    wubu_specdec_t bad; memset(&bad, 0, sizeof bad); bad.draft_len = 0;
    CHECK(wubu_specdec_verify(&bad, &s) == -1, "draft_len 0 -> -1");
    CHECK(wubu_specdec_rate(NULL) == 0.0f, "NULL rate -> 0");

    printf("\n=== test_specdec: %d checks, %d failures ===\n", checks, fails);
    return fails ? 1 : 0;
}
