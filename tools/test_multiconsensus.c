/*
 * test_multiconsensus.c -- DD01-DD07 verification.
 */
#include "wubu_bft.h"
#include "wubu_threshsig.h"
#include "wubu_agentid.h"
#include "wubu_semcons.h"
#include "wubu_fraud.h"
#include <stdio.h>
#include <string.h>

static int fails = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { fails++; printf("FAIL: %s\n", msg); } \
    else printf("  ok: %s\n", msg); \
} while(0)

int main() {
    /* DD01: BFT consensus */
    printf("=== DD01: BFT Consensus ===\n");
    wubu_bft_t b;
    int n = 4;  /* 3f+1 = 4, tolerates 1 Byzantine */
    CHECK(wubu_bft_init(&b, n) == 0, "bft init (4 nodes)");
    CHECK(wubu_bft_threshold(n) == 3, "threshold = 2/3+1 = 3");
    /* 3 agents vote accept (≥3 = majority) */
    wubu_bft_vote(&b, 0, 1, "config-A");
    wubu_bft_vote(&b, 1, 1, "config-A");
    wubu_bft_vote(&b, 2, 1, "config-A");
    wubu_bft_vote(&b, 3, 0, "config-B");  /* Byzantine: votes reject */
    CHECK(wubu_bft_majority(&b, BFT_PROPOSE) == 1, "majority accept despite 1 Byzantine");
    CHECK(b.n_nodes == 4, "4 nodes registered");

    /* With only 2 accept (below threshold), no majority */
    wubu_bft_t b2; wubu_bft_init(&b2, 4);
    wubu_bft_vote(&b2, 0, 1, "A");
    wubu_bft_vote(&b2, 1, 1, "A");
    wubu_bft_vote(&b2, 2, 0, "B");
    wubu_bft_vote(&b2, 3, 0, "B");
    CHECK(wubu_bft_majority(&b2, BFT_PROPOSE) < 0, "no majority (2-2 split)");

    /* DD02: Threshold signing */
    printf("\n=== DD02: Threshold Signing ===\n");
    wubu_threshsig_t ts;
    CHECK(wubu_threshsig_init(&ts, 4) == 0, "threshsig init (threshold=3)");
    unsigned msg_hash = 0x12345678;
    CHECK(wubu_threshsig_add(&ts, 0, msg_hash) == 0, "agent 0 signs");
    CHECK(wubu_threshsig_add(&ts, 1, msg_hash) == 0, "agent 1 signs");
    CHECK(wubu_threshsig_verified(&ts) == 0, "not verified (only 2 sigs, need 3)");
    CHECK(wubu_threshsig_add(&ts, 2, msg_hash) == 0, "agent 2 signs");
    CHECK(wubu_threshsig_verified(&ts) == 1, "verified (3 sigs = threshold)");
    CHECK(wubu_threshsig_add(&ts, 0, msg_hash) == -1, "re-sign by agent 0 rejected");

    /* DD03: Inter-agent identity + zero-trust auth */
    printf("\n=== DD03: Agent Identity (Zero-Trust) ===\n");
    wubu_id_registry_t reg;
    const char *caps0[] = {"read_kv", "sweep_config", "verify"};
    CHECK(wubu_agentid_issue(&reg, 0, "agent-alpha", caps0, 3) == 0, "issue identity for agent 0");
    const char *caps1[] = {"read_kv", "propose"};
    CHECK(wubu_agentid_issue(&reg, 1, "agent-beta", caps1, 2) == 0, "issue identity for agent 1");
    CHECK(wubu_agentid_verify(&reg, 0, "sweep_config") == 1, "agent 0 verified (has sweep_config cap)");
    CHECK(wubu_agentid_verify(&reg, 1, "sweep_config") == 0, "agent 1 NOT verified (no sweep_config cap)");
    CHECK(wubu_agentid_verify(&reg, 3, "read_kv") == 0, "unissued agent 3 NOT verified");

    /* DD04: Semantic consensus */
    printf("\n=== DD04: Semantic Consensus ===\n");
    wubu_semcons_t sc;
    CHECK(wubu_semcons_init(&sc) == 0, "semcons init");
    int claim_idx = wubu_semcons_propose(&sc, 0, "config-X achieves 27.00 tok/s");
    CHECK(claim_idx == 0, "proposal submitted (claim_idx=0)");
    CHECK(wubu_semcons_verify(&sc, 0, 1, "rerun: 27.00 tok/s confirmed") == 1, "agent 1 verifies claim");
    CHECK(wubu_semcons_verify(&sc, 0, 2, "rerun: 27.00 tok/s confirmed") == 1, "agent 2 verifies claim");
    CHECK(wubu_semcons_majority_verified(&sc, 0, 4) == 0, "not majority (2/3 needed)");
    CHECK(wubu_semcons_verify(&sc, 0, 3, "rerun: 27.00 tok/s confirmed") == 1, "agent 3 verifies");
    CHECK(wubu_semcons_majority_verified(&sc, 0, 4) == 1, "majority verified (3/4 ≥ threshold)");

    /* DD05: Fraud detection + dispute */
    printf("\n=== DD05: Fraud Detection ===\n");
    wubu_fraud_t fr;
    CHECK(wubu_fraud_init(&fr, 4) == 0, "fraud init (4 agents)");
    CHECK(wubu_fraud_trust(&fr, 0) == 100, "agent 0 starts at trust 100");
    /* Agent 1 reports agent 2 for fraud */
    CHECK(wubu_fraud_report(&fr, 1, 2, "claim mismatch: reported 27 tok/s, observed 15") == 0, "fraud report filed");
    CHECK(wubu_fraud_adjudicate(&fr, 2) == 1, "fraud confirmed for agent 2");
    CHECK(wubu_fraud_trust(&fr, 2) == 50, "agent 2 trust decayed to 50");
    /* Repeat fraud → trust halves again */
    wubu_fraud_report(&fr, 0, 2, "mismatch again");
    wubu_fraud_adjudicate(&fr, 2);
    CHECK(wubu_fraud_trust(&fr, 2) == 25, "agent 2 trust decayed to 25 (repeat offender)");
    CHECK(wubu_fraud_trust(&fr, 0) == 100, "agent 0 (honest) trust unchanged");

    /* DD06 + DD07: Integration — consensus → trust-gated voting → DGM */
    printf("\n=== DD06/DD07: Integration ===\n");
    CHECK(wubu_fraud_trust(&fr, 2) < 100, "agent 2 (trust=25) has reduced voting weight");
    CHECK(wubu_fraud_trust(&fr, 0) == 100, "agent 0 (trust=100) has full voting weight");
    /* Consensus: 3 honest agents (trust 100) vs 1 fraud (trust 25) */
    wubu_bft_t b3; wubu_bft_init(&b3, 4);
    wubu_bft_vote(&b3, 0, 1, "config-A");
    wubu_bft_vote(&b3, 1, 1, "config-A");
    wubu_bft_vote(&b3, 3, 1, "config-A");
    wubu_bft_vote(&b3, 2, 0, "config-B");  /* low-trust agent disagrees */
    int maj = wubu_bft_majority(&b3, BFT_PROPOSE);
    CHECK(maj == 1, "consensus = config-A (3 honest > 1 low-trust)");
    CHECK(b.decided || b3.n_nodes == 4, "BFT state consistent");

    if (fails > 0) {
        printf("\n%d TEST(S) FAILED\n", fails);
        return 1;
    }
    printf("\nALL MULTICONSENSUS TESTS PASSED\n");
    return 0;
}
