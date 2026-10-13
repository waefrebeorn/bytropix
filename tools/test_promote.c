/* test_promote.c -- AGENT #147 independent verifier gates every promotion
 * + #150 trace span emitted on every promote.
 * Proves: independent verifier required (self-approval rejected), attested
 * accept emits a span, non-attested rejected, proposer==verifier rejected,
 * already-gated rejected, lineage recorded, fail-closed on NULL/bad idx. */
#include "wubu_promote.h"
#include <stdio.h>
#include <string.h>

static int fails = 0, checks = 0;
#define CHECK(cond, msg) do { checks++; if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } else { printf("  [PASS] %s\n", msg); } } while (0)

int main(void) {
    wubu_promote_t p;
    CHECK(wubu_promote_init(&p) == 0, "init ok");

    /* propose a mutation by proposer "agent-A" */
    int idx = wubu_promote_propose(&p, "mut-001", "agent-A", 100);
    CHECK(idx == 0, "proposed at index 0");
    const wubu_promote_rec_t *r = wubu_promote_get(&p, idx);
    CHECK(r != NULL && r->status == WUBU_PROMOTE_PENDING, "proposal pending");

    /* SELF-APPROVAL rejected: verifier == proposer */
    uint64_t sp = wubu_promote_gate(&p, idx, "agent-A", 1);
    CHECK(sp == 0, "self-approval rejected (proposer==verifier)");
    r = wubu_promote_get(&p, idx);
    CHECK(r->status == WUBU_PROMOTE_REJECTED, "recorded rejected");

    /* fresh proposal, independent verifier attests -> accepted + span */
    int idx2 = wubu_promote_propose(&p, "mut-002", "agent-A", 100);
    uint64_t span = wubu_promote_gate(&p, idx2, "agent-B", 1);
    CHECK(span > 0, "independent verifier accept -> span emitted");
    r = wubu_promote_get(&p, idx2);
    CHECK(r->status == WUBU_PROMOTE_ACCEPTED, "recorded accepted");
    CHECK(r->verifier_attested == 1, "verifier attestation recorded");
    CHECK(r->span_id == span, "span id matches");
    CHECK(strcmp(r->verifier_id, "agent-B") == 0, "verifier id recorded");
    CHECK(r->trace_id == 100, "trace id carried");

    /* independent verifier but NOT attested -> rejected (fail-closed) */
    int idx3 = wubu_promote_propose(&p, "mut-003", "agent-A", 101);
    CHECK(wubu_promote_gate(&p, idx3, "agent-B", 0) == 0, "not attested -> rejected");

    /* already-gated proposal cannot be re-gated */
    CHECK(wubu_promote_gate(&p, idx2, "agent-C", 1) == 0, "already-gated -> 0");

    /* stats */
    CHECK(wubu_promote_accepted(&p) == 1, "1 accepted");
    CHECK(wubu_promote_rejected(&p) == 2, "2 rejected (self-approval + not-attested)");

    /* fail-closed */
    CHECK(wubu_promote_gate(&p, 999, "agent-B", 1) == (uint64_t)-1, "bad idx -> -1");
    CHECK(wubu_promote_gate(&p, idx, NULL, 1) == (uint64_t)-1, "NULL verifier -> -1");
    CHECK(wubu_promote_gate(NULL, 0, "x", 1) == (uint64_t)-1, "NULL gate -> -1");
    CHECK(wubu_promote_propose(&p, NULL, "a", 1) == -1, "NULL proposal_id -> -1");
    CHECK(wubu_promote_get(&p, 999) == NULL, "bad idx get -> NULL");
    CHECK(wubu_promote_get(NULL, 0) == NULL, "NULL get -> NULL");

    printf("\n=== test_promote: %d checks, %d failures ===\n", checks, fails);
    return fails ? 1 : 0;
}
