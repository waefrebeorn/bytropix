/* test_archive_wiring.c -- AGENT #149 Brain archive wired to Body operator.
 * Proves the end-to-end flow: a mutation is promoted through the Brain
 * archive (independent verifier gate) and then APPLIED by the Body operator
 * hook, then RESTARTED. The archive's accepted promotion feeds the operator's
 * apply; a rejected promotion is never applied. */
#include "wubu_promote.h"
#include "wubu_ophook.h"
#include <stdio.h>
#include <string.h>

static int fails = 0, checks = 0;
#define CHECK(cond, msg) do { checks++; if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } else { printf("  [PASS] %s\n", msg); } } while (0)

int main(void) {
    wubu_promote_t archive;
    wubu_ophook_t op;
    wubu_promote_init(&archive);
    wubu_ophook_init(&op);

    /* Brain: propose + independently verify a mutation */
    int idx = wubu_promote_propose(&archive, "mut-attn-1", "agent-A", 500);
    uint64_t span = wubu_promote_gate(&archive, idx, "agent-B", 1);
    CHECK(span > 0, "Brain: mutation promoted (independent verifier)");
    const wubu_promote_rec_t *rec = wubu_promote_get(&archive, idx);
    CHECK(rec->status == WUBU_PROMOTE_ACCEPTED, "Brain: promotion accepted");

    /* Body operator: apply the promoted mutation (wired from the archive) */
    int applied = wubu_ophook_issue(&op, WUBU_OP_APPLY, rec->proposal_id);
    CHECK(applied >= 0 && op.log[applied].applied == 1, "Body: operator applied the promoted mutation");
    CHECK(wubu_ophook_applied(&op, rec->proposal_id) == 1, "Body: mutation now applied");

    /* Body operator: restart the applied component */
    int restarted = wubu_ophook_issue(&op, WUBU_OP_RESTART, rec->proposal_id);
    CHECK(op.log[restarted].applied == 1, "Body: restarted the applied mutation");

    /* a REJECTED promotion is never applied to the Body */
    int bad = wubu_promote_propose(&archive, "mut-bad", "agent-A", 501);
    wubu_promote_gate(&archive, bad, "agent-A", 1);   /* self-approval -> rejected */
    const wubu_promote_rec_t *badrec = wubu_promote_get(&archive, bad);
    CHECK(badrec->status == WUBU_PROMOTE_REJECTED, "Brain: bad promotion rejected");
    /* the Body never sees it as applied */
    CHECK(wubu_ophook_applied(&op, badrec->proposal_id) == 0, "Body: rejected promotion never applied");

    printf("\n=== test_archive_wiring: %d checks, %d failures ===\n", checks, fails);
    return fails ? 1 : 0;
}
