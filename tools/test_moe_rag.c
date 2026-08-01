/*
 * test_moe_rag.c -- X01-X06 + Y01-Y04 verification.
 */
#include "wubu_moe_rag.h"
#include <stdio.h>
#include <math.h>

static int failures = 0;
#define CHECK(c, msg) do { if (!(c)) { printf("  FAIL: %s\n", msg); failures++; } } while (0)

int main(void) {
    printf("=== test_moe_rag (X01-X06/Y01-Y04) ===\n");

    /* X01 Top-K: gate [0.1,0.9,0.5,0.3], K=2 -> {1,2}. */
    float g[4] = {0.1f,0.9f,0.5f,0.3f};
    int sel[4]; int K = wubu_topk_route(g, 4, 2, sel);
    CHECK(K==2 && sel[0]==1 && sel[1]==2, "Top-2 -> {1,2}");

    /* X02 Expert-Choice: 3 tokens, 2 experts, C=2 -> each expert picks 2 tokens.
     * score token-major: token0->[1,0], token1->[0,1], token2->[1,1].
     * expert0 picks tokens {0,2} (scores 1,0,1); expert1 picks {1,2} (0,1,1). */
    float sc[6] = {1,0, 0,1, 1,1};
    int out[4]; int cnt[2];
    int off = wubu_expert_choice(sc, 3, 2, 2, out, cnt);
    CHECK(off==4, "2 experts * 2 tokens = 4");
    CHECK(cnt[0]==2 && cnt[1]==2, "each expert picks 2");
    /* expert0's chosen: tokens with top score0 -> 0 and 2. expert1 -> 1 and 2. */
    int e0a = (out[0]==0||out[0]==2)?1:0, e0b=(out[1]==0||out[1]==2)?1:0;
    CHECK(e0a&&e0b, "expert0 picks {0,2}");

    /* X03 shared-expert: routed={1}, N=4 -> out={1,2,1,1} (shared adds 1). */
    int r[1] = {1};
    int sh[4];
    wubu_shared_expert(r, 4, 1, sh);
    CHECK(sh[0]==1 && sh[1]==2 && sh[2]==1 && sh[3]==1, "shared agg = routed+1");

    /* X04 sigmoid gate: scores [2, -2], thr 0.7 -> sigmoid(2)=0.88>0.7 keep 0;
     * sigmoid(-2)=0.12<0.7 drop. sel={0}. */
    float sg[2] = {2.0f,-2.0f};
    int s4[2]; int n4 = wubu_sigmoid_gate(sg, 2, 0.7f, s4);
    CHECK(n4==1 && s4[0]==0, "sigmoid keeps expert 0 only");

    /* X05 prefetch: predicted {0,1,2}, cached {1}, N=3 -> prefetch {0,2}. */
    int pred[3] = {0,1,2};
    char cached[3] = {0,1,0};
    int pf[3]; int npf = wubu_expert_prefetch(pred, 3, cached, 3, pf);
    CHECK(npf==2, "prefetch 2 uncached");

    /* X06 capacity: 4 tokens, experts [0,0,1,1], N=2, cap=1.0 -> limit=2.
     * expert0 gets 2 (<=2 ok), expert1 gets 2 (<=2 ok). All kept. */
    int eo[4] = {0,0,1,1};
    char keep[4];
    int kp = wubu_capacity_factor(eo, 4, 2, 1.0f, keep);
    CHECK(kp==4, "within capacity -> all kept");

    /* Y01 KV Packet: tok_doc [0,0,1] -> doc_id same. */
    int td[3] = {0,0,1};
    int did[3];
    wubu_kvpacket_doc(td, 3, did);
    CHECK(did[0]==0 && did[2]==1, "kvpacket doc ids");

    /* Y02 RACC: is_retrieved {1,0,1} -> keep {1,0,1}, cnt 2. */
    char ir[3] = {1,0,1};
    char rk[3]; int rc = wubu_racc_keep(ir, 3, rk);
    CHECK(rc==2 && rk[0]==1 && rk[1]==0, "racc keeps retrieved");

    /* Y03 CAG ready. */
    CHECK(wubu_cag_ready(1)==1 && wubu_cag_ready(0)==0, "cag ready flag");

    /* Y04 cross-doc ns. */
    CHECK(wubu_crossdoc_ns(td, 2)==1, "crossdoc ns = doc 1");

    if (failures == 0) { printf("ALL MOE-RAG TESTS PASSED\n"); return 0; }
    printf("%d MOE-RAG TEST(S) FAILED\n", failures);
    return 1;
}
