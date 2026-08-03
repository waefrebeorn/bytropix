/* test_vision.c -- Theme JB complete: the multimodal vision frontier. */
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "wubu_vision.h"

static int failures = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  FAIL: %s\n", m); failures++; } } while (0)
#define NEAR(a, b, t) CHECK(fabsf((a) - (b)) < (t), #a " ~= " #b)

int main(void)
{
    printf("=== test_vision (JB complete) ===\n");
    {
        float scores[5] = { 0.9f, 0.2f, 0.8f, 0.1f, 0.7f };
        int keep[5];
        CHECK(wubu_vision_selector(scores, 5, 0.5f, keep) == 3, "selected");
        CHECK(keep[0] == 0 && keep[1] == 2 && keep[2] == 4, "indices");
    }
    NEAR(wubu_vision_text_eff(100, 50), 0.5f, 1e-5f);
    {
        int out = 0;
        CHECK(wubu_vision_img_compress(100, 4, &out) == 0 && out == 25, "merged");
    }
    {
        int out = 0;
        CHECK(wubu_vision_vid_compress(100, 30, 0.5f, &out) == 0 && out == 50, "compressed");
    }
    {
        int out = 0;
        CHECK(wubu_vision_audio_compress(256, 0.5f, &out) == 0 && out == 128, "audio compressed");
    }
    {
        float vis[2] = { 1, 0 }, txt[2] = { 1, 0 }, sim;
        wubu_vision_clip_align(vis, txt, 2, &sim);
        NEAR(sim, 1.0f, 1e-5f);
    }
    {
        float patches[4][2] = { {1, 0}, {1.01f, 0}, {0, 1}, {0, 1.01f} };
        int keep[4];
        CHECK(wubu_vision_redundancy(&patches[0][0], 4, 2, 0.1f, keep) == 2, "deduped");
    }
    {
        int alloc[2];
        CHECK(wubu_vision_kv_budget(100, 100, 150, alloc) == 0, "within budget");
        CHECK(wubu_vision_kv_budget(100, 100, 100, alloc) == 0, "over budget");
    }
    {
        float attn[4] = { 0.9f, 0.1f, 0.8f, 0.2f };
        int keep[4];
        CHECK(wubu_vision_sparse(attn, 4, 0.5f, keep) == 2, "sparse");
    }
    {
        float feat[5] = { 0.1f, 0.9f, 0.3f, 0.8f, 0.2f };
        int topk[3];
        CHECK(wubu_vision_importance(feat, 5, 3, topk) == 3, "top-3");
        CHECK(topk[0] == 1 && topk[1] == 3, "most important");
    }
    {
        float audio[3] = { 1, 0, 0 }, vis[3] = { 0, 1, 0 }, fused[3];
        wubu_vision_av_fusion(audio, vis, 3, fused);
        NEAR(fused[0], 0.5f, 1e-5f);
        NEAR(fused[1], 0.5f, 1e-5f);
    }
    {
        int va = 0, ta = 0;
        CHECK(wubu_vision_budget_plan(100, 100, 150, &va, &ta) == 0, "within budget");
    }
    NEAR(wubu_vision_enc_eff(100, 768), 7.68f, 1e-2f);
    {
        float sal[4] = { 0.1f, 0.9f, 0.2f, 0.8f };
        int evict[4];
        CHECK(wubu_vision_evict(sal, 4, 0.5f, evict) == 2, "evicted low-salience");
    }
    {
        float vp[2] = { 1, 0 }, tp[2] = { 0, 1 }, shared[2];
        wubu_vision_prefix(vp, tp, 2, shared);
        NEAR(shared[0], 0.5f, 1e-5f);
        NEAR(shared[1], 0.5f, 1e-5f);
    }
    {
        float tokens[4][2] = { {1, 0}, {0, 1}, {1, 1}, {0, 0} };
        float out[8];
        CHECK(wubu_vision_stream(&tokens[0][0], 4, 2, 2, out) == 2, "streamed");
    }
    NEAR(wubu_vision_energy(0, 100, 0.5f), 50.0f, 1e-4f);
    {
        float tok[4][2] = { {1, 0}, {1.01f, 0}, {0, 1}, {0, 1.01f} };
        int keep[4];
        CHECK(wubu_vision_dedup(&tok[0][0], 4, 2, 0.1f, keep) == 2, "deduped");
    }
    {
        float task[3] = { 1, 0, 0 };
        float w[3];
        wubu_vision_route(task, 3, w);
        NEAR(w[0], 1.0f, 1e-5f);
    }

    if (failures == 0) printf("ALL VISION TESTS PASSED\n");
    else printf("%d VISION FAILURES\n", failures);
    return failures ? 1 : 0;
}