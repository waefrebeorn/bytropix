/* test_compress2.c -- Theme IZ complete: the context-compression frontier. */
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "wubu_compress2.h"

static int failures = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  FAIL: %s\n", m); failures++; } } while (0)
#define NEAR(a, b, t) CHECK(fabsf((a) - (b)) < (t), #a " ~= " #b)

int main(void)
{
    printf("=== test_compress2 (IZ complete) ===\n");
    {
        float p[5] = { 10, 5, 20, 3, 15 };
        int keep[5];
        CHECK(wubu_comp2_llmlingua(p, 5, 10.0f, keep) == 3, "perplexity drop");
        CHECK(keep[0] == 0 && keep[1] == 1 && keep[2] == 3, "kept indices");
    }
    {
        float s[4] = { 0.1f, 0.8f, 0.3f, 0.9f };
        int keep[4];
        CHECK(wubu_comp2_llmlingua2(s, 4, 0.5f, keep) == 2, "classifier kept");
        CHECK(keep[0] == 1 && keep[1] == 3, "high scores");
    }
    {
        int q[5] = { 0, 1, 0, 1, 0 };
        int order[5];
        wubu_comp2_reorder(q, 5, order);
        CHECK(order[0] == 1 && order[1] == 3, "questions first");
    }
    {
        float info[4] = { 0.1f, 0.9f, 0.3f, 0.8f };
        int keep[4];
        CHECK(wubu_comp2_self_info(info, 4, 0.5f, keep) == 2, "top-k kept");
    }
    {
        float sc[5] = { 0.9f, 0.2f, 0.8f, 0.1f, 0.7f };
        int keep[5];
        CHECK(wubu_comp2_recmp(sc, 5, 0.7f, 0.6f, keep) == 3, "RECMP kept");
    }
    {
        float emb[3][2] = { {1, 0}, {0, 1}, {3, 4} };
        int atoms[3];
        CHECK(wubu_comp2_doc2atom(&emb[0][0], 3, 2, 2.0f, atoms) == 1,
              "one atom");
    }
    {
        long evict = 0;
        CHECK(wubu_comp2_cartridge(120, 100, &evict) == 1 && evict == 20,
              "evict needed");
    }
    CHECK(wubu_comp2_lamr((float[]){ 0.9f, 0.1f }, (float[]){ 0.2f, 0.8f }, 2, 0.5f, 0.5f) == 1,
          "LaMR one kept");
    {
        float dens[4] = { 0.1f, 0.8f, 0.3f, 0.9f };
        int seg[4];
        CHECK(wubu_comp2_sesrag(dens, 4, 0.5f, seg) == 2, "SES-RAG segments");
    }
    {
        float toks[5] = { 0.1f, 0.9f, 0.3f, 0.8f, 0.2f };
        int meta[2];
        CHECK(wubu_comp2_grc(toks, 5, 2, meta) == 2, "GRC meta tokens");
        CHECK(meta[0] == 1 && meta[1] == 3, "top-2");
    }
    CHECK(wubu_comp2_epc(0.8f, 0.3f) == 1, "EPC retain");
    CHECK(wubu_comp2_epc(0.2f, 0.5f) == 0, "EPC drop");
    {
        float imp[4] = { 0.1f, 0.9f, 0.3f, 0.8f };
        int order[4];
        wubu_comp2_lim(imp, 4, order);
        CHECK(order[0] == 1 && order[1] == 3, "LIM important first");
    }
    CHECK(wubu_comp2_budget(100, 0.8f, 500) == 540, "density budget");
    CHECK(wubu_comp2_tool_schema("{\"type\":\"object\"}", 18, 0.5f) == 9,
          "schema compressed");
    {
        float ctx[8] = { 1, 2, 3, 4, 5, 6, 7, 8 };
        float latent[2];
        CHECK(wubu_comp2_autoenc(ctx, 4, 2, latent, 2) == 2, "autoencoder");
    }
    {
        float doc[3][2] = { {1, 2}, {3, 4}, {5, 6} };
        float lora[2];
        CHECK(wubu_comp2_distill(&doc[0][0], 3, 2, lora) == 2, "distilled");
        NEAR(lora[0], 3.0f, 1e-5f);
    }
    {
        float kv[4] = { 1, 2, 3, 4 };
        float mem[2];
        CHECK(wubu_comp2_latent_mem(kv, 2, 2, mem) == 2, "latent memory");
    }
    {
        float attn[5] = { 0.1f, 0.2f, 0.3f, 0.2f, 0.1f };
        int pages[3];
        CHECK(wubu_comp2_paged(attn, 5, 2, pages) == 3, "paged");
    }
    CHECK(wubu_comp2_governor(0.5f, 0.5f, 0.6f) == 1, "governor satisfied");
    NEAR(wubu_comp2_fidelity((float[]){ 1, 2, 3 }, (float[]){ 1.1f, 2.1f, 3.1f }, 3), 0.1f, 1e-5f);

    if (failures == 0) printf("ALL COMPRESS2 TESTS PASSED\n");
    else printf("%d COMPRESS2 FAILURES\n", failures);
    return failures ? 1 : 0;
}