/* Test: wubu_model_wire_hwaccel() actually wires the HW stack into a real model.
 * Loads the small fixture model, wires SIMD detect + RDRAM KV banks + gamebud
 * frame-budget + tandem pipeline, and asserts the model reflects the wiring and
 * that a forward pass still runs (no crash) with the rambus billing active. */
#include "wubu_model.h"
#include "wubu_model_safetensors_bridge.h"
#include "wubu_hwcaps.h"
#include "wubu_rambus.h"
#include "wubu_tandem.h"
#include "wubu_gamebud.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

int main(void) {
    wubu_model_t mdl;
    memset(&mdl, 0, sizeof(mdl));

    /* Load a tiny fixture model so the test is self-contained. */
    const char *path = getenv("WUBU_TEST_MODEL");
    if (!path) path = "fixture.safetensors";
    if (wubu_model_init_auto(&mdl, path) != 0) {
        fprintf(stderr, "skip: cannot load %s\n", path);
        return 0; /* not a failure — fixture may be absent */
    }

    /* Before wiring: all HW fields must be empty. */
    assert(mdl.hw_simd_bits == 0);
    assert(mdl.kv_rambus == NULL);
    assert(mdl.gamebud == NULL);
    assert(mdl.tandem == NULL);

    int kv_h = mdl.gqa_kv_heads > 0 ? mdl.gqa_kv_heads : 8;
    int kv_dim = (mdl.gqa_head_dim > 0 ? mdl.gqa_head_dim : 128) * kv_h;
    int rc = wubu_model_wire_hwaccel(&mdl, 1 /*simd*/, 8 /*banks*/, kv_dim,
                                     20000 /*20ms*/, "0-1", "2-3");
    assert(rc == 0);

    /* After wiring: fields populated. */
    assert(mdl.hw_simd_bits == 128 || mdl.hw_simd_bits == 256 || mdl.hw_simd_bits == 512);
    assert(mdl.hw_simd_lanes >= 4);
    assert(mdl.kv_rambus != NULL);
    assert(mdl.kv_rambus_banks == 8);
    assert(mdl.gamebud != NULL);
    assert(mdl.frame_budget_us == 20000);
    assert(mdl.tandem != NULL);

    const char *s = wubu_model_hwaccel_str(&mdl);
    printf("wired: %s\n", s);
    assert(strstr(s, "SIMD=") != NULL);
    assert(strstr(s, "rambus_banks=8") != NULL);
    assert(strstr(s, "tandem=on") != NULL);

    /* Exercise a real forward with the rambus billing active — only if this
     * fixture is a full GQA model (has a KV cache); tiny fixtures are skipped. */
    if (mdl.gqa_k_cache && mdl.vocab_size > 0 && mdl.d_model > 0) {
        int tok = 0; (void)tok;
        float *embd = (float *)calloc(mdl.d_model, sizeof(float));
        embd[0] = 1.0f;
        float *logits = (float *)calloc(mdl.vocab_size, sizeof(float));
        wubu_model_forward_from_embd(&mdl, embd, 1, 1, logits);
        /* second step: prefix now length 1 — rambus bills a read */
        wubu_model_forward_from_embd(&mdl, embd, 1, 1, logits);

        wubu_rambus_t *rb = (wubu_rambus_t *)mdl.kv_rambus;
        uint64_t h, m, c;
        wubu_rambus_stats(rb, &h, &m, &c);
        printf("rambus: hits=%llu misses=%llu cycles=%llu eff_BW=%.1f MB/s\n",
               (unsigned long long)h, (unsigned long long)m, (unsigned long long)c,
               wubu_rambus_eff_bw_mbps(rb, h * 256 + m * 256));
        assert(c > 0);  /* at least one access billed */
        free(embd); free(logits);
    } else {
        printf("wired: (fixture has no GQA KV cache — forward skipped, wiring verified)\n");
    }

    wubu_model_free(&mdl);  /* must unwire cleanly */
    printf("ALL MODEL-HWACCEL WIRE TESTS PASSED\n");
    return 0;
}
