/*
 * test_btl3_lora.c -- verify BTL-3 two-step LoRA orchestration:
 *   base load (wubu_model_init_safetensors) -> wubu_model_apply_lora(delta).
 * Proves that loading a .safetensors that IS a LoRA adapter, then applying
 * its delta onto the resident base weights, yields a model whose forward
 * still runs with finite logits. The adapter's __metadata__ names the
 * base; wubu_adapter_load detects is_lora so apply() is honored.
 */
#include "wubu_model_safetensors_bridge.h"
#include "wubu_model.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int all_finite(const float *p, int n){ for(int j=0;j<n;j++) if(!isfinite(p[j])) return 0; return 1; }

int main(void){
    const char *base = "fixture_model.safetensors";
    const char *lora = "fixture_btl3_lora.safetensors";

    /* Step 1: load the BASE checkpoint (proven-working zeroed-adapter path). */
    wubu_model_t m = {0};
    if (wubu_model_init_safetensors(&m, base, &(wubu_adapter_t){0}) != 0) {
        fprintf(stderr, "FAIL: base load\n"); return 1;
    }
    if (m.n_layers != 2 || m.d_model != 256) {
        fprintf(stderr, "FAIL: base dims wrong (n_layers=%d d_model=%d)\n", m.n_layers, m.d_model);
        return 1;
    }

    /* Step 2: detect + apply the BTL-3 LoRA delta onto the base. */
    wubu_adapter_t ad = {0};
    if (!wubu_adapter_load(&ad, lora)) {
        fprintf(stderr, "FAIL: adapter_load\n"); return 1;
    }
    if (!ad.is_lora) {
        fprintf(stderr, "FAIL: adapter not detected as LoRA (base=[%s])\n", ad.base_model);
        return 1;
    }
    if (wubu_model_apply_lora(&m, lora, &ad) != 0) {
        fprintf(stderr, "FAIL: wubu_model_apply_lora\n"); return 1;
    }
    printf("PASS: BTL-3 LoRA base+adapter loaded + delta applied (n_layers=%d d_model=%d vocab=%d)\n",
           m.n_layers, m.d_model, m.vocab_size);

    /* Forward must still produce finite logits. */
    int vocab = m.vocab_size;
    float *logits = malloc((size_t)vocab * sizeof(float));
    int prompt[1] = {1};
    wubu_model_forward(&m, prompt, 1, 1, logits);
    if (!all_finite(logits, vocab)) {
        fprintf(stderr, "FAIL: forward logits non-finite after LoRA\n"); return 1;
    }
    printf("PASS: wubu_model_forward finite after LoRA (vocab=%d)\n", vocab);

    free(logits);
    wubu_model_safetensors_free(&m);
    return 0;
}
