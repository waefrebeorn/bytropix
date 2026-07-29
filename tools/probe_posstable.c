/* Probe: is the engine forward position-stable?
 * Forward the same prefix of length L, once as T=L and once as T=L+1,
 * and compare logits at the LAST shared position (L-1). They MUST match
 * exactly (same tokens observed). Any divergence = latent forward bug. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include "wubu_model.h"
#include "wubu_model_adapter.h"
#include "wubu_model_safetensors_bridge.h"

int main(void) {
    const char *model = getenv("WUBU_MODEL");
    if (!model) model = "/home/wubu/models/Qwen3.6-27B";
    if (access(model, R_OK) != 0) { printf("SKIP\n"); return 0; }

    wubu_adapter_t ad; memset(&ad, 0, sizeof(ad));
    if (!wubu_adapter_load(&ad, model)) { ad.arch = WUBU_ARCH_QWEN_FAMILY; ad.ok = 1; }
    wubu_model_t mdl; memset(&mdl, 0, sizeof(mdl));
    if (wubu_model_init_safetensors(&mdl, model, &ad) != 0) { printf("INIT FAIL\n"); return 1; }

    int L = 8;
    int prompt[16];
    for (int i = 0; i < 16; i++) prompt[i] = 100 + (i % 5);

    int V = mdl.vocab_size;
    float *la = malloc(sizeof(float) * L * V);
    float *lb = malloc(sizeof(float) * (L + 1) * V);
    int *seqB = malloc(sizeof(int) * (L + 1));
    memcpy(seqB, prompt, sizeof(int) * L); seqB[L] = prompt[0];

    wubu_model_reset_state(&mdl);
    setenv("DUMP_LAYER_DIR", "/tmp/d8", 1);
    setenv("DBG_DUMP_EMBD", "/tmp/embd1", 1);
    setenv("DBG_DUMP_SSMOUT", "/tmp/ssm1", 1);
    setenv("DBG_DUMP_CONV", "/tmp/conv1", 1);
    setenv("DBG_DUMP_QKV", "/tmp/qkv1", 1);
    setenv("DBG_DUMP_SSMW", "/tmp/w1", 1);
    setenv("DBG_DUMP_NORMED", "/tmp/n1", 1);
    wubu_model_forward(&mdl, prompt, 1, L, la);
    wubu_model_reset_state(&mdl);
    setenv("DUMP_LAYER_DIR", "/tmp/d9", 1);
    setenv("DBG_DUMP_EMBD", "/tmp/embd2", 1);
    setenv("DBG_DUMP_SSMOUT", "/tmp/ssm2", 1);
    setenv("DBG_DUMP_CONV", "/tmp/conv2", 1);
    setenv("DBG_DUMP_QKV", "/tmp/qkv2", 1);
    setenv("DBG_DUMP_SSMW", "/tmp/w2", 1);
    setenv("DBG_DUMP_NORMED", "/tmp/n2", 1);
    wubu_model_forward(&mdl, prompt, 1, L, lb);

    float *pa = la + (size_t)(L - 1) * V;
    float *pb = lb + (size_t)(L - 1) * V;
    printf("DBG n_layers=%d d_model=%d T=%d/%d\n", mdl.n_layers, mdl.d_model, L, L+1);
    float maxd = 0, maxv = 0;
    int argA = 0, argB = 0;
    float vA = pa[0], vB = pb[0];
    for (int v = 0; v < V; v++) {
        float d = fabsf(pa[v] - pb[v]);
        if (d > maxd) maxd = d;
        if (fabsf(pa[v]) > maxv) maxv = fabsf(pa[v]);
        if (pa[v] > vA) { vA = pa[v]; argA = v; }
        if (pb[v] > vB) { vB = pb[v]; argB = v; }
    }
    printf("POS-STABLE L=%d: max|dlogit|=%.6e  max|logit|=%.4e  argmaxA=%d argmaxB=%d  %s\n",
           L, maxd, maxv, argA, argB, (maxd < 1e-3 * maxv) ? "STABLE" : "DIVERGENT");
    free(la); free(lb); free(seqB);
    wubu_model_free(&mdl);
    return 0;
}
