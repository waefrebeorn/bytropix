/*
 * test_proj_accuracy.c -- verify quantized_matmul's projection math against an
 * unambiguous scalar oracle (y[j] = sum_k W[j*n_rows + k] * x[k]) on REAL
 * Qwen3.6-27B layer-0 gate_proj weights (F16). This is the unit test that
 * catches the transpose bug: a broken layout reads W as [n_rows,n_cols] and
 * produces a different (wrong) vector for non-square mats.
 */
#include "wubu_model_safetensors_bridge.h"
#include "wubu_dims.h"
#include "wubu_model.h"
#include "gguf_reader.h"
#include "safetensors_reader.h"
#include "wubu_safetensors_shard.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    const char *dir = (argc>1)?argv[1]:"/home/wubu/models/Qwen3.6-27B";
    setenv("MAX_LAYERS","1",1);

    wubu_adapter_t ad; memset(&ad,0,sizeof(ad));
    if (!wubu_adapter_load(&ad,dir)) { ad.arch=WUBU_ARCH_QWEN_FAMILY; ad.ok=1; }
    wubu_model_t m;
    if (wubu_model_init_safetensors(&m,dir,&ad)!=0){ fprintf(stderr,"init FAIL\n"); return 1; }

    int D = m.d_model;            /* n_rows = d_model */
    wubu_shard_ctx_t *sc = (wubu_shard_ctx_t*)m.shard_ctx;

    int dtype; int64_t row;
    const uint8_t *raw = wubu_shard_raw(sc,
        "model.language_model.layers.0.mlp.gate_proj.weight", &dtype, &row);
    if (!raw) { fprintf(stderr,"load gate_proj FAIL\n"); return 1; }
    /* gate_proj is [dff, d_model]; row length = d_model, n_rows = dff */
    int dff = (int)(row);
    printf("d_model=%d dff=%d dtype=%d\n", D, dff, dtype);
    if (row != D) { fprintf(stderr,"unexpected row %lld != d_model\n",(long long)row); return 1; }
    const uint16_t *W16 = (const uint16_t*)raw;

    /* shard dtype enum matches ggml_type: 1=F16, 2=BF16. Use the real type. */
    int wt = (dtype == 1) ? GGML_TYPE_F16 : (dtype == 2) ? GGML_TYPE_BF16 : GGML_TYPE_F32;

    /* input activations (same pattern as test_probe_qwen) */
    float *x = (float*)malloc(D*sizeof(float));
    for (int i=0;i<D;i++) x[i]=(float)(i%7-3)*0.01f;

    /* (1) bytropix quantized_matmul (GEMV path, correct dtype) */
    float *y = (float*)malloc(dff*sizeof(float));
    quantized_matmul(x, W16, wt, D, dff, 0, y);

    /* (2) unambiguous scalar oracle y[j] = sum_k W[j*D + k]*x[k] */
    float *ref = (float*)malloc(dff*sizeof(float));
    for (int j=0;j<dff;j++){
        float s=0; const uint16_t *wr = W16 + (size_t)j*D;
        for (int k=0;k<D;k++) s += st_bf16_to_f32(wr[k]) * x[k];
        ref[j]=s;
    }

    float maxerr=0, dot=0, norm_y=0, norm_r=0;
    for (int j=0;j<dff;j++){
        float e=fabsf(y[j]-ref[j]); if (e>maxerr) maxerr=e;
        dot += y[j]*ref[j]; norm_y += y[j]*y[j]; norm_r += ref[j]*ref[j];
    }
    float cos = (norm_y>0&&norm_r>0) ? dot/(sqrtf(norm_y)*sqrtf(norm_r)) : 0;
    printf("maxErr=%g cosine=%f\n", maxerr, cos);
    int ok = (cos>0.99999f) && (maxerr < 1e-2f);
    printf("%s\n", ok?"PASS":"FAIL");

    free(x); free(y); free(ref);
    wubu_model_safetensors_free(&m);
    return ok?0:1;
}
