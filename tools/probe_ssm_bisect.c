/* Bisect the SSM forward directly: build a deterministic pseudo-input x of
 * shape [maxT, d_model], call wubu_ssm_forward with T=L and T=L+1 (cache=0,
 * fresh state) and compare delta_out at position L-1. Position L-1 only sees
 * tokens 0..L-1, so outputs MUST match. Any diff => bug in SSM math. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "wubu_ssm.h"
#include "wubu_dims.h"

#define MAXT 16
#define DM 4096
static ssm_layer_weights W;
static float xbuf[MAXT * DM];

static void fill_w(void) {
    memset(&W, 0, sizeof(W));
    W.f32_mode = 1;
    int D = DM, C = CONV_DIM, V = VALUE_DIM, R = DT_RANK;
    W.attn_qkv_weight_f32 = malloc((size_t)D * C * sizeof(float));
    W.attn_gate_weight_f32 = malloc((size_t)D * V * sizeof(float));
    W.ssm_beta_weight  = malloc((size_t)D * R * sizeof(float));
    W.ssm_alpha_weight = malloc((size_t)D * R * sizeof(float));
    W.ssm_a = malloc(sizeof(float) * R);
    W.ssm_dt_bias = malloc(sizeof(float) * R);
    W.ssm_conv1d_weight = malloc((size_t)CONV_DIM * CONV_KERNEL * sizeof(float));
    W.ssm_norm_weight = malloc(sizeof(float) * SSM_D_STATE);
    W.ssm_out_weight_f32 = malloc((size_t)V * D * sizeof(float));
    srand(12345);
    #define RF(p,n) do{ for(int i=0;i<(n);i++) (p)[i]=(float)rand()/RAND_MAX*2-1; }while(0)
    RF(W.attn_qkv_weight_f32, D*C);
    RF(W.attn_gate_weight_f32, D*V);
    RF(W.ssm_beta_weight, D*R);
    RF(W.ssm_alpha_weight, D*R);
    RF(W.ssm_a, R);
    RF(W.ssm_dt_bias, R);
    RF(W.ssm_conv1d_weight, CONV_DIM*CONV_KERNEL);
    RF(W.ssm_norm_weight, SSM_D_STATE);
    RF(W.ssm_out_weight_f32, V*D);
    for (int i = 0; i < MAXT * DM; i++) xbuf[i] = (float)rand()/RAND_MAX*2-1;
}

int main(void) {
    fill_w();
    int D = DM, V = VALUE_DIM;
    int L = 8;
    float *st = calloc(SSM_V_HEADS * SSM_D_STATE * SSM_D_STATE, sizeof(float));
    float *cs = calloc((CONV_KERNEL-1) * CONV_DIM, sizeof(float));
    float *outA = malloc((size_t)L * V * sizeof(float));
    float *outB = malloc((size_t)(L+1) * V * sizeof(float));

    wubu_ssm_forward(xbuf, 1, L, &W, st, cs, outA, NULL, NULL);
    memset(st, 0, SSM_V_HEADS*SSM_D_STATE*SSM_D_STATE*sizeof(float));
    memset(cs, 0, (CONV_KERNEL-1)*CONV_DIM*sizeof(float));
    wubu_ssm_forward(xbuf, 1, L+1, &W, st, cs, outB, NULL, NULL);

    float *pa = outA + (size_t)(L-1)*V;
    float *pb = outB + (size_t)(L-1)*V;
    float sumA=0;
    for (int v=0; v<V; v++) sumA += fabsf(outA[v]);
    printf("DBG outA sum|val|=%.4e  outB[0]=%.4e  outA[L-1][0..2]=%.4e %.4e %.4e\n",
           sumA, outB[0], pa[0], pa[1], pa[2]);
    float maxd=0, maxv=0;
    for (int v=0; v<V; v++){ float d=fabsf(pa[v]-pb[v]); if(d>maxd)maxd=d; if(fabsf(pa[v])>maxv)maxv=fabsf(pa[v]); }
    printf("SSM POS-STABLE L=%d: max|d|=%.6e max|val|=%.4e %s\n", L, maxd, maxv,
           (maxv>1e-6 && maxd < 1e-3*maxv) ? "STABLE":"DIVERGENT");
    return 0;
}
