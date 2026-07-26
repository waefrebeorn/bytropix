/* test_lora.c -- verify rank-r LoRA merge + forward against a reference matmul. */
#include "wubu_lora.h"
#include <stdio.h>
#include <math.h>

static int approx(float a, float b, float tol) { return fabsf(a-b) <= tol; }

int main(void) {
    int rank = 2, in_f = 3, out_f = 2;
    float alpha = 4.0f;  // scale = 2.0

    // A [rank, in_f] = [[1,2,3],[4,5,6]]
    float A[6] = {1,2,3, 4,5,6};
    // B [out_f, rank] = [[7,8],[9,10]]  row0=[7,8], row1=[9,10]
    float B[4] = {7,8, 9,10};

    wubu_lora_t *l = wubu_lora_create(rank, alpha, in_f, out_f);
    if (!l) { fprintf(stderr, "FAIL: create\n"); return 1; }
    if (wubu_lora_load_f32(l, A, B) != 0) { fprintf(stderr, "FAIL: load\n"); wubu_lora_free(l); return 1; }
    if (!approx(wubu_lora_scale(l), 2.0f, 1e-5f)) { fprintf(stderr, "FAIL: scale=%g\n", wubu_lora_scale(l)); wubu_lora_free(l); return 1; }

    // delta = scale * (B^T @ A): B^T is [rank,out_f];  B^T@A = [out_f, in_f]
    // row0: [7*1+8*4, 7*2+8*5, 7*3+8*6] = [39,54,69] *2 = [78,108,138]
    // row1: [9*1+10*4, 9*2+10*5, 9*3+10*6] = [49,68,87] *2 = [98,136,174]
    float W[6] = {0,0,0, 0,0,0};
    if (wubu_lora_apply(l, W) != 0) { fprintf(stderr, "FAIL: apply\n"); wubu_lora_free(l); return 1; }
    float eW[6] = {78,108,138, 98,136,174};
    for (int i = 0; i < 6; i++) if (!approx(W[i], eW[i], 1e-3f)) { fprintf(stderr, "FAIL: W[%d]=%g exp=%g\n", i, W[i], eW[i]); wubu_lora_free(l); return 1; }

    // forward: x=[1,1,1] -> scale * ((x@A^T)@B^T)
    // x@A^T = [1*1+1*2+1*3, 1*4+1*5+1*6]=[6,15]
    // [6,15]@B^T = [6*7+15*9, 6*8+15*10] = [177, 192] *2
    float x[3] = {1,1,1};
    float out[2] = {0,0};
    if (wubu_lora_forward(l, x, out) != 0) { fprintf(stderr, "FAIL: forward\n"); wubu_lora_free(l); return 1; }
    if (!approx(out[0], 324.0f, 1e-2f) || !approx(out[1], 408.0f, 1e-2f)) {
        fprintf(stderr, "FAIL: forward out=[%g,%g]\n", out[0], out[1]); wubu_lora_free(l); return 1;
    }

    wubu_lora_free(l);
    printf("PASS: lora (rank-%d merge + forward)\n", rank);
    return 0;
}
