/* Test: wubu_mega (Round-3 #233 — MEGA EMA+gated step). */
#include "wubu_mega.h"
#include <stdio.h>
#include <assert.h>
#include <math.h>

int main(void) {
    wubu_mega_t *m = wubu_mega_create(8, 4);
    assert(m);
    float state[4] = {0,0,0,0};
    float x[8]; for (int i=0;i<8;i++) x[i] = 0.5f;
    float attn[8]; for (int i=0;i<8;i++) attn[i] = (i%2)?0.3f:-0.3f;
    float out[8];
    /* Step with forget=2 (sigmoid~0.88), input=1 (0.73), gate=0 (0.5). */
    wubu_mega_step(m, x, state, 2.0f, 1.0f, 0.0f, attn, out);
    for (int i=0;i<4;i++) assert(state[i] > 0.0f && isfinite(state[i]));
    for (int i=0;i<8;i++) assert(isfinite(out[i]));
    printf("MEGA state[0] = %.4f (expect 0.5*0.73=0.365)\n", state[0]);
    assert(fabsf(state[0] - 0.365f) < 1e-3f);
    /* Gate=0 halves the fused output. */
    float expected = 0.5f * (attn[0] + state[0]);
    printf("MEGA out[0] = %.4f (expect ~%.4f)\n", out[0], expected);
    assert(fabsf(out[0] - expected) < 1e-3f);
    wubu_mega_free(m);
    assert(wubu_mega_create(0,4) == NULL);
    printf("ALL MEGA TESTS PASSED\n");
    return 0;
}
