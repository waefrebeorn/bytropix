/* Test: wubu_attnres (Round-4 #413 — Attention Residuals cross-layer). */
#include "wubu_attnres.h"
#include <stdio.h>
#include <assert.h>
#include <math.h>

int main(void) {
    wubu_attnres_t *a = wubu_attnres_create(8, 4);
    assert(a);
    /* identity at init */
    assert(wubu_attnres_identity_ok(a));
    /* read with zero gates => y == x (pure local residual) */
    float x[8], y[8];
    for(int i=0;i<8;i++) x[i]=(i==2)?1.0f:0.0f;
    wubu_attnres_read(a, x, y);
    printf("AttnRes identity read y[2]=%.4f (expect 1)\n", y[2]);
    assert(fabsf(y[2]-1.0f)<1e-6f);
    /* set read gate on slot 0, write something to slot 0, read back */
    wubu_attnres_set_read_gate(a, 0, 0.5f);
    float out[8]; for(int i=0;i<8;i++) out[i]=(i==1)?2.0f:0.0f;
    wubu_attnres_write(a, out);          /* write gate=0 so slot unchanged (still 0) */
    wubu_attnres_read(a, x, y);
    printf("AttnRes read (write gate 0) y[2]=%.4f y[1]=%.4f (expect 1, 0)\n", y[2], y[1]);
    assert(fabsf(y[2]-1.0f)<1e-6f && fabsf(y[1])<1e-6f);
    /* now set write gate, write, read */
    wubu_attnres_set_write_gate(a, 0, 1.0f);
    wubu_attnres_write(a, out);          /* slot0 = 1.0*out => slot0[1]=2 */
    wubu_attnres_read(a, x, y);          /* y = x + 0.5*slot0 => y[1] += 0.5*2 =1 */
    printf("AttnRes read (write gate 1) y[1]=%.4f (expect 1)\n", y[1]);
    assert(fabsf(y[1]-1.0f)<1e-6f);
    wubu_attnres_free(a);
    assert(wubu_attnres_create(0,4)==NULL);   /* DA */
    assert(wubu_attnres_identity_ok(NULL)==0);
    printf("ALL ATTNRES TESTS PASSED\n");
    return 0;
}
