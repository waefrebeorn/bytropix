#include "wubu_nf4.h"
#include <stdio.h>
#include <math.h>
int main() {
    wubu_nf4_block blk;
    float in[32], out[32];
    int errors = 0;
    /* Test with values spread across levels (not all-equal which maps to extremes) */
    for (int lv = 0; lv < 16; lv++) {
        /* Create block where each element maps to a different level */
        for (int i = 0; i < 16; i++) {
            in[i*2]   = WUBU_NF4_LEVELS[lv];
            in[i*2+1] = WUBU_NF4_LEVELS[i];
        }
        wubu_nf4_quantize_block(in, &blk);
        wubu_nf4_dequantize_block(&blk, out);
        int first_idx = (blk.packed[0] >> 4) & 0xF;
        printf("level[%2d]=% .6f → idx=%2d %s\n", lv, WUBU_NF4_LEVELS[lv], first_idx,
               first_idx == lv ? "OK" : "WRONG");
        if (first_idx != lv) errors++;
    }
    printf("\n%d errors\n", errors);
    return errors;
}
