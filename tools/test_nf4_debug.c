#include "wubu_nf4.h"
#include <stdio.h>
#include <math.h>
int main() {
    /* Directly test quantize on known values */
    float in[32];
    wubu_nf4_block blk;
    
    /* Fill with -0.696 * 0.5 = -0.348 */
    for (int i = 0; i < 32; i++) in[i] = -0.696193f * 0.5f;
    wubu_nf4_quantize_block(in, &blk);
    printf("input=-0.348, scale=%.6f\n", blk.scale);
    printf("normalized = %.6f\n", -0.3480965f / blk.scale);
    int idx0 = (blk.packed[0] >> 4) & 0xF;
    printf("idx0=%d, level[%d]=%.6f\n", idx0, idx0, WUBU_NF4_LEVELS[idx0]);
    printf("reconstructed = %.6f\n\n", blk.scale * WUBU_NF4_LEVELS[idx0]);
    
    /* Test with value exactly at level 1 */
    for (int i = 0; i < 32; i++) in[i] = -0.696193f;
    wubu_nf4_quantize_block(in, &blk);
    printf("input=-0.696193, scale=%.6f\n", blk.scale);
    printf("normalized = %.6f\n", -0.696193f / blk.scale);
    idx0 = (blk.packed[0] >> 4) & 0xF;
    printf("idx0=%d, level[%d]=%.6f\n", idx0, idx0, WUBU_NF4_LEVELS[idx0]);
    printf("reconstructed = %.6f\n", blk.scale * WUBU_NF4_LEVELS[idx0]);
    return 0;
}
