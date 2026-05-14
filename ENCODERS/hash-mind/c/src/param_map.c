/**
 * gradient_check.c — Identify failing weight by index
 */
#include "tokenizer.h"
#include "hashmind_model.h"
#include "nn_ops.h"
#include <stdio.h>

int main() {
    typedef long l2;
    l2 emb = (l2)VOCAB * D_MODEL;  /* token_embed */
    l2 hp = D_MODEL;                /* hash_projector */
    l2 blk = (l2)D_MODEL*D_MODEL*3   /* qkv_w */
            + (l2)D_MODEL*D_MODEL     /* out_w */
            + (l2)D_MODEL*D_FF        /* ffn1_w */
            + (l2)D_FF*D_MODEL        /* ffn2_w */
            + D_MODEL                  /* ln1_gamma */
            + D_MODEL                  /* ln1_beta */
            + D_MODEL                  /* ln2_gamma */
            + D_MODEL;                /* ln2_beta */
    l2 out = (l2)D_MODEL * VOCAB;    /* out_w */

    l2 total = emb + hp + blk * N_LAYERS + out;
    printf("token_embed: [0, %ld)\n", emb);
    printf("hash_projector: [%ld, %ld)\n", emb, emb + hp);
    printf("Block 0: [%ld, %ld)\n", emb + hp, emb + hp + blk);
    printf("  qkv_w: [%ld, %ld)\n", emb + hp, emb + hp + (l2)D_MODEL*D_MODEL*3);
    printf("  out_w: [%ld, %ld)\n", emb + hp + (l2)D_MODEL*D_MODEL*3,
           emb + hp + (l2)D_MODEL*D_MODEL*3 + (l2)D_MODEL*D_MODEL);
    printf("  ffn1_w: [%ld, %ld)\n",
           emb + hp + (l2)D_MODEL*D_MODEL*3 + (l2)D_MODEL*D_MODEL,
           emb + hp + (l2)D_MODEL*D_MODEL*3 + (l2)D_MODEL*D_MODEL + (l2)D_MODEL*D_FF);
    printf("  ffn2_w: [%ld, %ld)\n",
           emb + hp + (l2)D_MODEL*D_MODEL*3 + (l2)D_MODEL*D_MODEL + (l2)D_MODEL*D_FF,
           emb + hp + (l2)D_MODEL*D_MODEL*3 + (l2)D_MODEL*D_MODEL + (l2)D_MODEL*D_FF + (l2)D_FF*D_MODEL);
    printf("  ln1: [%ld, %ld)\n",
           emb + hp + (l2)D_MODEL*D_MODEL*3 + (l2)D_MODEL*D_MODEL + (l2)D_MODEL*D_FF + (l2)D_FF*D_MODEL,
           emb + hp + (l2)D_MODEL*D_MODEL*3 + (l2)D_MODEL*D_MODEL + (l2)D_MODEL*D_FF + (l2)D_FF*D_MODEL + D_MODEL*2);
    printf("Block 1: [%ld, %ld)\n", emb + hp + blk, emb + hp + blk*2);
    printf("Block 1 begins at: %ld\n", emb + hp + blk * 1);
    printf("Block 2 begins at: %ld\n", emb + hp + blk * 2);
    printf("Block 3 begins at: %ld\n", emb + hp + blk * 3);
    printf("out_w: [%ld, %ld)\n", emb + hp + blk * N_LAYERS, total);
    
    printf("\nIndex 6208 = ");
    l2 pos = 6208;
    if (pos < emb) printf("token_embed[%ld][%ld]\n", pos / D_MODEL, pos % D_MODEL);
    else if (pos < emb + hp) printf("hash_projector[%ld]\n", pos - emb);
    else {
        l2 rel = pos - emb - hp;
        if (rel < blk) {
            int b = 0;
            if (rel < (l2)D_MODEL*D_MODEL*3) printf("block 0 qkv_w[%ld][%ld]\n", rel / (D_MODEL*3), rel % (D_MODEL*3));
            else if (rel < (l2)D_MODEL*D_MODEL*3 + (l2)D_MODEL*D_MODEL) {
                rel -= (l2)D_MODEL*D_MODEL*3;
                printf("block 0 out_w[%ld][%ld]\n", rel / D_MODEL, rel % D_MODEL);
            } else {
                rel -= (l2)D_MODEL*D_MODEL*3 + (l2)D_MODEL*D_MODEL;
                if (rel < (l2)D_MODEL*D_FF) printf("block 0 ffn1_w[%ld][%ld]\n", rel / D_FF, rel % D_FF);
                else {
                    rel -= (l2)D_MODEL*D_FF;
                    if (rel < (l2)D_FF*D_MODEL) printf("block 0 ffn2_w[%ld][%ld]\n", rel / D_MODEL, rel % D_MODEL);
                    else {
                        rel -= (l2)D_FF*D_MODEL;
                        if (rel < D_MODEL) printf("block 0 ln1_gamma[%ld]\n", rel);
                        else {
                            rel -= D_MODEL;
                            if (rel < D_MODEL) printf("block 0 ln1_beta[%ld]\n", rel);
                            else printf("block 0 (unexpected)\n");
                        }
                    }
                }
            }
        } else printf("beyond block 0\n");
    }
    
    printf("\nIndex 18660 = ");
    pos = 18660;
    l2 blk0_end = emb + hp + blk;
    if (pos < blk0_end) printf("block 0\n");
    else if (pos < emb + hp + blk*2) printf("block 1 (offset %ld)\n", pos - blk0_end);
    else printf("later\n");

    printf("\nTotal params: %ld\n", total);
    printf("VOCAB=%d, D_MODEL=%d, D_FF=%d, D_HEAD=%d, N_LAYERS=%d\n",
           VOCAB, D_MODEL, D_FF, D_HEAD, N_LAYERS);
    return 0;
}
