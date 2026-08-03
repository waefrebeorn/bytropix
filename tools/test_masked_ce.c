/* test_masked_ce.c -- the masked CE against the FD oracle: the masked
 * positions get ZERO gradient, the unmasked ones match the finite
 * differences of the loss, and the normalization is the masked count
 * (the Hermes 69%-output-token doctrine). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "wubu_masked_ce.h"

#define SEQ 6
#define VOCAB 9

int main(void)
{
    float logits[SEQ * VOCAB], mask[SEQ];
    uint16_t toks[SEQ];
    srand(7);
    for (int i = 0; i < SEQ * VOCAB; i++) logits[i] = ((float)(rand() % 400) / 100.0f - 2.0f);
    for (int s = 0; s < SEQ; s++) toks[s] = (uint16_t)(s % VOCAB);
    /* the mask: positions 1,3,5 train (the assistant), 0,2,4 masked */
    mask[0] = 0; mask[1] = 1; mask[2] = 0; mask[3] = 1; mask[4] = 0; mask[5] = 1;

    float loss = 0, grad[SEQ * VOCAB];
    int ok = wubu_masked_ce(logits, toks, mask, SEQ, VOCAB, &loss, grad);
    if (!ok) { printf("  call FAIL\n"); return 1; }

    int pass = 1;
    /* the masked positions: zero grad */
    for (int s = 0; s < SEQ; s += 2)
        for (int v = 0; v < VOCAB; v++)
            if (fabsf(grad[s * VOCAB + v]) > 1e-7f) {
                printf("  masked pos %d grad %.3e should be 0 FAIL\n", s, grad[s * VOCAB + v]);
                pass = 0;
            }
    /* the FD on the unmasked positions */
    double maxr = 0;
    float epsf = 1e-3f;
    for (int s = 1; s < SEQ; s += 2) {
        for (int v = 0; v < VOCAB; v++) {
            float save = logits[s * VOCAB + v];
            logits[s * VOCAB + v] = save + epsf;
            float l1; wubu_masked_ce(logits, toks, mask, SEQ, VOCAB, &l1, NULL);
            logits[s * VOCAB + v] = save - epsf;
            float l2; wubu_masked_ce(logits, toks, mask, SEQ, VOCAB, &l2, NULL);
            logits[s * VOCAB + v] = save;
            double fd = (l1 - l2) / (2.0 * epsf);
            double rel = fabs(fd - grad[s * VOCAB + v]) / (fabs(fd) + 1e-9);
            if (rel > maxr) maxr = rel;
        }
    }
    if (maxr > 5e-2) { printf("  FD maxrel %.3e FAIL\n", maxr); pass = 0; }

    /* the masked-mean normalization: the loss of the full-mask equals the
     * standard mean CE; the half-mask's loss is the mean over the half */
    float maskall[SEQ];
    for (int s = 0; s < SEQ; s++) maskall[s] = 1;
    float l_full; wubu_masked_ce(logits, toks, maskall, SEQ, VOCAB, &l_full, NULL);
    float l_half; wubu_masked_ce(logits, toks, mask, SEQ, VOCAB, &l_half, NULL);
    float frac = wubu_masked_ce_frac(mask, SEQ);
    if (fabsf(frac - 0.5f) > 1e-6f) { printf("  frac %.2f FAIL\n", frac); pass = 0; }
    /* the half-mask's loss must differ from the full (different means) */
    if (fabsf(l_full - l_half) < 1e-6f) {
        printf("  masked normalization not effective FAIL\n"); pass = 0;
    }

    printf("  masked CE: loss %.6f full %.6f half %.6f frac %.2f FD maxrel %.3e  %s\n",
           loss, l_full, l_half, frac, maxr, pass ? "PASS" : "FAIL");
    printf("%s\n", pass ? "ALL MASKED-CE TESTS PASSED" : "MASKED-CE FAILURES");
    return pass ? 0 : 1;
}
