#include "tokenizer.h"
#include "hashmind_model.h"
#include "hashmind_data.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main() {
    Tokenizer tok;
    tokenizer_init(&tok);

    HashMindModel model;
    HashMindGrad grad;
    HashMindMomentum vel;
    hashmind_model_init(&model);
    hashmind_zero_grad(&grad);
    memset(&vel,0,sizeof(vel));
    
    TrainCtx ctx = {.lr=0.001f, .momentum=0.9f, .weight_decay=1e-4f, .step=0, .model=&model, .grad=&grad, .vel=&vel};

    printf("Loading data...\n"); fflush(stdout);
    FILE* f = fopen("training_sample.txt","rb");
    if (!f) { printf("FAIL: cannot open\n"); return 1; }
    fseek(f,0,SEEK_END); long fsize=ftell(f); rewind(f);
    char* text=(char*)malloc(fsize+1);
    size_t nread = fread(text,1,fsize,f);
    if (nread != (size_t)fsize) { printf("FAIL: short read %zu vs %ld\n", nread, fsize); return 1; }
    text[fsize]=0; fclose(f);
    printf("Data: %ld bytes\n", fsize); fflush(stdout);

    TextData td;
    textdata_init(&td,&tok,text);
    printf("Tokens: %d\n", td.num_tokens); fflush(stdout);

    /* Just do a quick small test */
    printf("Running 15000 steps...\n"); fflush(stdout);
    int nan_steps=0;
    int last_log=0;
    for (int step=0; step<15000; step++) {
        TrainExample ex;
        if (!textdata_next(&td,&ex)) { textdata_reset(&td); textdata_next(&td,&ex); }

        float logits[VOCAB_SIZE];
        BlockActs acts;
        hashmind_forward(&model, ex.input_hashes, CONTEXT_LEN,
                         ex.input_tokens, CONTEXT_LEN, logits, &acts);
        float loss = nn_cross_entropy_loss(logits, ex.target_token, VOCAB_SIZE);
        
        if (isnan(loss) || isinf(loss)) { nan_steps++; continue; }

        float dlogits[VOCAB_SIZE];
        nn_cross_entropy_grad(logits, ex.target_token, VOCAB_SIZE, dlogits);
        hashmind_backward(&model, &grad, dlogits, ex.input_hashes, CONTEXT_LEN,
                          ex.input_tokens, CONTEXT_LEN, &acts);
        hashmind_apply_gradients(&ctx);

        if (step % 1000 == 0) {
            printf("  step %d: loss=%.4f\n", step, loss);
            fflush(stdout);
        }
    }
    printf("Done (nan=%d)\n", nan_steps);
    
    free(text);
    textdata_free(&td);
    return 0;
}
