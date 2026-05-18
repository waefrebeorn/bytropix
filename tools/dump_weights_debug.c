/* Compare BOS embedding + first few weights between our model and llama.cpp */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "wubu_model.h"
#include "gguf_reader.h"

int main(int argc, char **argv) {
    const char *path = "/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf";
    wubu_model_t mdl;
    if (!wubu_model_init(&mdl, path)) return 1;
    
    int D = D_MODEL;
    float *bos_emb;
    if (mdl.use_embedding_file) {
        const char *emb_path = "data/qwen36_embeddings_c.bin.raw";
        FILE *emb_f = fopen(emb_path, "rb");
        fseek(emb_f, 248044LL * D * sizeof(float), SEEK_SET);
        bos_emb = (float *)malloc(D * sizeof(float));
        fread(bos_emb, sizeof(float), D, emb_f);
        fclose(emb_f);
    } else {
        bos_emb = mdl.token_embd + 248044LL * D;
    }
    
    // Dump BOS embedding raw
    FILE *f = fopen("/tmp/our_bos_emb.bin", "wb");
    fwrite(bos_emb, sizeof(float), D, f);
    fclose(f);
    printf("BOS emb saved\n");
    
    // Dump output weight first few rows
    // output.weight in GGUF has dims[0]=2048, dims[1]=248320
    // stored as [vocab, D_MODEL] in memory
    f = fopen("/tmp/our_outw_first.bin", "wb");
    fwrite(mdl.output_weight, sizeof(float), D * 100, f); // first 100 vocab x 2048 dims
    fclose(f);
    
    float *w = mdl.output_weight;
    printf("Output weight first 10 floats: ");
    for (int i = 0; i < 10; i++) printf("%.4f ", w[i]);
    printf("\n");
    
    // Also check embedding stats
    float em = 0, es = 0;
    for (int i = 0; i < D; i++) em += bos_emb[i];
    em /= D;
    for (int i = 0; i < D; i++) es += (bos_emb[i]-em)*(bos_emb[i]-em);
    es = sqrtf(es/D);
    printf("Embedding: mean=%.6f std=%.6f\n", em, es);
    
    wubu_model_free(&mdl);
    return 0;
}
