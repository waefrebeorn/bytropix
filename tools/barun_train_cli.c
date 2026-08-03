/*
 * barun_train_cli.c -- the AGI training loop runner (the seed grows).
 *
 * Reads uint16 token streams (.tok, produced by barun_tokenc from the
 * corpus on the SD card), trains BarunLM-35M with the reference recipe
 * (Muon for matrices, AdamW for the embedding/norms, mean-reduced CE),
 * and checkpoints to the SD card every N steps.
 *
 * Usage:
 *   barun_train_cli --model models/barun/model.safetensors
 *                    --tok /home/wubu/sdcard/corpus/tokens/*.tok
 *                    --steps 100 --lr 1e-4 --out /home/wubu/sdcard/corpus/checkpoints/seed-1.st
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "wubu_barun.h"
#include "wubu_barun_train.h"

static const char *arg_get(int argc, char **argv, const char *name,
                           const char *def)
{
    for (int i = 1; i < argc - 1; i++)
        if (strcmp(argv[i], name) == 0) return argv[i + 1];
    return def;
}
static int arg_int(int argc, char **argv, const char *name, int def)
{
    const char *v = arg_get(argc, argv, name, NULL);
    return v ? atoi(v) : def;
}
static float arg_float(int argc, char **argv, const char *name, float def)
{
    const char *v = arg_get(argc, argv, name, NULL);
    return v ? (float)atof(v) : def;
}

/* read a .tok stream into a buffer; returns the token count. */
static long read_tokens(const char *path, uint16_t *buf, long cap)
{
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    long n = 0;
    while (n < cap && fread(&buf[n], 2, 1, f) == 1) n++;
    fclose(f);
    return n;
}

/* write the raw weights to a checkpoint file (a simple float dump --
 * the safetensors save is the next milestone). */
static int save_checkpoint(const barun_model_t *m, const char *path)
{
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    /* header: magic + param count */
    uint32_t magic = 0xBA000001u;
    fwrite(&magic, 4, 1, f);
    long n = barun_parameter_count(m);
    fwrite(&n, sizeof(long), 1, f);
    fwrite(m->embedding, sizeof(float), 16384 * 448, f);
    fwrite(m->final_norm, sizeof(float), 448, f);
    for (int i = 0; i < BARUN_LAYERS; i++) {
        barun_block_t *b = (barun_block_t *)&m->blocks[i];
        fwrite(b->q_proj, sizeof(float), 448 * 448, f);
        fwrite(b->k_proj, sizeof(float), 448 * 64, f);
        fwrite(b->v_proj, sizeof(float), 448 * 64, f);
        fwrite(b->o_proj, sizeof(float), 448 * 448, f);
        fwrite(b->g_proj, sizeof(float), 448 * 448, f);
        fwrite(b->q_norm, sizeof(float), 64, f);
        fwrite(b->k_norm, sizeof(float), 64, f);
        fwrite(b->attn_norm, sizeof(float), 448, f);
        fwrite(b->gate_up, sizeof(float), 448 * 2456, f);
        fwrite(b->down, sizeof(float), 1228 * 448, f);
        fwrite(b->ffn_norm, sizeof(float), 448, f);
    }
    for (int i = 0; i < BARUN_SELECTORS; i++)
        fwrite(m->selectors[i], sizeof(float), 448, f);
    fclose(f);
    return 0;
}

int main(int argc, char **argv)
{
    const char *model_path = arg_get(argc, argv, "--model",
                                     "models/barun/model.safetensors");
    const char *tok_glob = arg_get(argc, argv, "--tok",
                                   "/home/wubu/sdcard/corpus/tokens/cosmopedia-v2-00000.tok");
    const char *out_path = arg_get(argc, argv, "--out",
                                   "/home/wubu/sdcard/corpus/checkpoints/seed.st");
    int max_steps = arg_int(argc, argv, "--steps", 50);
    float lr = arg_float(argc, argv, "--lr", 1e-4f);
    int seq = arg_int(argc, argv, "--seq", 128);
    int ckpt_every = arg_int(argc, argv, "--ckpt", 10);

    printf("barun_train_cli: loading %s ...\n", model_path);
    barun_model_t m;
    if (barun_load(&m, model_path) != 0) {
        fprintf(stderr, "cannot load %s\n", model_path);
        return 1;
    }
    printf("barun_train_cli: %ld parameters\n", barun_parameter_count(&m));

    /* load the corpus (expand the glob via a helper: we accept ONE file
     * for now; the multi-file loop is the next step) */
    uint16_t *corpus = (uint16_t *)malloc(sizeof(uint16_t) * (1 << 22));
    long corpus_n = read_tokens(tok_glob, corpus, 1 << 22);
    if (corpus_n <= 0) {
        fprintf(stderr, "cannot read corpus %s\n", tok_glob);
        return 1;
    }
    printf("barun_train_cli: corpus %ld tokens\n", corpus_n);

    barun_buf_t b;
    if (barun_buf_alloc(&b, BARUN_MAX_SEQ) != 0) return 1;
    barun_train_t tr;
    if (barun_train_init(&tr, &m) != 0) return 1;

    barun_train_cfg_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.lr = lr;
    cfg.weight_decay = 0.1f;
    cfg.muon_momentum = 0.95f;
    cfg.warmup_steps = (uint32_t)(max_steps / 10);
    cfg.max_steps = (uint32_t)max_steps;

    /* the training loop: sliding windows over the corpus */
    uint16_t win[BARUN_MAX_SEQ];
    long pos = 0;
    double loss_ema = -1;
    for (int step = 1; step <= max_steps; step++) {
        if (pos + seq > corpus_n) pos = 0;   /* epoch wrap */
        for (int i = 0; i < seq; i++) win[i] = corpus[pos + i];
        pos += seq;
        float loss = barun_train_step_loop(&m, &tr, &b, win, (size_t)seq,
                                           &cfg, (uint32_t)step);
        loss_ema = loss_ema < 0 ? loss : 0.9 * loss_ema + 0.1 * loss;
        if (step % 5 == 0 || step == 1)
            printf("  step %4d: loss %.4f (ema %.4f)\n", step, loss, loss_ema);
        if (step % ckpt_every == 0) {
            char ck[512];
            snprintf(ck, sizeof(ck), "%s-%04d.st", out_path, step);
            if (save_checkpoint(&m, ck) == 0)
                printf("  checkpoint -> %s\n", ck);
        }
    }
    if (save_checkpoint(&m, out_path) == 0)
        printf("final checkpoint -> %s\n", out_path);

    barun_train_free(&tr);
    barun_free(&m, &b);
    free(corpus);
    printf("barun_train_cli: done\n");
    return 0;
}
