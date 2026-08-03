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
#include "wubu_grow.h"
#include "wubu_plateau.h"

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

/* load a checkpoint dump into a freshly built model (the inverse of
 * save_checkpoint: magic + param count + the raw weights). Returns 0 on
 * success (the model owns the allocated buffers). */
static int load_checkpoint(barun_model_t *m, const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    uint32_t magic = 0;
    if (fread(&magic, 4, 1, f) != 1 ||
        (magic != 0xBA000001u && magic != 0xBA000002u)) { fclose(f); return -1; }
    int nl = 0;
    if (magic == 0xBA000002u) {
        if (fread(&nl, 4, 1, f) != 1) { fclose(f); return -1; }
        if (nl < 1 || nl > BARUN_LAYERS) { fclose(f); return -1; }
    }
    long n = 0;
    if (fread(&n, sizeof(long), 1, f) != 1) { fclose(f); return -1; }
    /* build the model from fresh buffers (the released sizes) */
    float *embedding = (float *)malloc(sizeof(float) * 16384 * 448);
    float *final_norm = (float *)malloc(sizeof(float) * 448);
    float **sel = (float **)calloc(BARUN_SELECTORS, sizeof(float *));
    barun_block_t *blocks = (barun_block_t *)calloc(BARUN_LAYERS, sizeof(barun_block_t));
    if (!embedding || !final_norm || !sel || !blocks) { fclose(f); return -1; }
    for (int i = 0; i < BARUN_SELECTORS; i++) sel[i] = (float *)malloc(sizeof(float) * 448);
    barun_block_t *b = blocks;
    for (int i = 0; i < BARUN_LAYERS; i++, b++) {
        b->q_proj    = (float *)malloc(sizeof(float) * 448 * 448);
        b->k_proj    = (float *)malloc(sizeof(float) * 448 * 64);
        b->v_proj    = (float *)malloc(sizeof(float) * 448 * 64);
        b->o_proj    = (float *)malloc(sizeof(float) * 448 * 448);
        b->g_proj    = (float *)malloc(sizeof(float) * 448 * 448);
        b->q_norm    = (float *)malloc(sizeof(float) * 64);
        b->k_norm    = (float *)malloc(sizeof(float) * 64);
        b->attn_norm = (float *)malloc(sizeof(float) * 448);
        b->gate_up   = (float *)malloc(sizeof(float) * 448 * 2456);
        b->down      = (float *)malloc(sizeof(float) * 1228 * 448);
        b->ffn_norm  = (float *)malloc(sizeof(float) * 448);
        if (!b->q_proj || !b->k_proj || !b->v_proj || !b->o_proj || !b->g_proj ||
            !b->q_norm || !b->k_norm || !b->attn_norm || !b->gate_up || !b->down ||
            !b->ffn_norm) { fclose(f); return -1; }
    }
    if (fread(embedding, sizeof(float), 16384 * 448, f) != 16384 * 448 ||
        fread(final_norm, sizeof(float), 448, f) != 448) { fclose(f); return -1; }
    b = blocks;
    for (int i = 0; i < BARUN_LAYERS; i++, b++) {
        if (fread(b->q_proj, sizeof(float), 448 * 448, f) != 448 * 448 ||
            fread(b->k_proj, sizeof(float), 448 * 64, f) != 448 * 64 ||
            fread(b->v_proj, sizeof(float), 448 * 64, f) != 448 * 64 ||
            fread(b->o_proj, sizeof(float), 448 * 448, f) != 448 * 448 ||
            fread(b->g_proj, sizeof(float), 448 * 448, f) != 448 * 448 ||
            fread(b->q_norm, sizeof(float), 64, f) != 64 ||
            fread(b->k_norm, sizeof(float), 64, f) != 64 ||
            fread(b->attn_norm, sizeof(float), 448, f) != 448 ||
            fread(b->gate_up, sizeof(float), 448 * 2456, f) != 448 * 2456 ||
            fread(b->down, sizeof(float), 1228 * 448, f) != 1228 * 448 ||
            fread(b->ffn_norm, sizeof(float), 448, f) != 448) { fclose(f); return -1; }
    }
    for (int i = 0; i < BARUN_SELECTORS; i++)
        if (fread(sel[i], sizeof(float), 448, f) != 448) { fclose(f); return -1; }
    fclose(f);
    if (barun_model_init(m, embedding, final_norm, blocks, sel) != 0) return -1;
    if (nl > 0) m->n_layers = nl;   /* the v2 progressive state */
    /* the dump's count is the ACTIVE count (a v1 progressive checkpoint
     * saved with fewer layers); it must not EXCEED the built full count */
    if (n > barun_parameter_count(m)) { fprintf(stderr, "checkpoint count mismatch (%ld vs %ld)\n", n, barun_parameter_count(m)); return -1; }
    return 0;
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
    /* header: magic v2 (the n_layers) + param count */
    uint32_t magic = 0xBA000002u;
    fwrite(&magic, 4, 1, f);
    int nl = m->n_layers;
    fwrite(&nl, 4, 1, f);
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
    const char *resume = arg_get(argc, argv, "--resume", NULL);
    int max_steps = arg_int(argc, argv, "--steps", 50);
    float lr = arg_float(argc, argv, "--lr", 1e-4f);
    float muon_lr = arg_float(argc, argv, "--muon-lr", 1e-3f);
    float adam_lr = arg_float(argc, argv, "--adam-lr", 1e-3f);
    int seq = arg_int(argc, argv, "--seq", 128);
    int ckpt_every = arg_int(argc, argv, "--ckpt", 10);
    int grow_check = arg_int(argc, argv, "--grow-check", 0);
    int base_layers = arg_int(argc, argv, "--base-layers", 0);

    barun_model_t m;
    if (resume) {
        printf("barun_train_cli: resuming from %s ...\n", resume);
        if (load_checkpoint(&m, resume) != 0) {
            fprintf(stderr, "cannot load checkpoint %s\n", resume);
            return 1;
        }
    } else {
        printf("barun_train_cli: loading %s ...\n", model_path);
        if (barun_load(&m, model_path) != 0) {
            fprintf(stderr, "cannot load %s\n", model_path);
            return 1;
        }
    }
    printf("barun_train_cli: %ld parameters\n", barun_parameter_count(&m));
    if (base_layers > 0) {
        if (base_layers > BARUN_LAYERS) base_layers = BARUN_LAYERS;
        m.n_layers = base_layers;   /* the progressive start (Bu: start small) */
        printf("barun_train_cli: progressive start at %d layers\n", m.n_layers);
    }

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
    cfg.muon_lr = muon_lr;    /* the Moonlight RMS-0.2 scale makes the Muon
                                 group LR comparable to AdamW (recipe:
                                 2e-2 from scratch; 1e-3 for fine-tuning
                                 the released checkpoint) */
    cfg.adam_lr = adam_lr;
    cfg.weight_decay = 0.1f;
    cfg.muon_momentum = 0.95f;
    cfg.grad_clip = 1.0f;     /* the recipe's global-norm clip */
    cfg.warmup_steps = (uint32_t)(max_steps / 10);
    cfg.max_steps = (uint32_t)max_steps;

    /* the training loop: sliding windows over the corpus */
    uint16_t win[BARUN_MAX_SEQ];
    long pos = 0;
    double loss_ema = -1;
    float loss_hist[64];
    int hist_n = 0;
    for (int step = 1; step <= max_steps; step++) {
        if (pos + seq > corpus_n) pos = 0;   /* epoch wrap */
        for (int i = 0; i < seq; i++) win[i] = corpus[pos + i];
        pos += seq;
        float loss = barun_train_step_loop(&m, &tr, &b, win, (size_t)seq,
                                           &cfg, (uint32_t)step);
        loss_ema = loss_ema < 0 ? loss : 0.9 * loss_ema + 0.1 * loss;
        if (grow_check > 0 && step % grow_check == 0) {
            loss_hist[hist_n % 64] = (float)loss_ema;
            hist_n++;
            if (hist_n >= 32 && m.n_layers < BARUN_LAYERS &&
                /* the adaptive threshold: the absolute floor OR 0.5% of
                 * the loss magnitude -- the 0.001 floor alone sat below
                 * the fine-tune-scale noise (~2.8 loss, ~0.02 slope
                 * noise) and the growth never fired */
                wubu_plateau_detect(loss_hist, hist_n > 64 ? 64 : hist_n,
                                    32, 0.001f > 0.005f * (float)loss_ema
                                        ? 0.001f : 0.005f * (float)loss_ema)) {
                int pos_g = m.n_layers / 2;   /* the progressive deepening */
                if (wubu_grow_insert_block(&m, pos_g) &&
                    wubu_train_grow(&tr, pos_g, m.n_layers)) {
                    printf("  GROW at step %d: n_layers %d -> %d (plateau)\n",
                           step, m.n_layers - 1, m.n_layers);
                }
            }
        }
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
