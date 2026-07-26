/*
 * wubu_ssd_moe.c -- SSD-paged MoE slot-bank (ds4-ssd technique).
 * See include/wubu_ssd_moe.h. Self-contained; C11; opaque ctx.
 *
 * Slot-bank design (mirrors Anemll ds4-ssd):
 *   - Per layer we keep `slot_bank` resident F32 expert slots.
 *   - Each expert's three matrices (gate/up/down) are stored BF16 sequentially
 *     on disk in experts.<L>.bin; slot size (bytes) is the same for all experts.
 *   - wubu_ssd_moe_get(layer,e): if e is in a slot -> hit; else pick the LRU
 *     slot, pread() the expert's bytes, dequant BF16->F32, mark resident+used.
 *   - An OS-level page cache absorbs repeated reads, so within a decode pass
 *     most experts are already warm (ds4-ssd "decode bank" behaviour).
 */
#include "wubu_ssd_moe.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdint.h>
#include <sys/stat.h>

#define SSD_MOE_MAX_LAYERS 256
#define SSD_MOE_MAX_SLOTS  256

/* BF16 <-> F32. BF16 is the top 16 bits of an F32. */
static inline uint16_t f32_to_bf16(float v) {
    uint32_t bits; memcpy(&bits, &v, 4);
    return (uint16_t)(bits >> 16);
}
static inline float bf16_to_f32(uint16_t h) {
    uint32_t bits = (uint32_t)h << 16;
    float v; memcpy(&v, &bits, 4);
    return v;
}

typedef struct {
    int      expert;      /* which expert currently occupies this slot, or -1 */
    int      lru;         /* monotonic timestamp; higher = more recently used */
    float   *gate;        /* [d_model*d_ff] F32 resident */
    float   *up;
    float   *down;
} ssd_slot_t;

typedef struct {
    int       fd;             /* experts.<L>.bin, -1 if not open */
    ssd_slot_t slots[SSD_MOE_MAX_SLOTS];
    int       n_slots;        /* = slot_bank */
    int64_t   expert_bytes;   /* BF16 bytes per expert (gate+up+down) */
} ssd_layer_t;

struct wubu_ssd_moe {
    ssd_layer_t layers[SSD_MOE_MAX_LAYERS];
    int n_layers;
    int n_experts;
    int d_model;
    int d_ff;
    int dff_dm;               /* d_ff * d_model (elems per expert matrix) */
    int slot_bank;
    char root[1024];
    long   stat_pageins;
    long   stat_hits;
    long long stat_bytes;
    long   lru_clock;
};

/* Per-expert on-disk size in BF16 bytes (3 matrices). */
static int64_t expert_bytes_calc(int d_model, int d_ff) {
    int64_t per = (int64_t)d_model * d_ff;       /* gate */
    per += (int64_t)d_model * d_ff;              /* up   */
    per += (int64_t)d_ff * d_model;              /* down */
    return per * 2;                              /* BF16 = 2 bytes */
}

static int open_layer_file(wubu_ssd_moe_t *m, int layer) {
    ssd_layer_t *sl = &m->layers[layer];
    if (sl->fd >= 0) return 1;
    char path[1200];
    snprintf(path, sizeof(path), "%s/experts.%d.bin", m->root, layer);
    sl->fd = open(path, O_RDONLY);
    return sl->fd >= 0;
}

wubu_ssd_moe_t *wubu_ssd_moe_open(const char *sidecar_dir, int slot_bank) {
    wubu_ssd_moe_t *m = (wubu_ssd_moe_t *)calloc(1, sizeof(*m));
    if (!m) return NULL;
    snprintf(m->root, sizeof(m->root), "%s", sidecar_dir);
    m->slot_bank = slot_bank < 1 ? 1 : (slot_bank > SSD_MOE_MAX_SLOTS ? SSD_MOE_MAX_SLOTS : slot_bank);

    /* Read manifest.json for dims. */
    char mp[1200]; snprintf(mp, sizeof(mp), "%s/manifest.json", sidecar_dir);
    FILE *f = fopen(mp, "r");
    if (!f) { free(m); return NULL; }
    /* Minimal parse: look for "key": value ints. */
    char buf[4096]; size_t got = fread(buf, 1, sizeof(buf)-1, f); buf[got]=0; fclose(f);
    int grab(const char*k){ const char*p=strstr(buf,k); if(!p) return -1; p+=strlen(k);
        while(*p && (*p<'0'||*p>'9') && *p!='-') p++; return (int)strtol(p,NULL,10); }
    m->n_layers = grab("\"n_layers\""); if (m->n_layers<=0) m->n_layers = grab("n_layers");
    m->n_experts= grab("\"n_experts\""); if (m->n_experts<=0) m->n_experts= grab("n_experts");
    m->d_model  = grab("\"d_model\"");  if (m->d_model<=0)  m->d_model = grab("d_model");
    m->d_ff     = grab("\"d_ff\"");     if (m->d_ff<=0)     m->d_ff = grab("d_ff");
    if (m->n_layers<=0||m->n_experts<=0||m->d_model<=0||m->d_ff<=0) { free(m); return NULL; }

    m->dff_dm = m->d_model * m->d_ff;
    int64_t eb = expert_bytes_calc(m->d_model, m->d_ff);

    for (int l = 0; l < m->n_layers && l < SSD_MOE_MAX_LAYERS; l++) {
        ssd_layer_t *sl = &m->layers[l];
        sl->fd = -1;
        sl->n_slots = m->slot_bank;
        sl->expert_bytes = eb;
        for (int s = 0; s < sl->n_slots; s++) {
            sl->slots[s].expert = -1;
            sl->slots[s].lru = 0;
            sl->slots[s].gate = (float *)malloc((size_t)m->dff_dm * sizeof(float));
            sl->slots[s].up   = (float *)malloc((size_t)m->dff_dm * sizeof(float));
            sl->slots[s].down = (float *)malloc((size_t)m->dff_dm * sizeof(float));
            if (!sl->slots[s].gate || !sl->slots[s].up || !sl->slots[s].down) {
                wubu_ssd_moe_close(m); return NULL;
            }
        }
    }
    return m;
}

int wubu_ssd_moe_get(wubu_ssd_moe_t *m, int layer, int expert, float *out[3]) {
    if (!m || layer < 0 || layer >= m->n_layers || expert < 0 || expert >= m->n_experts)
        return -1;
    ssd_layer_t *sl = &m->layers[layer];
    if (!open_layer_file(m, layer)) return -1;

    /* Slot search. */
    int hit = -1;
    for (int s = 0; s < sl->n_slots; s++)
        if (sl->slots[s].expert == expert) { hit = s; break; }
    if (hit >= 0) {
        sl->slots[hit].lru = ++m->lru_clock;
        out[0] = sl->slots[hit].gate; out[1] = sl->slots[hit].up; out[2] = sl->slots[hit].down;
        m->stat_hits++;
        return 1; /* hit */
    }

    /* Miss: choose LRU slot (or first empty). */
    int victim = 0; long best = sl->slots[0].lru;
    for (int s = 1; s < sl->n_slots; s++)
        if (sl->slots[s].lru < best) { best = sl->slots[s].lru; victim = s; }

    /* Read this expert's BF16 bytes from disk. */
    int64_t off = (int64_t)expert * sl->expert_bytes;
    size_t total = (size_t)sl->expert_bytes;
    uint8_t *raw = (uint8_t *)malloc(total);
    if (!raw) return -1;
    /* pread may return short reads; loop. */
    size_t done = 0;
    while (done < total) {
        ssize_t r = pread(sl->fd, raw + done, total - done, (off_t)(off + done));
        if (r <= 0) { free(raw); return -1; }
        done += (size_t)r;
    }
    m->stat_bytes += (long long)total;
    m->stat_pageins++;

    /* Dequant BF16 -> F32 into the slot. Layout: gate | up | down. */
    const uint16_t *b = (const uint16_t *)raw;
    int64_t n = m->dff_dm;
    for (int64_t i = 0; i < n; i++) sl->slots[victim].gate[i] = bf16_to_f32(b[i]);
    for (int64_t i = 0; i < n; i++) sl->slots[victim].up[i]   = bf16_to_f32(b[n + i]);
    for (int64_t i = 0; i < n; i++) sl->slots[victim].down[i] = bf16_to_f32(b[2*n + i]);

    free(raw);
    sl->slots[victim].expert = expert;
    sl->slots[victim].lru = ++m->lru_clock;
    out[0] = sl->slots[victim].gate; out[1] = sl->slots[victim].up; out[2] = sl->slots[victim].down;
    return 0; /* page-in */
}

int wubu_ssd_moe_n_layers(const wubu_ssd_moe_t *m){ return m?m->n_layers:0; }
int wubu_ssd_moe_n_experts(const wubu_ssd_moe_t *m){ return m?m->n_experts:0; }
int wubu_ssd_moe_d_model(const wubu_ssd_moe_t *m){ return m?m->d_model:0; }
int wubu_ssd_moe_d_ff(const wubu_ssd_moe_t *m){ return m?m->d_ff:0; }

void wubu_ssd_moe_stats(const wubu_ssd_moe_t *m, long *pageins, long *hits, long long *bytes_read){
    if(pageins)*pageins=m?m->stat_pageins:0;
    if(hits)*hits=m?m->stat_hits:0;
    if(bytes_read)*bytes_read=m?m->stat_bytes:0;
}

void wubu_ssd_moe_close(wubu_ssd_moe_t *m){
    if(!m) return;
    for(int l=0;l<m->n_layers && l<SSD_MOE_MAX_LAYERS;l++){
        ssd_layer_t *sl=&m->layers[l];
        if(sl->fd>=0) close(sl->fd);
        for(int s=0;s<sl->n_slots;s++){ free(sl->slots[s].gate); free(sl->slots[s].up); free(sl->slots[s].down); }
    }
    free(m);
}

/* ---- Packer ---- */
void wubu_ssd_moe_pack_layer(const char *sidecar_dir, int layer,
                             int n_experts, int d_model, int d_ff,
                             const float *gate, const float *up, const float *down){
    char path[1200];
    snprintf(path, sizeof(path), "%s/experts.%d.bin", sidecar_dir, layer);
    /* Append (create or extend). */
    int fd = open(path, O_WRONLY | O_CREAT, 0644);
    if (fd < 0) return;
    int64_t n = (int64_t)d_model * d_ff;
    int64_t per_expert_bytes = n * 3 * 2; /* 3 matrices, BF16 */
    uint8_t *raw = (uint8_t *)malloc((size_t)per_expert_bytes);
    if (!raw) { close(fd); return; }
    for (int e = 0; e < n_experts; e++) {
        const float *g = gate + e*(size_t)n;
        const float *u = up   + e*(size_t)n;
        const float *d = down + e*(size_t)n;
        uint16_t *b = (uint16_t *)raw;
        for (int64_t i = 0; i < n; i++) b[i]       = f32_to_bf16(g[i]);
        for (int64_t i = 0; i < n; i++) b[n + i]   = f32_to_bf16(u[i]);
        for (int64_t i = 0; i < n; i++) b[2*n + i] = f32_to_bf16(d[i]);
        size_t off = (size_t)e * (size_t)per_expert_bytes;
        /* pwrite at absolute offset (file may already hold earlier experts). */
        size_t done = 0;
        while (done < (size_t)per_expert_bytes) {
            ssize_t w = pwrite(fd, raw + done, (size_t)per_expert_bytes - done, (off_t)(off + done));
            if (w <= 0) break;
            done += (size_t)w;
        }
    }
    free(raw);
    close(fd);
}

void wubu_ssd_moe_write_manifest(const char *sidecar_dir, int n_layers,
                                 int n_experts, int d_model, int d_ff,
                                 int n_active, int slot_bank){
    char mp[1200];
    snprintf(mp, sizeof(mp), "%s/manifest.json", sidecar_dir);
    FILE *f = fopen(mp, "w");
    if (!f) return;
    fprintf(f, "{\n");
    fprintf(f, "  \"n_layers\": %d,\n", n_layers);
    fprintf(f, "  \"n_experts\": %d,\n", n_experts);
    fprintf(f, "  \"d_model\": %d,\n", d_model);
    fprintf(f, "  \"d_ff\": %d,\n", d_ff);
    fprintf(f, "  \"n_active\": %d,\n", n_active);
    fprintf(f, "  \"slot_bank\": %d,\n", slot_bank);
    fprintf(f, "  \"fmt\": \"bf16\",\n");
    fprintf(f, "  \"technique\": \"ds4-ssd slot-bank: dense resident, routed experts paged from SSD\"\n");
    fprintf(f, "}\n");
    fclose(f);
}
