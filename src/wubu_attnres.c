/*
 * wubu_attnres.c — Attention Residuals (AttnRes, Kimi K3) (Round-4 #413/#414/#419).
 * C11, self-contained. AttnRes lets a layer READ representations written by
 * earlier layers and WRITE its own for later layers (cross-layer residual
 * stream, sibling of mHC but attention-specific). Implemented as a read/write
 * gate over a small set of "residual slots" with identity-preserving init.
 */
#include "wubu_attnres.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

struct wubu_attnres {
    int dim;
    int n_slots;      /* number of cross-layer residual slots */
    float *read_gate; /* n_slots: how much each slot contributes to this layer's input */
    float *write_gate;/* n_slots: how much this layer's output is stored to each slot */
    float *slots;     /* n_slots * dim: the persistent cross-layer state */
};

wubu_attnres_t *wubu_attnres_create(int dim, int n_slots) {
    if (dim <= 0 || n_slots <= 0) return NULL;
    wubu_attnres_t *a = (wubu_attnres_t *)calloc(1, sizeof(*a));
    if (!a) return NULL;
    a->dim = dim; a->n_slots = n_slots;
    a->read_gate = (float *)calloc(n_slots, sizeof(float));
    a->write_gate = (float *)calloc(n_slots, sizeof(float));
    a->slots = (float *)calloc((size_t)n_slots * dim, sizeof(float));
    if (!a->read_gate || !a->write_gate || !a->slots) { wubu_attnres_free(a); return NULL; }
    /* Identity init: read gate 0 (no cross-layer injection), write gate 0 (no store). */
    return a;
}

void wubu_attnres_free(wubu_attnres_t *a) {
    if (!a) return;
    free(a->read_gate); free(a->write_gate); free(a->slots); free(a);
}

/* Verify identity property: all gates zero => this layer neither reads nor writes
 * cross-layer state (pure local residual). */
int wubu_attnres_identity_ok(const wubu_attnres_t *a) {
    if (!a) return 0;
    for (int i = 0; i < a->n_slots; i++)
        if (a->read_gate[i] != 0.0f || a->write_gate[i] != 0.0f) return 0;
    return 1;
}

/* Forward: x (dim) -> y (dim) = x + sum_s read_gate[s]*slots[s]. */
void wubu_attnres_read(const wubu_attnres_t *a, const float *x, float *y) {
    int dim = a->dim, ns = a->n_slots;
    for (int i = 0; i < dim; i++) y[i] = x[i];
    for (int s = 0; s < ns; s++) {
        float g = a->read_gate[s];
        if (g == 0.0f) continue;
        const float *slot = a->slots + (size_t)s * dim;
        for (int i = 0; i < dim; i++) y[i] += g * slot[i];
    }
}

/* Write: store (write_gate[s] * out) into slot s, for all s. */
void wubu_attnres_write(wubu_attnres_t *a, const float *out) {
    int dim = a->dim, ns = a->n_slots;
    for (int s = 0; s < ns; s++) {
        float g = a->write_gate[s];
        if (g == 0.0f) continue;
        float *slot = a->slots + (size_t)s * dim;
        for (int i = 0; i < dim; i++) slot[i] = g * out[i];
    }
}

void wubu_attnres_set_read_gate(wubu_attnres_t *a, int slot, float g) {
    if (a && slot >= 0 && slot < a->n_slots) a->read_gate[slot] = g;
}
void wubu_attnres_set_write_gate(wubu_attnres_t *a, int slot, float g) {
    if (a && slot >= 0 && slot < a->n_slots) a->write_gate[slot] = g;
}
