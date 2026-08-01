/*
 * wubu_ecs.c — ECS-style component store for engine state (doc C06). C11.
 */
#include "wubu_ecs.h"
#include <stdlib.h>
#include <string.h>

typedef struct {
    char    name[48];
    size_t  size;
    void   *data;
    void  (*dtor)(void *);
} ecs_comp_t;

struct wubu_ecs {
    int       cap;
    int       n;
    ecs_comp_t *c;
};

wubu_ecs_t *wubu_ecs_create(int cap) {
    if (cap < 1) cap = 1;
    wubu_ecs_t *e = (wubu_ecs_t *)calloc(1, sizeof(*e));
    if (!e) return NULL;
    e->cap = cap;
    e->n = 0;
    e->c = (ecs_comp_t *)calloc(cap, sizeof(ecs_comp_t));
    if (!e->c) { free(e); return NULL; }
    return e;
}

void wubu_ecs_free(wubu_ecs_t *e) {
    if (!e) return;
    for (int i = 0; i < e->n; i++) {
        if (e->c[i].dtor) e->c[i].dtor(e->c[i].data);
        else free(e->c[i].data);
    }
    free(e->c);
    free(e);
}

static int find_name(wubu_ecs_t *e, const char *name) {
    for (int i = 0; i < e->n; i++)
        if (strcmp(e->c[i].name, name) == 0) return i;
    return -1;
}

int wubu_ecs_add(wubu_ecs_t *e, const char *name, size_t size, void (*dtor)(void *)) {
    if (!e || !name || size == 0) return -1;
    if (find_name(e, name) >= 0) return -1;        /* name clash */
    if (e->n >= e->cap) return -1;                   /* full */
    int id = e->n++;
    snprintf(e->c[id].name, sizeof(e->c[id].name), "%s", name);
    e->c[id].size = size;
    e->c[id].dtor = dtor;
    e->c[id].data = calloc(1, size);
    if (!e->c[id].data) { e->n--; return -1; }
    return id;
}

int wubu_ecs_find(wubu_ecs_t *e, const char *name) {
    if (!e || !name) return -1;
    return find_name(e, name);
}

void *wubu_ecs_get(wubu_ecs_t *e, int cid) {
    if (!e || cid < 0 || cid >= e->n) return NULL;
    return e->c[cid].data;
}

size_t wubu_ecs_size(wubu_ecs_t *e, int cid) {
    if (!e || cid < 0 || cid >= e->n) return 0;
    return e->c[cid].size;
}

const char *wubu_ecs_name(wubu_ecs_t *e, int cid) {
    if (!e || cid < 0 || cid >= e->n) return NULL;
    return e->c[cid].name;
}

int wubu_ecs_count(wubu_ecs_t *e) { return e ? e->n : 0; }

uint8_t *wubu_ecs_snapshot(wubu_ecs_t *e, size_t *out_bytes) {
    if (!e || !out_bytes) return NULL;
    /* layout: [n:int32][ for each: name(48) + size(int64) + bytes ] */
    size_t total = 4;
    for (int i = 0; i < e->n; i++) total += 48 + 8 + e->c[i].size;
    uint8_t *buf = (uint8_t *)malloc(total);
    if (!buf) return NULL;
    size_t off = 0;
    memcpy(buf + off, &e->n, 4); off += 4;
    for (int i = 0; i < e->n; i++) {
        memcpy(buf + off, e->c[i].name, 48); off += 48;
        memcpy(buf + off, &e->c[i].size, 8); off += 8;
        memcpy(buf + off, e->c[i].data, e->c[i].size); off += e->c[i].size;
    }
    *out_bytes = total;
    return buf;
}

int wubu_ecs_restore(wubu_ecs_t *e, const uint8_t *buf, size_t bytes) {
    if (!e || !buf || bytes < 4) return -1;
    size_t off = 0;
    int32_t n; memcpy(&n, buf + off, 4); off += 4;
    if (n != e->n) return -1;                         /* component set mismatch */
    for (int i = 0; i < n; i++) {
        char name[48]; memcpy(name, buf + off, 48); off += 48;
        int64_t sz;   memcpy(&sz, buf + off, 8);     off += 8;
        if (strcmp(name, e->c[i].name) != 0) return -1;
        if ((size_t)sz != e->c[i].size) return -1;
        memcpy(e->c[i].data, buf + off, e->c[i].size); off += e->c[i].size;
    }
    if (off != bytes) return -1;
    return 0;
}
