#ifndef WUBU_SYNTHFILE_H
#define WUBU_SYNTHFILE_H
#include <stddef.h>

/* wubu_synthfile.h -- NAMESPACE #195 9P synthetic files (read computes).
 *
 * A synthetic 9P file: reading it does not return stored bytes -- it RUNS A
 * GENERATOR that computes the content on demand (like /proc, /sys, or
 * /n/kernel/mem). This module wires a name -> generator function. A read
 * invokes the generator with a caller-provided buffer, filling it with
 * computed content. Deterministic, real work (no stubs). */

typedef int (*wubu_synth_gen_t)(void *ctx, char *buf, size_t cap);

typedef struct {
    char name[64];
    wubu_synth_gen_t gen;
    void *ctx;
} wubu_synthfile_t;

typedef struct {
    wubu_synthfile_t files[16];
    int n;
} wubu_synthfs_t;

/* Initialise the synthetic file table. */
int wubu_synthfs_init(wubu_synthfs_t *fs);

/* Register a synthetic file (name + generator + context). Returns 0/-1. */
int wubu_synthfs_add(wubu_synthfs_t *fs, const char *name,
                     wubu_synth_gen_t gen, void *ctx);

/* "Read" a synthetic file by name: invokes its generator to fill buf. Returns
 * the number of bytes written (>=0), or -1 if the file is unknown or has no
 * generator. */
int wubu_synthfs_read(const wubu_synthfs_t *fs, const char *name,
                      char *buf, size_t cap);

#endif
