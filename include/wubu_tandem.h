/*
 * wubu_tandem.h — N64 RCP-style tandem pipeline for inference (doc "tandem").
 *
 * The Nintendo 64 Reality Coprocessor had two stages: the RSP (geometry/reality
 * signal processor, ran vertex transforms) feeding the RDP (reality display
 * processor, ran rasterization) over a FIFO. They ran IN TANDEM: while the RDP
 * drew the previous frame, the RSP prepared the next. We mirror that:
 *
 *   STAGE A ("RSP"):  compute-bound prefill / long-GEMM work (the "geometry").
 *   STAGE B ("RDP"):  latency-bound decode GEMV / attention (the "pixels").
 *
 * A ping-pong handoff buffer carries KV/activations from A to B. While B is
 * emitting tokens for step N, A is already prefilling step N+1. This is the
 * PD-disaggregation idea (D03) lifted to a continuous two-stage pipeline, with
 * a frame-budget governor (see wubu_gamebud) keeping each "frame" (decode step)
 * inside its time slice.
 *
 * Self-contained C11 + POSIX threads. No third-party deps.
 */
#ifndef WUBU_TANDEM_H
#define WUBU_TANDEM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*wubu_tandem_fn)(void *arg, int stage, int frame);

typedef struct wubu_tandem wubu_tandem_t;

/* Create a tandem engine.
 * n_a  : RSP-stage (prefill) worker count
 * n_b  : RDP-stage (decode) worker count
 * a_cores / b_cores: optional pinned core lists (NULL = OS free)
 * ring : handoff ring-buffer depth between stages (>=2; ping-pong uses 2) */
wubu_tandem_t *wubu_tandem_create(int n_a, int n_b,
                                  const char *a_cores, const char *b_cores,
                                  int ring);

void wubu_tandem_free(wubu_tandem_t *t);

/* Register the A (prefill) and B (decode) stage callbacks. */
void wubu_tandem_set_a(wubu_tandem_t *t, wubu_tandem_fn fn);
void wubu_tandem_set_b(wubu_tandem_t *t, wubu_tandem_fn fn);

/* Submit a frame: arg is carried through both stages. Blocks until both stages
 * finished this frame. Returns 0 on success, -1 on shutdown. */
int wubu_tandem_submit(wubu_tandem_t *t, void *arg);

/* Block until all submitted frames have been fully consumed by stage B.
 * Call before wubu_tandem_stats to guarantee final counts. */
void wubu_tandem_drain(wubu_tandem_t *t);

/* Stats: frames completed, A-stage busy ticks, B-stage busy ticks. */
void wubu_tandem_stats(const wubu_tandem_t *t,
                       uint64_t *frames, uint64_t *a_busy, uint64_t *b_busy);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_TANDEM_H */
