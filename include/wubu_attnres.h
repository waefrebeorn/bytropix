#ifndef WUBU_ATTNRES_H
#define WUBU_ATTNRES_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct wubu_attnres wubu_attnres_t;

/* Attention Residuals: cross-layer read/write of a small residual-slot buffer. */
wubu_attnres_t *wubu_attnres_create(int dim, int n_slots);
void wubu_attnres_free(wubu_attnres_t *a);
int  wubu_attnres_identity_ok(const wubu_attnres_t *a);
void wubu_attnres_read(const wubu_attnres_t *a, const float *x, float *y);
void wubu_attnres_write(wubu_attnres_t *a, const float *out);
/* Public gate setters (struct is opaque). */
void wubu_attnres_set_read_gate(wubu_attnres_t *a, int slot, float g);
void wubu_attnres_set_write_gate(wubu_attnres_t *a, int slot, float g);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_ATTNRES_H */
