#ifndef WUBU_ROOFLINE_H
#define WUBU_ROOFLINE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int n_layers;
    int n_kv_heads;     /* GQA kv heads */
    int head_dim;
    int bw_bits;        /* weight precision bits */
    int bkv_bits;       /* KV precision bits */
    double beta_eff_tb_s; /* effective HBM bandwidth TB/s */
} wubu_roofline_cfg_t;

typedef enum {
    WUBU_COMPRESS_NONE = 0,
    WUBU_COMPRESS_WEIGHTS,
    WUBU_COMPRESS_KV
} wubu_compress_target_t;

wubu_roofline_cfg_t wubu_roofline_default(void);
void wubu_roofline_io(const wubu_roofline_cfg_t *c, double P_params,
                      int B, int s, double *W_gb, double *K_gb);
double wubu_roofline_bstar(const wubu_roofline_cfg_t *c, double P_params, int s);
wubu_compress_target_t wubu_roofline_advise(const wubu_roofline_cfg_t *c,
                                            double P_params, int B, int s);
double wubu_roofline_tpot_ms(const wubu_roofline_cfg_t *c, double P_params,
                             int B, int s);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_ROOFLINE_H */
