/*
 * wubu_pd_serve.h -- Disaggregated PD serving (AB01-AB06) + dynamic depth (AC01-AC03).
 */
#ifndef WUBU_PD_SERVE_H
#define WUBU_PD_SERVE_H

/* AB01 pool split configured. */
int wubu_pd_split(int n_prefill, int n_decode);
/* AB02 KV handoff ready. */
int wubu_kv_handoff_ready(int prefill_done, int prompt_len);
/* AB03 pull-based decode routing. */
int wubu_pull_route(int decode_qlen, int high_water);
/* AB04 heterogeneous tier mapping. */
void wubu_hetero_map(int *prefill_tier, int *decode_tier);
/* AB05 KV transfer fits TTFT budget. */
int wubu_kv_xfer_fits(double kv_bytes, double bandwidth, double ttft_budget);
/* AB06 prefix-aware reuse. */
int wubu_prefix_reuse(unsigned req_hash, unsigned cache_hash);
/* AC01 MoD layer-skip. */
int wubu_mod_execute(float gate, float thr);
/* AC02 mixture-of-depths capacity. */
int wubu_mod_capacity(int depth, int cap);
/* AC03 early-exit confidence. */
int wubu_early_exit(float conf, float thr);

#endif /* WUBU_PD_SERVE_H */
