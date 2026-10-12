/* wubu_grouped_query_heads.c — GQA head mapping (BRAIN #234).
 * C11, self-contained. n_q query heads share n_kv KV heads; query head q maps
 * to KV head floor(q / group_size) where group_size = n_q / n_kv.
 */
#include "wubu_grouped_query_heads.h"

int wubu_gqa_group_size(int n_q, int n_kv) {
    if (n_q <= 0 || n_kv <= 0) return -1;
    if (n_q % n_kv != 0) return -1;
    return n_q / n_kv;
}

int wubu_gqa_kv_head(int q, int n_q, int n_kv) {
    if (q < 0 || q >= n_q) return -1;
    int g = wubu_gqa_group_size(n_q, n_kv);
    if (g < 0) return -1;
    return q / g;
}
