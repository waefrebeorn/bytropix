/*
 * wubu_moe_rag.h -- MoE routing (X01-X06) + retrieval-augmented KV (Y01-Y04).
 */
#ifndef WUBU_MOE_RAG_H
#define WUBU_MOE_RAG_H

/* X01 Top-K router. */
int wubu_topk_route(const float *gate, int N, int K, int *sel);
/* X02 Expert-Choice. */
int wubu_expert_choice(const float *score, int ntok, int N, int C, int *out, int *cnt);
/* X03 shared-expert aggregation. */
int wubu_shared_expert(const int *routed, int N, int K, int *out);
/* X04 sigmoid gating. */
int wubu_sigmoid_gate(const float *score, int N, float thr, int *sel);
/* X05 predictive expert prefetch. */
int wubu_expert_prefetch(const int *predicted, int np, const char *cached, int N, int *prefetch);
/* X06 capacity factor / token dropping. */
int wubu_capacity_factor(const int *expert_of, int ntok, int N, float cap, char *keep);
/* Y01 KV Packet doc id. */
int wubu_kvpacket_doc(const int *tok_doc, int n, int *doc_id);
/* Y02 RACC keep-mask. */
int wubu_racc_keep(const char *is_retrieved, int n, char *keep);
/* Y03 CAG ready. */
int wubu_cag_ready(int doc_loaded);
/* Y04 cross-document namespace. */
int wubu_crossdoc_ns(const int *tok_doc, int i);

#endif /* WUBU_MOE_RAG_H */
