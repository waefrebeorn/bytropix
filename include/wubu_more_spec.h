/*
 * wubu_more_spec.h -- Remaining speculative-decoding policy variants
 * (M07/M08/M09/M10/M15/M17/M18/M19/M20). Pure policy functions.
 */
#ifndef WUBU_MORE_SPEC_H
#define WUBU_MORE_SPEC_H

int wubu_rest_accept(float residual, float thr);
int wubu_tree_restructure(int accepted_depth, int budget);
int wubu_contrastive_accept(float draft_p, float ref_p);
int wubu_distil_gate(float quality_est, float qmin);
int wubu_spec_moe_skip(float route_score, float floor);
int wubu_cascade_accept(int draft_match);
int wubu_swap_check(int *streak, float acceptance, float low, int patience);
int wubu_layer_resume(int streamed, int total);
int wubu_cascade_earlyexit(int draft_match, int layer, int floor_layer);

#endif /* WUBU_MORE_SPEC_H */
