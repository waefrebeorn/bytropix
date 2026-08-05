#ifndef WUBU_MODEL_FWD_H
#define WUBU_MODEL_FWD_H

/*
 * wubu_model_fwd.h — forward declarations for the model layer.
 *
 * Lets consumers declare pointers to model types WITHOUT pulling in the
 * full wubu_model.h (and its transitive SSM/MoE/KV/arena includes).
 *
 * ADR-002: opaque structs at every module seam. Include THIS header when
 * you only need to pass model handles around; include wubu_model.h only
 * where you need the full struct layout or the model API functions.
 *
 * NOTE: the full definitions live in wubu_model.h (wubu_model_t,
 * wubu_layer_t, mtp_head_t) and wubu_ssm.h (ssm_layer_weights,
 * gqa_layer_weights), wubu_moe.h (moe_weights_t). The struct tags are
 * defined there; this header only forward-declares the typedefs.
 */

typedef struct ssm_layer_weights ssm_layer_weights;
typedef struct gqa_layer_weights gqa_layer_weights;
typedef struct moe_weights_t moe_weights_t;
typedef struct wubu_layer_t wubu_layer_t;
typedef struct mtp_head_t mtp_head_t;
typedef struct wubu_model_t wubu_model_t;

#endif /* WUBU_MODEL_FWD_H */
