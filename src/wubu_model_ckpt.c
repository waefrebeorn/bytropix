/* wubu_model_ckpt.c — Model state save/restore for speculative decode rollback.
 * Extracted from wubu_model.c per ADR-002 opaque-struct seam.
 */

#include "wubu_model.h"
#include <string.h>
#include <stdlib.h>

/* Save the model's persistent SSM/conv state and cache lengths
 * for speculative decode rollback.  Lazy allocation on first call. */
bool wubu_model_checkpoint(wubu_model_t *model)
{
    if (!model->ssm_states_saved) {
        int n_layers = model->n_layers;
        int ssm_sz = n_layers * model->ssm_v_heads * model->ssm_d_state * model->ssm_d_state;
        int conv_sz = n_layers * (model->conv_kernel - 1) * model->conv_dim;
        model->ssm_states_saved = (float *)malloc((ssm_sz + conv_sz) * sizeof(float));
        if (!model->ssm_states_saved) return false;
        model->conv_states_saved = model->ssm_states_saved + ssm_sz;
    }
    int n_layers = model->n_layers;
    int ssm_sz = n_layers * model->ssm_v_heads * model->ssm_d_state * model->ssm_d_state;
    int conv_sz = n_layers * (model->conv_kernel - 1) * model->conv_dim;
    memcpy(model->ssm_states, model->ssm_states_saved, (ssm_sz + conv_sz) * sizeof(float));
    model->gqa_cache_len_saved = model->gqa_cache_len;
    model->mtp_cache_len_saved = model->mtp.cache_len;
    return true;
}

/* Restore the model to the saved checkpoint state. */
void wubu_model_rollback(wubu_model_t *model)
{
    if (!model->ssm_states_saved) return;
    int n_layers = model->n_layers;
    int ssm_sz = n_layers * model->ssm_v_heads * model->ssm_d_state * model->ssm_d_state;
    int conv_sz = n_layers * (model->conv_kernel - 1) * model->conv_dim;
    memcpy(model->ssm_states, model->ssm_states_saved, (ssm_sz + conv_sz) * sizeof(float));
    model->gqa_cache_len = model->gqa_cache_len_saved;
    model->mtp.cache_len = model->mtp_cache_len_saved;
}
