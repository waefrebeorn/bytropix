#ifndef WUBU_MOE_H
#define WUBU_MOE_H

#include <stdbool.h>

#include "gguf_reader.h"

#ifdef __cplusplus
extern "C" {
#endif

// MoE hyperparameters for Qwen3.6-35B-A3B
#define N_EXPERTS       256   // total routed experts
#define N_ACTIVE_EXPTS  8     // top-k experts per token
#define D_FF            512   // expert intermediate dimension
#define SHARED_D_FF     512   // shared expert intermediate dimension

// MoE weights for one layer
typedef struct {
    // Router
    float *ffn_gate_inp;      // [D_MODEL, N_EXPERTS] = [2048, 256] — router weight
    
    // Routed experts (3D tensors, expert index is slowest dim)
    float *ffn_gate_exps;     // [D_MODEL, D_FF, N_EXPERTS] = [2048, 512, 256]
    float *ffn_up_exps;       // [D_MODEL, D_FF, N_EXPERTS] = [2048, 512, 256]
    float *ffn_down_exps;     // [D_FF, D_MODEL, N_EXPERTS] = [512, 2048, 256]
    
    // Shared expert (always active)
    float *ffn_gate_shexp;    // [D_MODEL, SHARED_D_FF] = [2048, 512]
    float *ffn_up_shexp;      // [D_MODEL, SHARED_D_FF] = [2048, 512]
    float *ffn_down_shexp;    // [SHARED_D_FF, D_MODEL] = [512, 2048]
    
    // Router bias for shared expert
    float *ffn_gate_inp_shexp; // [D_MODEL] — not used in forward, kept for completeness
    
    // Whether weights are loaded
    bool loaded;
} moe_weights_t;

// MoE forward pass for one layer
// x: [B, T, D_MODEL] — input (post-attention normalized)
// output: [B, T, D_MODEL] — MoE output
void wubu_moe_forward(const float *x, int B, int T,
                      const moe_weights_t *w,
                      float *output);

// Load one layer's MoE weights from an open GGUF context
// Allocates and dequantizes all 3 expert tensors (O(3 GB))
// Caller must free with wubu_moe_free_layer after use
// Returns 1 on success, 0 on failure
int wubu_moe_load_layer(gguf_ctx *ctx, int layer, moe_weights_t *moe);

// Free one layer's MoE weights
void wubu_moe_free_layer(moe_weights_t *moe);

// Helper: compute router logits and select top-k experts
// scores: [B*T, N_EXPERTS] — output router logits
void wubu_moe_router(const float *x, int B, int T,
                     const float *gate_inp,
                     float *scores);

// MoE backward pass
// d_output: [B*T, D_MODEL] — gradient at MoE output (FFN path)
// normed2: [B*T, D_MODEL] — saved MoE input
// d_normed2: [B*T, D_MODEL] — output gradient w.r.t. MoE input
// weight_grad_bufs: pre-allocated zero-initialized gradient buffers (or NULL to skip)
void wubu_moe_backward(const float *d_output, int B, int T,
                       const float *normed2,
                       const moe_weights_t *w,
                       float *d_normed2,
                       float *d_gate_inp,
                       float *d_gate_exps,
                       float *d_up_exps,
                       float *d_down_exps,
                       float *d_gate_shexp,
                       float *d_up_shexp,
                       float *d_down_shexp);

#ifdef __cplusplus
}
#endif

#endif // WUBU_MOE_H
