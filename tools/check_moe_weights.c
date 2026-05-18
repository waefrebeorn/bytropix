#include "wubu_model.h"
#include "gguf_reader.h"
#include <stdio.h>
#include <math.h>

int main() {
    gguf_ctx *ctx = gguf_open("/home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf");
    if(!ctx) return 1;
    gguf_buffer_data(ctx);
    
    char mn[256];
    for(int l=0; l < 1; l++) {
        printf("=== Layer %d ===\n", l);
        
        // Gate inp
        snprintf(mn, sizeof(mn), "blk.%d.ffn_gate_inp.weight", l);
        gguf_tensor_info *t = gguf_find_tensor(ctx, mn);
        float *dgi = (float*)malloc((int64_t)D_MODEL*N_EXPERTS*sizeof(float));
        gguf_dequantize((const uint8_t*)ctx->data_blob + t->data_offset, t->ggml_type,
            (int64_t)D_MODEL*N_EXPERTS, dgi);
        float gmin=1e30,gmax=-1e30;
        for(int i=0;i<(int64_t)D_MODEL*N_EXPERTS && i<100000;i++){
            if(dgi[i]<gmin)gmin=dgi[i];if(dgi[i]>gmax)gmax=dgi[i];
        }
        printf("  gate_inp sample 100K: min=%.2f max=%.2f\n", gmin,gmax);
        free(dgi);

        // Check shared expert
        snprintf(mn, sizeof(mn), "blk.%d.ffn_gate_shexp.weight", l);
        t = gguf_find_tensor(ctx, mn);
        if(t){
            float *gs = (float*)malloc((int64_t)D_MODEL*SHARED_D_FF*sizeof(float));
            gguf_dequantize((const uint8_t*)ctx->data_blob + t->data_offset, t->ggml_type,
                (int64_t)D_MODEL*SHARED_D_FF, gs);
            float smin=1e30,smax=-1e30;
            for(int i=0;i<(int64_t)D_MODEL*SHARED_D_FF && i<100000;i++){
                if(gs[i]<smin)smin=gs[i];if(gs[i]>smax)smax=gs[i];
            }
            printf("  gate_shexp sample 100K: min=%.2f max=%.2f\n", smin,smax);
            free(gs);
        }
        
        // Check one expert (expert 0 gate)
        snprintf(mn, sizeof(mn), "blk.%d.ffn_gate_exps.weight", l);
        t = gguf_find_tensor(ctx, mn);
        if(t){
            int64_t expert_raw = gguf_raw_size(t->ggml_type, (int64_t)D_MODEL*D_FF);
            float *eg = (float*)malloc((int64_t)D_MODEL*D_FF*sizeof(float));
            // Expert 0
            gguf_dequantize((const uint8_t*)ctx->data_blob + t->data_offset,
                t->ggml_type, (int64_t)D_MODEL*D_FF, eg);
            float emin=1e30,emax=-1e30;
            for(int i=0;i<(int64_t)D_MODEL*D_FF && i<100000;i++){
                if(eg[i]<emin)emin=eg[i];if(eg[i]>emax)emax=eg[i];
            }
            printf("  expert0 gate sample 100K: min=%.2f max=%.2f\n", emin,emax);
            free(eg);
        }
        
        // Check down expert
        snprintf(mn, sizeof(mn), "blk.%d.ffn_down_exps.weight", l);
        t = gguf_find_tensor(ctx, mn);
        if(t){
            float *ed = (float*)malloc((int64_t)D_FF*D_MODEL*sizeof(float));
            gguf_dequantize((const uint8_t*)ctx->data_blob + t->data_offset,
                t->ggml_type, (int64_t)D_FF*D_MODEL, ed);
            float dmin=1e30,dmax=-1e30;
            for(int i=0;i<(int64_t)D_FF*D_MODEL && i<100000;i++){
                if(ed[i]<dmin)dmin=ed[i];if(ed[i]>dmax)dmax=ed[i];
            }
            printf("  expert0 down sample 100K: min=%.2f max=%.2f\n", dmin,dmax);
            free(ed);
        }
    }
    
    gguf_close(ctx);
    return 0;
}
