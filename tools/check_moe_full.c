#include "wubu_model.h"
#include "gguf_reader.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

int main() {
    gguf_ctx *ctx = gguf_open("/home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf");
    if(!ctx) return 1;
    gguf_buffer_data(ctx);
    
    char mn[256];
    // Check all of shared expert gate_shexp for layer 0
    snprintf(mn,sizeof(mn),"blk.0.ffn_gate_shexp.weight");
    gguf_tensor_info *t = gguf_find_tensor(ctx,mn);
    if(t){
        int64_t n = (int64_t)D_MODEL * SHARED_D_FF;
        float *data = (float*)malloc(n * sizeof(float));
        gguf_dequantize((const uint8_t*)ctx->data_blob + t->data_offset, t->ggml_type, n, data);
        // Full scan
        float gmin=1e30,gmax=-1e30; int gmini=0,gmaxi=0;
        for(int64_t i=0;i<n;i++){
            if(data[i]<gmin){gmin=data[i];gmini=i;}
            if(data[i]>gmax){gmax=data[i];gmaxi=i;}
        }
        printf("gate_shexp full: min=%.2e idx=%lld max=%.2e idx=%lld\n",gmin,(long long)gmini,gmax,(long long)gmaxi);
        
        // For dim 2035 (D_MODEL=2048), check which SHARED_D_FF=512 entries contribute
        int dim = 2035;
        printf("\ngate_shexp rows contributing to output dim %d:\n",dim);
        float mx=-1e30; int mxi=-1;
        for(int k=0;k<SHARED_D_FF;k++){
            float w = data[k * D_MODEL + dim];
            if(fabsf(w)>mx){mx=fabsf(w);mxi=k;}
        }
        printf("  max |weight| = %.2e at shared_ff_idx=%d\n",mx,mxi);
        
        // Check down_shexp for same dim
        snprintf(mn,sizeof(mn),"blk.0.ffn_down_shexp.weight");
        gguf_tensor_info *t2 = gguf_find_tensor(ctx,mn);
        if(t2){
            int64_t n2 = (int64_t)SHARED_D_FF * D_MODEL;
            float *d2 = (float*)malloc(n2 * sizeof(float));
            gguf_dequantize((const uint8_t*)ctx->data_blob + t2->data_offset, t2->ggml_type, n2, d2);
            float dmin=1e30,dmax=-1e30; int dmini=0,dmaxi=0;
            for(int64_t i=0;i<n2;i++){
                if(d2[i]<dmin){dmin=d2[i];dmini=i;}
                if(d2[i]>dmax){dmax=d2[i];dmaxi=i;}
            }
            printf("down_shexp full: min=%.2e idx=%lld max=%.2e idx=%lld\n",dmin,(long long)dmini,dmax,(long long)dmaxi);
            
            // Check down weight for dim 2035: down[shared_k * D_MODEL + 2035]
            mx=-1e30; mxi=-1;
            for(int k=0;k<SHARED_D_FF;k++){
                float w = d2[k * D_MODEL + dim];
                if(fabsf(w)>mx){mx=fabsf(w);mxi=k;}
            }
            printf("  down_shexp max |weight| for dim %d = %.2e at shared_idx=%d\n",dim,mx,mxi);
        }
        free(data);
    }
    
    // Check expert gate for layer 0
    snprintf(mn,sizeof(mn),"blk.0.ffn_gate_exps.weight");
    t = gguf_find_tensor(ctx,mn);
    if(t){
        int64_t n = (int64_t)D_MODEL * D_FF;
        float *data = (float*)malloc(n * sizeof(float));
        // Expert 0
        int64_t expert_raw = gguf_raw_size(t->ggml_type, n);
        printf("expert_raw = %lld bytes\n",(long long)expert_raw);
        gguf_dequantize((const uint8_t*)ctx->data_blob + t->data_offset, t->ggml_type, n, data);
        float gmin=1e30,gmax=-1e30;
        for(int64_t i=0;i<n;i++){
            if(data[i]<gmin)gmin=data[i];
            if(data[i]>gmax)gmax=data[i];
        }
        printf("expert0 gate full: min=%.2e max=%.2e\n",gmin,gmax);
        
        // Check specifically D_MODEL x D_FF dimension
        // gate_out[j] = sum x[k] * gate_weight[k * D_FF + j]
        // For j ≈ 0..10, compute typical value with x=1
        printf("\nTypical gate_out[0..5] with x=1:\n");
        for(int j=0;j<5;j++){
            double sum=0;
            for(int k=0;k<D_MODEL;k++) sum += data[k * D_FF + j];
            printf("  gate[%d] = %.2e\n",j,sum);
        }
        
        // max possible gate_out: find the column with max sum |w|
        double max_col_sum=0; int max_col=-1;
        for(int j=0;j<D_FF;j++){
            double sum=0;
            for(int k=0;k<D_MODEL;k++) sum += fabs(data[k * D_FF + j]);
            if(sum>max_col_sum){max_col_sum=sum;max_col=j;}
        }
        printf("max col sum = %.2e at col %d\n",max_col_sum,max_col);
        free(data);
    }
    
    gguf_close(ctx);
    return 0;
}
