#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main(void) {
    gguf_ctx *ctx = gguf_open("/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf");
    if (!ctx) return 1;
    gguf_buffer_data(ctx);
    
    // Find expert tensors
    gguf_tensor_info *t = gguf_find_tensor(ctx, "blk.0.ffn_gate_exps.weight");
    if (!t) { printf("NOT FOUND\n"); return 1; }
    
    printf("ffn_gate_exps: type=%d dims=[%ld,%ld,%ld] n_elems=%ld\n",
           t->ggml_type, (long)t->dims[0], (long)t->dims[1], (long)t->dims[2],
           (long)(t->dims[0]*t->dims[1]*t->dims[2]));
    
    // For MoE experts, each expert has [n_rows, n_cols, 1] slice
    // Storage: rows=n_cols, cols=n_rows, experts=n_experts
    // So expert 0: first n_cols*n_rows elements in flat array
    int n_elems_per_exp = (int)(t->dims[0] * t->dims[1]); // 2048*512 = 1048576
    int n_rows = (int)t->dims[0];  // 2048
    int n_cols = (int)t->dims[1];  // 512
    
    printf("Expert 0: %d rows x %d cols = %d elems\n", n_rows, n_cols, n_elems_per_exp);
    
    // Load expert 0 as F32 reference
    float *W_f32 = (float *)malloc(n_elems_per_exp * sizeof(float));
    if (!gguf_read_tensor_f32(ctx, t, W_f32, n_elems_per_exp)) {
        printf("F32 read failed\n");
        free(W_f32);
        // Fallback: quantized matmul only
        float *x = (float *)malloc(n_rows * sizeof(float));
        srand(42);
        for (int i = 0; i < n_rows; i++) x[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        float *qy = (float *)calloc(n_cols, sizeof(float));
        
        const uint8_t *qdata = (const uint8_t *)ctx->data_blob + t->data_offset;
        quantized_matmul(x, qdata, t->ggml_type, n_rows, n_cols, 0, qy);
        
        double sum = 0, sum2 = 0;
        for (int j = 0; j < n_cols; j++) { sum += qy[j]; sum2 += qy[j]*qy[j]; }
        printf("IQ2_XXS expert 0: first5=%.6f %.6f %.6f %.6f %.6f\n",
               (double)qy[0],(double)qy[1],(double)qy[2],(double)qy[3],(double)qy[4]);
        printf("  mean=%.4f rms=%.4f\n", sum/n_cols, sqrt(sum2/n_cols));
        
        // Test IQ3_XXS (down_exps)
        gguf_tensor_info *t2 = gguf_find_tensor(ctx, "blk.0.ffn_down_exps.weight");
        if (t2) {
            int n_rows2 = (int)t2->dims[0], n_cols2 = (int)t2->dims[1];
            printf("\nIQ3_XXS: type=%d dims=[%ld,%ld], expert slice [%dx%d]\n",
                   t2->ggml_type, (long)t2->dims[0], (long)t2->dims[1], n_rows2, n_cols2);
            const uint8_t *qdata2 = (const uint8_t *)ctx->data_blob + t2->data_offset;
            float *qy2 = (float *)calloc(n_cols2, sizeof(float));
            float *x2 = (float *)malloc(n_rows2 * sizeof(float));
            srand(99);
            for (int i = 0; i < n_rows2; i++) x2[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
            quantized_matmul(x2, qdata2, t2->ggml_type, n_rows2, n_cols2, 0, qy2);
            double s1=0,s2=0;
            for (int j = 0; j < n_cols2; j++) { s1 += qy2[j]; s2 += qy2[j]*qy2[j]; }
            printf("  first5=%.6f %.6f %.6f %.6f %.6f mean=%.4f rms=%.4f\n",
                   (double)qy2[0],(double)qy2[1],(double)qy2[2],(double)qy2[3],(double)qy2[4],
                   s1/n_cols2, sqrt(s2/n_cols2));
            free(x2); free(qy2);
        }
        
        // Compare against F32 dequant from our own dequant functions
        if (t->ggml_type == 16) { // IQ2_XXS
            float *ref_deq = (float *)malloc(n_elems_per_exp * sizeof(float));
            dequantize_iq2_xxs_row(qdata, ref_deq, n_elems_per_exp);
            float *y_ref = (float *)calloc(n_cols, sizeof(float));
            for (int j = 0; j < n_cols; j++) {
                double sum = 0.0;
                for (int i = 0; i < n_rows; i++) sum += (double)x[i] * (double)ref_deq[j * n_rows + i];
                y_ref[j] = (float)sum;
            }
            double dot=0,n1=0,n2=0,max_e=0;
            for (int j = 0; j < n_cols; j++) {
                dot += (double)qy[j] * (double)y_ref[j];
                n1 += (double)qy[j] * (double)qy[j];
                n2 += (double)y_ref[j] * (double)y_ref[j];
                double e = fabs((double)qy[j] - (double)y_ref[j]);
                if (e > max_e) max_e = e;
            }
            printf("\nIQ2_XXS vs F32 dequant: cos-sim=%.10f max_err=%.6f\n",
                   dot/(sqrt(n1)*sqrt(n2)), max_e);
            free(ref_deq); free(y_ref);
        }
        
        free(x); free(qy);
    }
    
    gguf_close(ctx);
    return 0;
}
