#include "gguf_reader.h"
#include "gpu_quant_matmul.h"
#include "wubu_ssm.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define QK_K 256
#define D_MODEL 2048
#define CONV_DIM 8192
#define VALUE_DIM 4096

void dequantize_q5_k_row(const uint8_t *data, float *output, int64_t n_elems);

int main() {
    gguf_ctx *ctx = gguf_open("/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf");
    gguf_buffer_data(ctx);
    const uint8_t *blob = (const uint8_t *)ctx->data_blob;
    
    // Find layer 0 SSM weights
    gguf_tensor_info *t_qkv = gguf_find_tensor(ctx, "blk.0.attn_qkv.weight");
    gguf_tensor_info *t_gate = gguf_find_tensor(ctx, "blk.0.attn_gate.weight");
    if (!t_qkv || !t_gate) return 1;
    
    int64_t n_qkv = (int64_t)D_MODEL * CONV_DIM;
    int64_t n_gate = (int64_t)D_MODEL * VALUE_DIM;
    int64_t qkv_raw_size = gguf_raw_size(t_qkv->ggml_type, n_qkv);
    int64_t gate_raw_size = gguf_raw_size(t_gate->ggml_type, n_gate);
    
    // Generate random input
    float x[D_MODEL];
    srand(42);
    for (int i = 0; i < D_MODEL; i++) x[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    
    // Path 1: GPU Q5_K quant matmul (same as hybrid prefill)
    printf("=== GPU Q5_K quant matmul ===\n");
    cudaStream_t st; cudaStreamCreate(&st);
    uint8_t *d_qkv_w, *d_gate_w; float *d_x, *d_qkv, *d_z;
    cudaMalloc(&d_qkv_w, qkv_raw_size);
    cudaMalloc(&d_gate_w, gate_raw_size);
    cudaMalloc(&d_x, D_MODEL * 4);
    cudaMalloc(&d_qkv, CONV_DIM * 4);
    cudaMalloc(&d_z, VALUE_DIM * 4);
    cudaMemcpy(d_qkv_w, blob + t_qkv->data_offset, qkv_raw_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_gate_w, blob + t_gate->data_offset, gate_raw_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, D_MODEL * 4, cudaMemcpyHostToDevice);
    
    // Run quant matmul
    wubu_cuda_quant_matmul(d_x, d_qkv_w, GGML_TYPE_Q5_K, D_MODEL, CONV_DIM, d_qkv, NULL, 0, st);
    wubu_cuda_quant_matmul(d_x, d_gate_w, GGML_TYPE_Q5_K, D_MODEL, VALUE_DIM, d_z, NULL, 0, st);
    cudaStreamSynchronize(st);
    
    float gpu_qkv[CONV_DIM], gpu_z[VALUE_DIM];
    cudaMemcpy(gpu_qkv, d_qkv, CONV_DIM * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(gpu_z, d_z, VALUE_DIM * 4, cudaMemcpyDeviceToHost);
    
    // Path 2: F32 dequant reference (true value)
    printf("=== F32 dequant reference ===\n");
    float *qkv_f32 = (float *)malloc(n_qkv * sizeof(float));
    float *gate_f32 = (float *)malloc(n_gate * sizeof(float));
    gguf_read_tensor_f32(ctx, t_qkv, qkv_f32, n_qkv);
    gguf_read_tensor_f32(ctx, t_gate, gate_f32, n_gate);
    
    float ref_qkv[CONV_DIM], ref_z[VALUE_DIM];
    for (int j = 0; j < CONV_DIM; j++) {
        double sum = 0;
        for (int k = 0; k < D_MODEL; k++)
            sum += (double)x[k] * (double)qkv_f32[k + (int64_t)j * D_MODEL];
        ref_qkv[j] = (float)sum;
    }
    for (int j = 0; j < VALUE_DIM; j++) {
        double sum = 0;
        for (int k = 0; k < D_MODEL; k++)
            sum += (double)x[k] * (double)gate_f32[k + (int64_t)j * D_MODEL];
        ref_z[j] = (float)sum;
    }
    
    // Path 3: CPU Q8_K requant path (what CPU SSM does)
    printf("=== CPU Q8_K requant matmul ===\n");
    // Quantize weight to Q8_K on-the-fly (what wubu_ssm_forward does)
    float cpu_qkv[CONV_DIM], cpu_z[VALUE_DIM];
    {
        // Use quantized matmul with Q8_K requant
        int64_t q8_size = gguf_raw_size(GGML_TYPE_Q8_K, D_MODEL);
        uint8_t *q8_w = (uint8_t *)malloc(q8_size * CONV_DIM);
        for (int j = 0; j < CONV_DIM; j++) {
            // Quantize column j to Q8_K
            const float *col = qkv_f32 + (int64_t)j * D_MODEL;
            uint8_t *q8_col = q8_w + q8_size * j;
            // This is a simplified Q8_K quant — just use the actual bytropix quantized matmul
            // by calling the quantized_matmul function directly
        }
        free(q8_w);
        // For simplicity, use F32 dot product (which Q8_K should approximate)
        // Actually, let me just use the F32 dequant and compare
        memcpy(cpu_qkv, gpu_qkv, CONV_DIM * 4);  // Placeholder
    }
    
    // Compare
    double d_gpu = 0, d_ref = 0, n_gpu = 0, n_ref = 0;
    for (int j = 0; j < CONV_DIM; j++) {
        d_gpu += (double)gpu_qkv[j] * (double)ref_qkv[j];
        n_gpu += (double)gpu_qkv[j] * (double)gpu_qkv[j];
        n_ref += (double)ref_qkv[j] * (double)ref_qkv[j];
    }
    printf("QKV: GPU vs F32 ref cos-sim=%.10f\n", d_gpu/(sqrt(n_gpu)*sqrt(n_ref)+1e-30));
    printf("  GPU qkv[0..4]: %.6f %.6f %.6f %.6f %.6f\n", gpu_qkv[0],gpu_qkv[1],gpu_qkv[2],gpu_qkv[3],gpu_qkv[4]);
    printf("  REF qkv[0..4]: %.6f %.6f %.6f %.6f %.6f\n", ref_qkv[0],ref_qkv[1],ref_qkv[2],ref_qkv[3],ref_qkv[4]);
    
    d_gpu = 0; d_ref = 0; n_gpu = 0; n_ref = 0;
    for (int j = 0; j < VALUE_DIM; j++) {
        d_gpu += (double)gpu_z[j] * (double)ref_z[j];
        n_gpu += (double)gpu_z[j] * (double)gpu_z[j];
        n_ref += (double)ref_z[j] * (double)ref_z[j];
    }
    printf("GATE: GPU vs F32 ref cos-sim=%.10f\n", d_gpu/(sqrt(n_gpu)*sqrt(n_ref)+1e-30));
    printf("  GPU z[0..4]: %.6f %.6f %.6f %.6f %.6f\n", gpu_z[0],gpu_z[1],gpu_z[2],gpu_z[3],gpu_z[4]);
    printf("  REF z[0..4]: %.6f %.6f %.6f %.6f %.6f\n", ref_z[0],ref_z[1],ref_z[2],ref_z[3],ref_z[4]);
    
    cudaFree(d_qkv_w); cudaFree(d_gate_w); cudaFree(d_x); cudaFree(d_qkv); cudaFree(d_z);
    cudaStreamDestroy(st);
    free(qkv_f32); free(gate_f32);
    gguf_close(ctx);
    return 0;
}
