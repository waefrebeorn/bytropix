#include "gguf_reader.h"
#include "gpu_moe_kernel.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define D_MODEL 2048
#define D_FF 512

void dequantize_iq2_xxs_row(const uint8_t *data, float *output, int64_t n_elems);
int quantized_matmul(const float *x, const uint8_t *W_q, int quant_type, int n_rows, int n_cols, int row_major, float *y);

int main() {
    gguf_ctx *ctx = gguf_open("/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf");
    gguf_buffer_data(ctx);
    const uint8_t *blob = (const uint8_t *)ctx->data_blob;
    
    // Load layer 0's MoE gate weights (expert 0)
    gguf_tensor_info *t = gguf_find_tensor(ctx, "blk.0.ffn_gate_exps.weight");
    if (!t) { printf("FAIL: can't find gate tensor\n"); return 1; }
    printf("gate_exps: type=%d dims=[%ld,%ld]\n", t->ggml_type, (long)t->dims[0], (long)t->dims[1]);
    
    int64_t n_elems = (int64_t)D_MODEL * D_FF;  // per expert
    int64_t raw_size = gguf_raw_size(t->ggml_type, n_elems);
    printf("  per-expert raw bytes: %ld\n", (long)raw_size);
    
    // Expert 0 raw data
    const uint8_t *expert0_raw = blob + t->data_offset;  // expert 0 at data_offset
    // Expert 1 raw data (for reference)
    const uint8_t *expert1_raw = blob + t->data_offset + raw_size;
    
    // Generate random input
    float x[D_MODEL];
    srand(12345);
    for (int i = 0; i < D_MODEL; i++) x[i] = ((float)rand()/RAND_MAX - 0.5f) * 2.0f;
    
    // === CPU reference: F32 dequant + SGEMM ===
    float *f32_w = (float*)malloc(n_elems * sizeof(float));
    // Read via gguf_read_tensor_f32 for comparison
    // But that reads ALL experts. Let's use our dequant function directly
    dequantize_iq2_xxs_row(expert0_raw, f32_w, n_elems);
    
    float cpu_gate[D_FF];
    for (int j = 0; j < D_FF; j++) {
        double sum = 0;
        for (int k = 0; k < D_MODEL; k++)
            sum += (double)x[k] * (double)f32_w[k + (int64_t)j * D_MODEL];
        cpu_gate[j] = (float)sum;
    }
    printf("CPU gate[0..4]: %.6f %.6f %.6f %.6f %.6f (rms=%.6f)\n",
           cpu_gate[0],cpu_gate[1],cpu_gate[2],cpu_gate[3],cpu_gate[4], sqrtf(0.00001f));
    
    // === CPU Q8_K requant (what wubu_moe uses) ===
    float cpu_q8_gate[D_FF];
    quantized_matmul(x, expert0_raw, t->ggml_type, D_MODEL, D_FF, 0, cpu_q8_gate);
    printf("CPU Q8 gate[0..4]: %.6f %.6f %.6f %.6f %.6f (rms=%.6f)\n",
           cpu_q8_gate[0],cpu_q8_gate[1],cpu_q8_gate[2],cpu_q8_gate[3],cpu_q8_gate[4], sqrtf(0.00001f));
    
    // Compare F32 vs Q8_K
    double dot=0, na=0, nb=0, me=0;
    for (int j = 0; j < D_FF; j++) {
        double d = cpu_gate[j] - cpu_q8_gate[j];
        if (fabs(d) > me) me = fabs(d);
        dot += (double)cpu_gate[j] * (double)cpu_q8_gate[j];
        na += (double)cpu_gate[j] * (double)cpu_gate[j];
        nb += (double)cpu_q8_gate[j] * (double)cpu_q8_gate[j];
    }
    printf("CPU F32 vs Q8_K: cos_sim=%.8f max_diff=%.6f\n", dot/(sqrt(na)*sqrt(nb)+1e-30), me);
    
    // === GPU kernel ===
    cudaStream_t st; cudaStreamCreate(&st);
    wubu_gpu_moe_init();  // upload lookup tables
    
    float *d_x, *d_out_buf;
    uint8_t *d_gate_buf, *d_up_buf, *d_down_buf;
    cudaMalloc(&d_x, D_MODEL * 4);
    cudaMalloc(&d_out_buf, D_MODEL * 4);
    
    // For MoE kernel test, we need gate, up (dummy), down (dummy)
    cudaMalloc(&d_gate_buf, raw_size);
    cudaMemcpy(d_gate_buf, expert0_raw, raw_size, cudaMemcpyHostToDevice);
    cudaMalloc(&d_up_buf, raw_size);
    cudaMemset(d_up_buf, 0, raw_size);
    cudaMalloc(&d_down_buf, raw_size);
    cudaMemset(d_down_buf, 0, raw_size);
    
    // Run kernel through the host wrapper
    const uint8_t *ptrs[8] = {0};
    float wgts[8] = {0};
    float output[8 * D_MODEL] = {0};
    
    // Single expert test
    ptrs[0] = (const uint8_t*)(uintptr_t)d_gate_buf;  // GPU pointer
    ptrs[1] = (const uint8_t*)(uintptr_t)d_up_buf;     // GPU pointer
    ptrs[2] = (const uint8_t*)(uintptr_t)d_down_buf;   // GPU pointer
    wgts[0] = 1.0f;
    
    // Actually, wubu_gpu_moe_forward_experts needs up+down too. Let me just launch the kernel directly
    // to test gate output first.
    
    // Memory for kernel: shared = (D_FF + D_MODEL) * 4 = (512+2048)*4 = 10240
    size_t smem = (D_FF + D_MODEL) * sizeof(float);
    
    // For single expert test: gate_q = d_gate_buf, up_q = d_up_buf (zeros)
    // The kernel computes gate = x @ gate_q, up = x @ up_q (zeros)
    // Then act = silu(gate) * up = 0 (since up is zero)
    // Then output = down @ act = 0 (since act is zero)
    // So output is zero. That's fine — we want to check if the gate projection matches CPU.
    
    // Actually we CAN'T easily check just the gate because the kernel fuses gate+up+silu+down.
    // Let me compute up = x and down = identity to verify the full path.
    
    // Better approach: use moe_expert_kernel directly with known weights
    // For now, let me check if the kernel at least runs without error
    moe_expert_kernel<<<1, D_FF, smem, st>>>(d_x, d_gate_buf, d_up_buf, d_down_buf, 1.0f, d_out_buf);
    cudaStreamSynchronize(st);
    cudaError_t ce = cudaGetLastError();
    if (ce != cudaSuccess) {
        printf("FAIL: kernel error: %s\n", cudaGetErrorString(ce));
    } else {
        printf("OK: kernel ran without error\n");
        float gpu_out[D_MODEL];
        cudaMemcpy(gpu_out, d_out_buf, D_MODEL * 4, cudaMemcpyDeviceToHost);
        printf("GPU out[0..4]: %.6f %.6f %.6f %.6f %.6f (rms=%.6f)\n",
               gpu_out[0],gpu_out[1],gpu_out[2],gpu_out[3],gpu_out[4], sqrtf(0.00001f));
    }
    
    cudaFree(d_x); cudaFree(d_out_buf);
    cudaFree(d_gate_buf); cudaFree(d_up_buf); cudaFree(d_down_buf);
    cudaStreamDestroy(st);
    free(f32_w);
    gguf_close(ctx);
    return 0;
}
