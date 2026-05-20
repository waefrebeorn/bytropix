#include "gguf_reader.h"
#include "gpu_moe_kernel.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#define D_MODEL 2048
#define D_FF 512
#define QK_K 256
#define BLK_SZ 66

int main() {
    gguf_ctx* ctx = gguf_open("/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf");
    gguf_buffer_data(ctx);
    gguf_tensor_info* t = gguf_find_tensor(ctx, "blk.0.ffn_gate_exps.weight");
    
    const uint8_t* blob = (const uint8_t*)ctx->data_blob + t->data_offset;
    int64_t gate_bytes = gguf_raw_size(t->ggml_type, (int64_t)D_MODEL * D_FF);
    
    // Read first block of expert 0, column 0
    // Column 0 starts at offset 0, block 0 at offset 0
    const uint8_t* block = blob;
    
    // Print raw block bytes
    printf("First 16 bytes of block 0: ");
    for (int i = 0; i < 16; i++) printf("%02x ", block[i]);
    printf("\n");
    
    // Check d (fp16 at bytes 0-1)
    uint16_t d_bits;
    memcpy(&d_bits, block, 2);
    int sign = (d_bits >> 15) & 1, exp = (d_bits >> 10) & 0x1F, mant = d_bits & 0x3FF;
    float d = (exp==0 ? 0.0f : ldexpf(1.0f+(float)mant/1024.0f, exp-15) * (sign?-1:1));
    printf("d bits=%04x exp=%d mant=%d d=%.6f\n", d_bits, exp, mant, d);
    
    // Row 0: already loaded x with zero values? Or does the kernel work?
    // Let me test: upload, run, check error
    float x[D_MODEL];
    for (int i = 0; i < D_MODEL; i++) x[i] = 1.0f; // all-ones
    
    wubu_gpu_moe_init();
    cudaStream_t st; cudaStreamCreate(&st);
    
    const uint8_t* gate_ptrs[8] = {blob, NULL, NULL, NULL, NULL, NULL, NULL, NULL};
    const uint8_t* up_ptrs[8]   = {blob, NULL, NULL, NULL, NULL, NULL, NULL, NULL};
    const uint8_t* down_ptrs[8] = {blob, NULL, NULL, NULL, NULL, NULL, NULL, NULL};
    float wgts[8] = {1.0f, 0, 0, 0, 0, 0, 0, 0};
    float output[8 * D_MODEL];
    
    printf("\nRunning with x=1.0...\n");
    wubu_gpu_moe_forward_experts(x, gate_ptrs, gate_bytes, up_ptrs, gate_bytes,
                                 down_ptrs, gate_bytes,
                                 t->ggml_type, t->ggml_type, t->ggml_type,
                                 wgts, output, st);
    
    printf("First 10 output: ");
    for (int i = 0; i < 10; i++) printf("%.6f ", output[i]);
    printf("\n");
    
    // Check CUDA errors
    cudaError_t err = cudaGetLastError();
    printf("CUDA error: %s\n", cudaGetErrorString(err));
    
    cudaStreamDestroy(st);
    gguf_close(ctx);
    return 0;
}
