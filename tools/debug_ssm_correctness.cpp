/**
 * Minimal: dequant one Q5_K block on GPU, compare with CPU dequant.
 */
#include "wubu_model.h"
#include "wubu_ssm.h"
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include "gpu_quant_matmul.h"

// GPU kernel: dequant one block and store 256 floats
__global__ void deq_test_kernel(const uint8_t *block, float *out) {
    int idx = threadIdx.x;
    if (idx >= 64) return;  // only need 64 threads for the loop
    
    // Replicate the kernel's dequant logic
    float d = *(const float*)&block; // avoid fp16_to_fp32 for now
    // Actually let me just read raw bytes
    
    __shared__ float vals[64];
    // Just read first 64 bytes of the block as floats
    vals[idx] = ((float*)block)[idx];
    __syncthreads();
    if (idx < 10) out[idx] = vals[idx];
}

int main() {
    wubu_model_t mdl;
    memset(&mdl,0,sizeof(mdl));
    if(!wubu_model_init(&mdl,"/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf"))return 1;
    
    const uint8_t *w = mdl.layers[0].ssm.attn_qkv_weight_q;
    
    // Print hex of first 8 bytes of the Q5_K block
    printf("Raw bytes of first Q5_K block:\n");
    for(int i=0;i<16;i++) printf("%02x ", w[i]);
    printf("\n");
    printf("As uint16: ");
    for(int i=0;i<8;i++) printf("%04x ", ((uint16_t*)w)[i]);
    printf("\n");
    printf("As uint32: ");
    for(int i=0;i<4;i++) printf("%08x ", ((uint32_t*)w)[i]);
    printf("\n");
    
    // What does fp16_to_fp32 of the first 2 bytes produce?
    uint16_t d_bits = *(const uint16_t*)w;
    float d_host;
    { uint32_t s=(d_bits>>15)&1,e=(d_bits>>10)&0x1F,m=d_bits&0x03FF;
      uint32_t f=e==0?(s<<31)|((127-15+1)<<23)|(m<<13):
               e==31?(s<<31)|(0xFF<<23)|(m<<13):
               (s<<31)|((127-15+e)<<23)|(m<<13);
      memcpy(&d_host,&f,4); }
    
    printf("d_bits=0x%04x sign=%d exp=%d mant=%d -> %e\n", d_bits, (d_bits>>15)&1, (d_bits>>10)&0x1F, d_bits&0x03FF, d_host);
    
    // GPU: upload first block, run deq_test_kernel  
    uint8_t *d_block;
    float *d_out;
    cudaMalloc(&d_block, 176);
    cudaMalloc(&d_out, 256*4);
    cudaMemcpy(d_block, w, 176, cudaMemcpyHostToDevice);
    
    deq_test_kernel<<<1, 256>>>(d_block, d_out);
    
    float h_out[10];
    cudaMemcpy(h_out, d_out, 10*4, cudaMemcpyDeviceToHost);
    printf("GPU raw float read of first 10 elements: ");
    for(int i=0;i<10;i++) printf("%.8f ", h_out[i]);
    printf("\n");
    
    cudaFree(d_block); cudaFree(d_out);
    wubu_model_free(&mdl);
    return 0;
}
