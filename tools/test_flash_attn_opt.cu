/**
 * Quick test for optimized flash attention decode kernel
 * Tests kernel launch with dummy data - no model loading
 */
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>

// Include the kernel implementation directly
#include "/home/wubu/bytropix/src/flash_attn_q4_0_opt.cu"

#define HEAD_DIM 256
#define N_Q_HEADS 16
#define N_KV_HEADS 2
#define Q_HEADS_PER_KV 8
#define BLOCK_THREADS 128
#define KV_TILE_SIZE 128

void test_flash_attn_kernel() {
    const int B = 1;
    const int Tq = 1;
    const int Tk = 1024;  // Small KV cache
    const int head_dim = HEAD_DIM;

    // Allocate device memory
    half *d_Q, *d_O;
    int *d_block_table;
    uint8_t *d_K_pool, *d_V_pool;

    cudaMalloc(&d_Q, B * N_Q_HEADS * head_dim * sizeof(half));
    cudaMalloc(&d_O, B * N_Q_HEADS * head_dim * sizeof(half));
    cudaMalloc(&d_block_table, N_KV_HEADS * ((Tk + 15) / 16) * sizeof(int));
    cudaMalloc(&d_K_pool, Tk * N_KV_HEADS * head_dim / 32 * 18);  // Q4_0
    cudaMalloc(&d_V_pool, Tk * N_KV_HEADS * head_dim / 32 * 10);  // Q2_0

    // Fill with dummy data
    cudaMemset(d_Q, 0, B * N_Q_HEADS * head_dim * sizeof(half));
    cudaMemset(d_K_pool, 0, Tk * N_KV_HEADS * head_dim / 32 * 18);
    cudaMemset(d_V_pool, 0, Tk * N_KV_HEADS * head_dim / 32 * 10);

    // Set Q to known pattern (all 1.0)
    half *h_Q = (half*)malloc(B * N_Q_HEADS * head_dim * sizeof(half));
    for (int i = 0; i < B * N_Q_HEADS * head_dim; i++) h_Q[i] = __float2half(1.0f);
    cudaMemcpy(d_Q, h_Q, B * N_Q_HEADS * head_dim * sizeof(half), cudaMemcpyHostToDevice);
    free(h_Q);

    // Launch kernel
    dim3 block(BLOCK_THREADS);
    dim3 grid(4, B, N_KV_HEADS);  // qh_group=4, batch=1, kv_heads=2

    // Shared memory calculation
    size_t q_smem = 4 * 2 * 256 * sizeof(half);
    size_t q_q8 = 4 * 2 * 64 * sizeof(int);
    size_t q_ds = 4 * 2 * 32 * sizeof(float2);
    size_t kv_stage_k = (head_dim / 32) * 18;
    size_t kv_stage_v = (head_dim / 32) * 10;
    size_t kq_max_sum = 4 * 2 * 2 * sizeof(float);
    size_t smem_bytes = q_smem + q_q8 + q_ds + kv_stage_k + kv_stage_v + kq_max_sum;

    printf("Launching kernel with grid=(%d,%d,%d), block=%d, smem=%zu KB\n",
           grid.x, grid.y, grid.z, block.x, smem_bytes / 1024);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    flash_attn_q4_0_decode_opt_kernel<HEAD_DIM, N_Q_HEADS, N_KV_HEADS, Q_HEADS_PER_KV, BLOCK_THREADS, KV_TILE_SIZE>
        <<<grid, block, smem_bytes, 0>>>(
        d_Q, d_block_table, d_K_pool, d_V_pool, d_O,
        1.0f / sqrtf(256.0f), true, 0,  // softmax_scale, causal_mask, window_size
        B, Tq, Tk, 1
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
    } else {
        printf("Kernel launched successfully\n");
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Kernel execution time: %.3f ms\n", ms);

    // Copy back and check
    half *h_O = (half*)malloc(B * N_Q_HEADS * head_dim * sizeof(half));
    cudaMemcpy(h_O, d_O, B * N_Q_HEADS * head_dim * sizeof(half), cudaMemcpyDeviceToHost);

    // Print first few output values
    for (int i = 0; i < 32; i++) {
        printf("O[%d] = %f\n", i, __half2float(h_O[i]));
    }

    free(h_O);
    cudaFree(d_Q); cudaFree(d_O); cudaFree(d_block_table);
    cudaFree(d_K_pool); cudaFree(d_V_pool);
    cudaEventDestroy(start); cudaEventDestroy(stop);
}

int main() {
    printf("Testing optimized flash attention decode kernel...\n");
    test_flash_attn_kernel();
    printf("Test complete\n");
    return 0;
}
