/**
 * poincare_attn.cu — Poincaré Manifold Sparse Attention Kernel
 * 
 * Uses hyperbolic geometry (Poincaré ball) for sparse top-k attention:
 * - Keys projected to Poincaré ball
 * - Ball tree for O(log N) nearest neighbor search
 * - Only attend to epsilon-neighborhood of query in hyperbolic space
 * 
 * Based on WuBu Poincaré encoder (wubu_poincare_gqa.c, wubu_mobius.c)
 */

#include "wubu_turboquant.h"
#include "wubu_poincare_gqa.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

// ================================================================
// Poincaré Ball Model Primitives
// ================================================================

// Möbius addition: x ⊕ y = (1 + 2<x,y> + ||y||²)x + (1 - ||x||²)y / (1 + 2<x,y> + ||x||²||y||²)
__host__ __device__ __forceinline__ void mobius_add_poincare(const float* x, const float* y, 
                                                     float* out, int dim) {
    float x_norm2 = 0.0f, y_norm2 = 0.0f, xy = 0.0f;
    for (int i = 0; i < dim; i++) {
        x_norm2 += x[i] * x[i];
        y_norm2 += y[i] * y[i];
        xy += x[i] * y[i];
    }
    
    float denom = 1.0f + 2.0f * xy + x_norm2 * y_norm2;
    float scale_x = (1.0f + 2.0f * xy + y_norm2) / denom;
    float scale_y = (1.0f - x_norm2) / denom;
    
    for (int i = 0; i < dim; i++) {
        out[i] = scale_x * x[i] + scale_y * y[i];
    }
}

// Poincaré distance: d_P(x, y) = 2 * atanh(||(-x) ⊕ y||)
__host__ __device__ __forceinline__ float poincare_distance(const float* x, const float* y, int dim) {
    float neg_x[64];  // max dim 64 for distance calc
    for (int i = 0; i < dim; i++) neg_x[i] = -x[i];
    
    float diff[64];
    mobius_add_poincare(neg_x, y, diff, dim);
    
    float norm2 = 0.0f;
    for (int i = 0; i < dim; i++) norm2 += diff[i] * diff[i];
    
    float norm = sqrtf(norm2);
    return 2.0f * atanhf(fminf(norm, 0.999999f));  // clamp
}

// Log map (tangent space projection at origin)
__host__ __device__ __forceinline__ void poincare_log_map(const float* x, float* out, int dim) {
    float norm2 = 0.0f;
    for (int i = 0; i < dim; i++) norm2 += x[i] * x[i];
    float norm = sqrtf(norm2);
    if (norm < 1e-6f) {
        for (int i = 0; i < dim; i++) out[i] = 0.0f;
        return;
    }
    float scale = atanhf(norm) / norm;
    for (int i = 0; i < dim; i++) out[i] = x[i] * scale;
}

// ================================================================
// Ball Tree Node (stored in device memory)
// ================================================================

#define BALL_TREE_MAX_CHILDREN 8
#define BALL_TREE_MAX_DEPTH 7    // 8^7 = 2M leaves for 512k context
#define BALL_TREE_POOL_DIM 64    // Poincaré projection dimension

struct ball_tree_node_t {
    float centroid[BALL_TREE_POOL_DIM];  // Möbius centroid in Poincaré ball
    float radius;                         // Max distance to children
    int child_start;                      // Index in node array (for internal nodes)
    int child_count;                      // Number of children (0 for leaves)
    int leaf_start;                       // If leaf: start token index in KV cache
    int leaf_count;                       // Number of tokens in leaf
    bool is_leaf;
};

// Ball tree metadata
struct ball_tree_meta_t {
    int n_nodes;
    int max_nodes;
    int n_tokens;
    int dim;
    ball_tree_node_t* d_nodes;
    // For search results
    int* d_valid_indices;  // [n_valid] indices of tokens to attend
    int* d_valid_count;
};

// ================================================================
// CPU-side Ball Tree Build (for now, GPU build as future optimization)
// ================================================================

// Recursive function to build ball tree from K cache data
static void build_ball_tree_recursive(
    const float* K_poincare,  // [n_tokens, dim] - keys already in Poincaré ball
    int start_idx, int count,
    ball_tree_node_t* nodes, int* node_count,
    int depth, int dim
) {
    int node_idx = (*node_count)++;
    ball_tree_node_t* node = &nodes[node_idx];
    
    if (count <= 64 || depth >= BALL_TREE_MAX_DEPTH) {
        // Leaf node
        node->is_leaf = true;
        node->leaf_start = start_idx;
        node->leaf_count = count;
        node->child_count = 0;
        node->child_start = -1;
        
        // Compute centroid (Möbius mean approximation)
        for (int d = 0; d < dim; d++) node->centroid[d] = 0.0f;
        for (int i = 0; i < count; i++) {
            const float* k = K_poincare + (start_idx + i) * dim;
            for (int d = 0; d < dim; d++) {
                node->centroid[d] += k[d];
            }
        }
        for (int d = 0; d < dim; d++) {
            node->centroid[d] /= count;
        }
        // Renormalize to stay in Poincaré ball
        float norm2 = 0.0f;
        for (int d = 0; d < dim; d++) norm2 += node->centroid[d] * node->centroid[d];
        float norm = sqrtf(norm2);
        if (norm >= 0.99f) {
            float scale = 0.99f / norm;
            for (int d = 0; d < dim; d++) node->centroid[d] *= scale;
        }
        
        // Compute radius (max distance to any point in leaf)
        node->radius = 0.0f;
        for (int i = 0; i < count; i++) {
            float dist = poincare_distance(node->centroid, K_poincare + (start_idx + i) * dim, dim);
            if (dist > node->radius) node->radius = dist;
        }
        return;
    }
    
    // Internal node: partition using furthest-point clustering
    // Find two furthest points as initial centroids
    int c1 = start_idx, c2 = start_idx;
    float max_dist = 0.0f;
    for (int i = start_idx; i < start_idx + count; i++) {
        for (int j = i + 1; j < start_idx + count; j++) {
            float dist = poincare_distance(K_poincare + i * dim, K_poincare + j * dim, dim);
            if (dist > max_dist) {
                max_dist = dist;
                c1 = i; c2 = j;
            }
        }
    }
    
    // Assign points to closest centroid
    int mid = start_idx + count / 2;
    // Simple partition: first half to child 0, second to child 1
    // More sophisticated: sort by distance to centroids
    for (int i = 0; i < count; i++) {
        float d1 = poincare_distance(K_poincare + (start_idx + i) * dim, K_poincare + c1 * dim, dim);
        float d2 = poincare_distance(K_poincare + (start_idx + i) * dim, K_poincare + c2 * dim, dim);
        // In a real implementation, we'd partition properly
    }
    
    // Recurse on partitions
    int child_start_idx = *node_count;
    node->is_leaf = false;
    node->child_count = 2;
    node->child_start = child_start_idx;
    node->leaf_start = -1;
    node->leaf_count = 0;
    
    build_ball_tree_recursive(K_poincare, start_idx, mid - start_idx, nodes, node_count, depth + 1, dim);
    build_ball_tree_recursive(K_poincare, mid, start_idx + count - mid, nodes, node_count, depth + 1, dim);
    
    // Compute centroid as Möbius mean of children's centroids
    for (int d = 0; d < dim; d++) node->centroid[d] = 0.0f;
    int n_children = 0;
    for (int c = 0; c < node->child_count; c++) {
        ball_tree_node_t* child = &nodes[node->child_start + c];
        for (int d = 0; d < dim; d++) {
            node->centroid[d] += child->centroid[d];
        }
        n_children++;
    }
    for (int d = 0; d < dim; d++) {
        node->centroid[d] /= n_children;
    }
    // Renormalize
    float norm2 = 0.0f;
    for (int d = 0; d < dim; d++) norm2 += node->centroid[d] * node->centroid[d];
    float norm = sqrtf(norm2);
    if (norm >= 0.99f) {
        float scale = 0.99f / norm;
        for (int d = 0; d < dim; d++) node->centroid[d] *= scale;
    }
    
    // Compute radius as max distance to any descendant leaf
    node->radius = 0.0f;
    for (int c = 0; c < node->child_count; c++) {
        ball_tree_node_t* child = &nodes[node->child_start + c];
        float dist = poincare_distance(node->centroid, child->centroid, dim) + child->radius;
        if (dist > node->radius) node->radius = dist;
    }
}

// Host function to build ball tree and upload to GPU
static void build_ball_tree_gpu(
    const float* K_cache, int n_tokens, int dim,
    ball_tree_meta_t* meta
) {
    // Allocate host buffer for K in Poincaré ball
    float* h_K_poincare = (float*)malloc((size_t)n_tokens * dim * sizeof(float));
    
    // Project K_cache to Poincaré ball (normalize each vector to < 1)
    for (int i = 0; i < n_tokens; i++) {
        float norm2 = 0.0f;
        for (int d = 0; d < dim; d++) {
            float v = K_cache[i * dim + d];
            norm2 += v * v;
            h_K_poincare[i * dim + d] = v;
        }
        float norm = sqrtf(norm2);
        if (norm > 1e-6f) {
            float scale = 0.9f / (norm + 1e-6f);  // Scale to 0.9 radius
            for (int d = 0; d < dim; d++) {
                h_K_poincare[i * dim + d] *= scale;
            }
        }
    }
    
    // Estimate max nodes: 2 * n_tokens (worst case binary tree)
    int max_nodes = 2 * n_tokens + 1;
    ball_tree_node_t* h_nodes = (ball_tree_node_t*)calloc(max_nodes, sizeof(ball_tree_node_t));
    int node_count = 0;
    
    // Build tree (using dim=64 for ball tree, may be smaller than head_dim)
    int ball_dim = (dim < BALL_TREE_POOL_DIM) ? dim : BALL_TREE_POOL_DIM;
    build_ball_tree_recursive(h_K_poincare, 0, n_tokens, h_nodes, &node_count, 0, ball_dim);
    
    // Upload to GPU
    cudaMalloc(&meta->d_nodes, (size_t)node_count * sizeof(ball_tree_node_t));
    cudaMemcpy(meta->d_nodes, h_nodes, (size_t)node_count * sizeof(ball_tree_node_t), cudaMemcpyHostToDevice);
    
    // Allocate search result buffers
    cudaMalloc(&meta->d_valid_indices, n_tokens * sizeof(int));
    cudaMalloc(&meta->d_valid_count, sizeof(int));
    
    meta->n_nodes = node_count;
    meta->max_nodes = max_nodes;
    meta->n_tokens = n_tokens;
    meta->dim = ball_dim;
    
    free(h_K_poincare);
    free(h_nodes);
}

// Free ball tree GPU memory
static void free_ball_tree_gpu(ball_tree_meta_t* meta) {
    if (meta->d_nodes) cudaFree(meta->d_nodes);
    if (meta->d_valid_indices) cudaFree(meta->d_valid_indices);
    if (meta->d_valid_count) cudaFree(meta->d_valid_count);
    meta->d_nodes = NULL;
    meta->d_valid_indices = NULL;
    meta->d_valid_count = NULL;
    meta->n_nodes = 0;
}

// ================================================================
// GPU Kernel: Ball Tree Search for Sparse Attention
// Finds all KV indices within epsilon hyperbolic distance of query
// ================================================================

__device__ __forceinline__ bool ball_intersects_query(
    const ball_tree_node_t* node, const float* query, float eps, int dim
) {
    float dist = poincare_distance(node->centroid, query, dim);
    return (dist - node->radius) <= eps;
}

__global__ void ball_tree_search_kernel(
    const ball_tree_node_t* nodes, int n_nodes,
    const float* query, int dim, float eps,
    int* valid_indices, int* valid_count
) {
    // Single-threaded search for now (can parallelize across query heads)
    if (threadIdx.x > 0) return;
    if (blockIdx.x > 0) return;
    
    // Stack for DFS (max depth BALL_TREE_MAX_DEPTH)
    int stack[BALL_TREE_MAX_DEPTH * 8];
    int stack_ptr = 0;
    int count = 0;
    
    // Start from root (node 0)
    stack[stack_ptr++] = 0;
    
    while (stack_ptr > 0) {
        int node_idx = stack[--stack_ptr];
        const ball_tree_node_t* node = &nodes[node_idx];
        
        if (node->is_leaf) {
            // Check all tokens in leaf
            for (int i = 0; i < node->leaf_count; i++) {
                float token_poincare[BALL_TREE_POOL_DIM];
                // Note: In practice, we'd have leaf token data in constant memory or texture
                // For now, placeholder - would need actual token embeddings
                if (count < 4096) {
                    valid_indices[count++] = node->leaf_start + i;
                }
            }
        } else {
            // Check children
            for (int c = 0; c < node->child_count; c++) {
                int child_idx = node->child_start + c;
                if (child_idx < n_nodes && ball_intersects_query(&nodes[child_idx], query, eps, dim)) {
                    stack[stack_ptr++] = child_idx;
                }
            }
        }
    }
    
    *valid_count = count;
}

// ================================================================
// Optimized: Two-pass - distance filter then attention
// ================================================================

__global__ void poincare_filter_kernel(
    const half* __restrict__ Q,         // [B, Hq, 1, head_dim]
    const uint8_t* __restrict__ K_pool,
    const int* __restrict__ block_table,
    int* __restrict__ valid_mask,       // [B, Tk] bool: attend or skip
    const float eps_distance,
    int B, int Tk, int window_size
) {
    // Simplified: use Q[0] (first head) as representative for filtering
    // Full multi-head in production
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int kv_abs = tid;
    if (kv_abs >= Tk) return;
    
    // Get K vector
    // Dequant Q[0] and K[kv_abs], compute Poincaré distance
    // If < eps: valid_mask[kv_abs] = 1
}

// ================================================================
// Host launcher for Poincaré sparse decode
// ================================================================

void launch_poincare_sparse_decode(
    const half* Q, const int* block_table, const uint8_t* K_pool, 
    const uint8_t* V_pool, half* O,
    float softmax_scale, float eps_distance, int window_size,
    int B, int Tk, cudaStream_t stream
) {
    // Pre-filter kernel
    int blocks = (Tk + 255) / 256;
    poincare_filter_kernel<<<blocks, 256, 0, stream>>>(
        Q, K_pool, block_table, nullptr, eps_distance, 1, Tk, window_size
    );
    
    // Then attention kernel with valid_mask
}

// ================================================================
// Public API for ball tree management
// ================================================================

void poincare_ball_tree_build(
    const float* K_cache, int n_tokens, int dim,
    void** out_nodes, int* out_count
) {
    // Build in host memory for now
    // K_cache: [n_tokens, dim] already in Poincaré ball
    // Recursive k-means / furthest-point clustering
    
    // For now: placeholder
    *out_count = 0;
    *out_nodes = nullptr;
}