// gaad_nesting_llm.h
#ifndef GAAD_NESTING_LLM_H
#define GAAD_NESTING_LLM_H

#include <stdint.h>
#include <stddef.h>

#define PHI 1.618033988749895
#define PHI_INV (1.0 / PHI)

typedef struct {
    int64_t start;   // token index (inclusive)
    int64_t end;     // token index (exclusive)
    int level;
    uint32_t id;
} Segment;

typedef struct Node {
    Segment seg;
    struct Node* children[2];  // binary split for 1D sequence
    int num_children;
    float* data;        // aggregated features for the segment (optional)
    size_t data_size;
} Node;

typedef struct {
    int64_t seq_len;
    Segment* initial_segments;
    int num_initial;
    Node* tree_root;
    uint8_t* mask;      // sparse attend mask (block or token level)
    size_t mask_n;      // number of leaf segments
} Context;

Context* create_context(int64_t seq_len, int max_segments);
void destroy_context(Context* ctx);

void decompose_segments(Context* ctx, const float* per_token_score);
void build_tree(Context* ctx, int max_depth);
void build_mask(Context* ctx, float threshold);

#endif