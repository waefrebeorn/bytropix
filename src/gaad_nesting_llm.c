// gaad_nesting_llm.c
#include "gaad_nesting_llm.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

static int64_t golden_split_pos(int64_t length) {
    return (int64_t)(length * PHI_INV + 0.5);
}

// Static helper for recursive free
static void gaad_free_node(Node* n) {
    if (!n) return;
    for (int i = 0; i < n->num_children; i++) gaad_free_node(n->children[i]);
    free(n->data);
    free(n);
}

Context* create_context(int64_t seq_len, int max_segments) {
    Context* ctx = calloc(1, sizeof(Context));
    ctx->seq_len = seq_len;
    ctx->initial_segments = malloc(max_segments * sizeof(Segment));
    return ctx;
}

void destroy_context(Context* ctx) {
    if (!ctx) return;
    free(ctx->initial_segments);
    if (ctx->tree_root) {
        gaad_free_node(ctx->tree_root);
    }
    free(ctx->mask);
    free(ctx);
}

void decompose_segments(Context* ctx, const float* per_token_score) {
    // Content-adaptive initial segmentation.
    // Example: place splits where score changes significantly.
    // Scaffold uses a simple uniform start + adjustment.
    int count = 0;
    int64_t pos = 0;
    int target = 8; // start coarse
    while (pos < ctx->seq_len && count < 64) {
        int64_t len = (ctx->seq_len - pos) / (target - count);
        if (len < 8) len = ctx->seq_len - pos;

        Segment s = {
            .start = pos,
            .end = pos + len,
            .level = 0,
            .id = count
        };

        // Adapt length using local score average or variance
        if (per_token_score) {
            // simple example: extend if high average score in window
            float avg = 0.0f;
            for (int64_t i = pos; i < pos + len && i < ctx->seq_len; ++i) avg += per_token_score[i];
            avg /= len;
            if (avg > 0.6f) len = (int64_t)(len * 1.2);
        }

        s.end = pos + len;
        if (s.end > ctx->seq_len) s.end = ctx->seq_len;

        ctx->initial_segments[count++] = s;
        pos = s.end;
    }
    ctx->num_initial = count;
}

static void add_child(Node* parent, int idx, int64_t split_pos) {
    parent->children[idx] = calloc(1, sizeof(Node));
    Node* c = parent->children[idx];
    c->seg.level = parent->seg.level + 1;
    c->seg.id = (parent->seg.id << 4) | idx;

    if (idx == 0) {
        c->seg.start = parent->seg.start;
        c->seg.end = split_pos;
    } else {
        c->seg.start = split_pos;
        c->seg.end = parent->seg.end;
    }
}

static void recurse_split(Node* node, int depth, int max_depth) {
    int64_t len = node->seg.end - node->seg.start;
    if (depth >= max_depth || len < 8) return;

    int64_t split = node->seg.start + golden_split_pos(len);
    if (split <= node->seg.start || split >= node->seg.end) return;

    node->num_children = 2;
    add_child(node, 0, split);
    add_child(node, 1, split);
    recurse_split(node->children[0], depth + 1, max_depth);
    recurse_split(node->children[1], depth + 1, max_depth);
}

void build_tree(Context* ctx, int max_depth) {
    if (ctx->num_initial == 0) return;
    ctx->tree_root = calloc(1, sizeof(Node));
    ctx->tree_root->seg = ctx->initial_segments[0];
    recurse_split(ctx->tree_root, 0, max_depth);
}

void build_mask(Context* ctx, float threshold) {
    // Collect leaf segments
    // Scaffold: treat leaf segments as blocks
    ctx->mask_n = 256; // example number of leaves
    ctx->mask = calloc(ctx->mask_n * ctx->mask_n, sizeof(uint8_t));

    for (size_t i = 0; i < ctx->mask_n; ++i) {
        for (size_t j = 0; j < ctx->mask_n; ++j) {
            // Attend if close in leaf order or share coarse ancestor
            if (llabs((long)i - (long)j) < 8 || ((i >> 3) == (j >> 3))) {
                ctx->mask[i * ctx->mask_n + j] = 1;
            }
        }
    }
    // Real version: walk tree to find LCA depth; attend if LCA depth >= threshold
}