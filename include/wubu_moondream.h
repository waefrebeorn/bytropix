/*
 * wubu_moondream.h -- MoonDream 3 vision model core (agentic vision). C11.
 * Agnostic: MoonDream 3 = a 9B MoE with a 2B vision-language model.
 * It has its own core with detect, caption, and tool-calling.
 * This module provides the C11 bridge: image→tensors→MoonDream
 * inference→structured output (detections, captions, tool calls).
 */
#ifndef WUBU_MOONDAREAM_H
#define WUBU_MOONDAREAM_H

#include <stdint.h>
#include <stddef.h>

/* MoonDream-3 input image (normalized to 0..1 floats, NHWC). */
typedef struct {
    float *pixels;   /* width * height * channels */
    int width;
    int height;
    int channels;
} wubu_image_t;

/* MoonDream-3 detect result (bounding box). */
typedef struct {
    float x_min, y_min, x_max, y_max;
    float confidence;
    const char *label;
} wubu_md3_detect_t;

/* MoonDream-3 detect output. */
typedef struct {
    wubu_md3_detect_t *objects;
    int n_objects;
    int max_objects;
} wubu_md3_result_t;

/* MoonDream-3 tool-call (the agentic vision core). */
typedef struct {
    const char *name;      /* the tool to call */
    const char *arguments; /* JSON-encoded arguments */
} wubu_md3_tool_t;

/* MoonDream-3 full output. */
typedef struct {
    char *caption;         /* the image caption */
    wubu_md3_result_t detections;
    wubu_md3_tool_t *tools;
    int n_tools;
} wubu_md3_output_t;

/* MD01: image preprocessing (pixel normalization + patch tokenization). */
int wubu_md3_preprocess(const uint8_t *raw, int w, int h, int c, wubu_image_t *out);

/* MD02: visual encoder forward (the 2B vision backbone). */
int wubu_md3_encode(const wubu_image_t *img, float *tokens, int max_tokens);

/* MD03: MoE forward (the 9B text core with shared+expert FFNs). */
int wubu_md3_moe_forward(const float *tokens, int n_tokens, const int *expert_ids,
                         int n_experts, float *out, int d_model);

/* MD04: object detection head. */
int wubu_md3_detect(const float *encoded, int n_tokens, const char *query,
                    wubu_md3_result_t *result, int max_objects);

/* MD05: caption generation. */
int wubu_md3_caption(const float *encoded, int n_tokens, char *caption, int cap);

/* MD06: tool-call extraction (agentic vision). */
int wubu_md3_toolcall(const float *encoded, int n_tokens, const char *query,
                      wubu_md3_tool_t *tools, int max_tools);

/* MD07: end-to-end MoonDream inference (preprocess → encode → decode). */
int wubu_md3_infer(const uint8_t *raw, int w, int h, int c, const char *prompt,
                   wubu_md3_output_t *out);

/* MD08: FlexAttention-style decoding (the token-by-token generation). */
int wubu_md3_step(const float *ctx, int ctx_len, float *next_logits, int vocab_size);

/* MD09: bounding box normalization (0..1 → pixel coords). */
int wubu_md3_box_normalize(const wubu_md3_detect_t *det, int img_w, int img_h,
                           int *px_min, int *py_min, int *px_max, int *py_max);

/* MD10: confidence calibration. */
float wubu_md3_calibrate(float raw_conf, float temp);

#endif