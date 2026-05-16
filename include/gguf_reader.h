#ifndef GGUF_READER_H
#define GGUF_READER_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

// GGML type enum (relevant subset)
enum ggml_type {
    GGML_TYPE_F32    = 0,
    GGML_TYPE_F16    = 1,
    GGML_TYPE_Q4_0   = 2,
    GGML_TYPE_Q4_1   = 3,
    GGML_TYPE_Q5_0   = 6,
    GGML_TYPE_Q5_1   = 7,
    GGML_TYPE_Q8_0   = 8,
    GGML_TYPE_Q8_1   = 9,
    GGML_TYPE_Q2_K   = 10,
    GGML_TYPE_Q3_K   = 11,
    GGML_TYPE_Q4_K   = 12,
    GGML_TYPE_Q5_K   = 13,
    GGML_TYPE_Q6_K   = 14,
    GGML_TYPE_Q8_K   = 15,
    GGML_TYPE_IQ2_XXS = 16,
    GGML_TYPE_IQ2_XS  = 17,
    GGML_TYPE_IQ3_XXS = 18,
    GGML_TYPE_IQ1_S   = 19,
    GGML_TYPE_IQ3_S   = 21,
    GGML_TYPE_IQ2_S   = 22,
    GGML_TYPE_IQ4_XS  = 23,
    GGML_TYPE_IQ1_M   = 29,
};

// GGUF tensor info
typedef struct {
    char name[256];
    int n_dims;
    int64_t dims[4];
    int ggml_type;        // enum ggml_type
    uint64_t data_offset; // offset from data blob start
} gguf_tensor_info;

// GGUF context
typedef struct {
    // File info
    uint32_t version;
    int64_t n_tensors;
    int64_t n_kv;
    uint32_t alignment;
    
    // Tensor info array
    gguf_tensor_info *tensors;
    
    // Data blob location in file
    uint64_t data_blob_offset;
    
    // File handle
    FILE *file;
    
    // Optional: buffered data blob (mmap or malloc'd copy)
    void *data_blob;
    size_t data_blob_size;
} gguf_ctx;

// Open GGUF file and parse headers
gguf_ctx* gguf_open(const char *path);

// Buffer entire data blob in RAM for fast tensor reads
// Call after gguf_open, before any gguf_read_tensor_f32 calls
int gguf_buffer_data(gguf_ctx *ctx);

// Calculate raw (quantized) byte size for a tensor type/element count
int64_t gguf_raw_size(int ggml_type, int64_t n_elems);

// Dequantize raw quantized bytes to f32
void gguf_dequantize(const uint8_t *data, int ggml_type, int64_t n_elems, float *output);

// Find a tensor by name
gguf_tensor_info* gguf_find_tensor(gguf_ctx *ctx, const char *name);

// Read tensor data (dequantized to float32)
// Returns number of floats written, or 0 on error
int gguf_read_tensor_f32(gguf_ctx *ctx, gguf_tensor_info *tensor, float *output, int64_t max_elems);

// Close GGUF file and free context
void gguf_close(gguf_ctx *ctx);

// IQ1_S / Q6_K dequantization (called internally by gguf_read_tensor_f32)
void dequantize_q6_K_row(const uint8_t *data, float *output, int64_t n_elems);
void dequantize_iq1_s_row(const uint8_t *data, float *output, int64_t n_elems);
void dequantize_iq2_xxs_row(const uint8_t *data, float *output, int64_t n_elems);
void dequantize_iq2_s_row(const uint8_t *data, float *output, int64_t n_elems);

// IQ2_XXS block-level operations (for on-the-fly dequant dot product)
void dequantize_iq2_xxs_block(const uint8_t *block, float *output);
float iq2_xxs_dot_block(const uint8_t *block, const float *x);
float iq2_xxs_dot_row(const uint8_t *qdata, int n_elems, const float *x);

// Embedding ops
void wubu_exp_map(const float *input, int dim, float R, float *output);
void wubu_log_map(const float *input, int dim, float R, float *output);
float wubu_norm(const float *v, int dim);
float wubu_dot(const float *a, const float *b, int dim);

#ifdef __cplusplus
}
#endif

#endif // GGUF_READER_H
