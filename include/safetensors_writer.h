/*
 * safetensors_writer.h -- write safetensors files (the missing half).
 *
 * We could READ safetensors (the reader) but never WRITE them, so every
 * checkpoint our trainer produced was a private .st dump no standard
 * tooling could open. This writer emits the real format: a JSON header
 * (tensor name -> dtype/shape/data_offsets) + the raw F32 blob, so the
 * bigger brother (Qwen), HF tooling, and any future framework can read
 * our trained seed directly.
 */
#ifndef SAFETENSORS_WRITER_H
#define SAFETENSORS_WRITER_H

#include <stdint.h>
#include <stddef.h>

/* A tensor to write. */
typedef struct {
    const char *name;
    const float *data;      /* F32 values */
    int64_t n_elems;
    /* shape: the first `n_dims` entries are the dims, row-major */
    int64_t dims[4];
    int n_dims;
} st_writer_tensor_t;

/* Write all tensors to `path`. Returns 0 on success. */
int st_write_f32(const char *path, const st_writer_tensor_t *tensors,
                 int n_tensors);

#endif
