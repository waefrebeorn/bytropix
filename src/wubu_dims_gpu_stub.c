/* CPU-only stub for the GPU dims sync symbol so CPU builds/tests link
 * without the CUDA toolkit. The real implementation lives in
 * wubu_dims_gpu.cu (GPU builds). On a machine without a working CUDA
 * toolchain this no-op keeps the engine's dims layer linkable. */
#include "wubu_dims.h"
void wubu_dims_sync_gpu(void) { }
