/*
 * wubu_dims_gpu.cu -- mirror the host WUBU_DIMS global into the CUDA
 * __constant__ symbol WUBU_DIMS_DEV so device kernels (which cannot read a
 * host global) see the active model dimensions.
 *
 * C11/CUDA, self-contained: only depends on wubu_dims.h.
 */
#include "wubu_dims.h"

// The device-side constant the kernels read (via the WUBU_DIMS macro under
// __CUDACC__). Defined exactly once, here. Matches the extern decl in the header.
extern __constant__ wubu_dims_t WUBU_DIMS_DEV;

// Host-callable: push the current host WUBU_DIMS into device constant memory.
// Safe to call any time the dims change (loader sets them before GPU work).
void wubu_dims_sync_gpu(void) {
    cudaError_t e = cudaMemcpyToSymbol(WUBU_DIMS_DEV, &WUBU_DIMS,
                                       sizeof(wubu_dims_t), 0,
                                       cudaMemcpyHostToDevice);
    (void)e; // best-effort; GPU init checks return separately if needed
}
