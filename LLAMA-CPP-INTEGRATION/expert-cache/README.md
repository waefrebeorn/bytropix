# PCIe MoE Expert Cache

GPU-side staging buffer for Mixture-of-Experts weight prefetching.

**Problem**: MoE models scatter-read expert weights from CPU memory via PCIe at ~1.22 GB/s, while PCIe 4.0 x16 supports 16 GB/s burst.

**Solution**: Synchronous `cudaMemcpy` burst copies 8 expert weights to a GPU staging buffer (~110 MB), then remaps expert IDs to staging indices. This reduces expert read time from ~28 ms (scatter) to ~3.4 ms (burst).

**Status**: Implementation in `llama-cpp-rotorquant` fork. See `LLAMA-CPP-INTEGRATION/README.md` for architecture.
