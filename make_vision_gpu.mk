# Add to Makefile rules section
src/cuda_vision.o: src/cuda_vision.cu include/cuda_vision.h include/wubu_vision.h
\t$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

infer_vision_gpu: tools/infer_vision_gpu.c $(CORE_OBJ) src/cuda_vision.o
\t$(CC) $(CFLAGS) $(CUDA_INC) -o $@ $^ $(LDFLAGS) $(CUDA_LIBS) -L/usr/local/cuda/lib64 -lstdc++
