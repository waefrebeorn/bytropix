CC = gcc
NVCC = nvcc
CFLAGS = -O2 -Wall -Wextra -Wno-unused-parameter -I include -fopenmp
LDFLAGS = -lm -fopenmp
NVCC_FLAGS = -O2 -I include -arch=sm_120
CUDA_LIBS = -lcublas -lcudart
CUDA_INC = -I/usr/local/cuda-13.1/include

# Remove old object dirs for clean build
.PHONY: all clean test test_poincare test_gpu test_gpu_run bench_e2e bench_e2e_run train_stub train_stub_run

all: test_ssm load_model test_gpu test_model

src/wubu_ssm.o: src/wubu_ssm.c include/wubu_ssm.h include/wubu_mobius.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_mobius.o: src/wubu_mobius.c include/wubu_mobius.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/gguf_reader.o: src/gguf_reader.c include/gguf_reader.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/cuda_kernels.o: src/cuda_kernels.cu include/cuda_kernels.h
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

test_ssm: test_ssm_forward.c src/wubu_ssm.o src/wubu_mobius.o src/gguf_reader.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_poincare_ssm: test_poincare_ssm.c src/wubu_ssm.o src/wubu_mobius.o src/gguf_reader.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_tokenizer: tools/test_tokenizer.c src/wubu_tokenizer.o src/gguf_reader.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_model: tools/test_model.c src/wubu_model.o src/wubu_ssm.o src/wubu_mobius.o src/gguf_reader.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_moe: tools/test_moe.c src/gguf_reader.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

load_model: tools/load_model_layer.c src/wubu_ssm.o src/wubu_mobius.o src/gguf_reader.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_gpu: tools/test_gpu.c src/wubu_ssm.o src/wubu_mobius.o src/gguf_reader.o src/cuda_kernels.o
	$(CC) $(CFLAGS) $(CUDA_INC) -o $@ $^ $(LDFLAGS) $(CUDA_LIBS) -L/usr/local/cuda/lib64 -lstdc++

src/bench.o: src/bench.c include/bench.h include/cuda_kernels.h include/wubu_ssm.h
	$(CC) $(CFLAGS) $(CUDA_INC) -c -o $@ $<

bench_e2e: tools/bench_e2e.c src/bench.o src/wubu_ssm.o src/wubu_mobius.o src/gguf_reader.o src/cuda_kernels.o
	$(CC) $(CFLAGS) $(CUDA_INC) -o $@ $^ $(LDFLAGS) $(CUDA_LIBS) -L/usr/local/cuda/lib64 -lstdc++

test: test_ssm
	./test_ssm

test_poincare: test_poincare_ssm
	./test_poincare_ssm

test_gpu_run: test_gpu
	./test_gpu

bench_e2e_run: bench_e2e
	./bench_e2e

train_stub: tools/train_stub.c
	$(CC) $(CFLAGS) -o $@ $< -lm -O0 -g

train_stub_run: train_stub
	./train_stub

test_parallel_scan: tools/test_parallel_scan.c src/wubu_ssm.o src/wubu_mobius.o src/gguf_reader.o src/cuda_kernels.o
	$(CC) $(CFLAGS) $(CUDA_INC) -o $@ $^ $(LDFLAGS) $(CUDA_LIBS) -L/usr/local/cuda/lib64 -lstdc++

test_fused: tools/test_fused.c src/wubu_ssm.o src/wubu_mobius.o src/gguf_reader.o src/cuda_kernels.o
	$(CC) $(CFLAGS) $(CUDA_INC) -o $@ $^ $(LDFLAGS) $(CUDA_LIBS) -L/usr/local/cuda/lib64 -lstdc++

test_fused_vs_old: tools/test_fused_vs_old.c src/wubu_ssm.o src/wubu_mobius.o src/gguf_reader.o src/cuda_kernels.o
	$(CC) $(CFLAGS) $(CUDA_INC) -o $@ $^ $(LDFLAGS) $(CUDA_LIBS) -L/usr/local/cuda/lib64 -lstdc++

tokenize_corpus: tools/tokenize_corpus.c src/wubu_tokenizer.o src/gguf_reader.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

train_real: tools/train_real.c src/wubu_model.o src/wubu_ssm.o src/wubu_mobius.o src/gguf_reader.o src/wubu_tokenizer.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

dump_mmproj: tools/dump_mmproj.c src/gguf_reader.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f test_ssm test_poincare_ssm load_model test_model test_gpu tokenize_corpus src/*.o
