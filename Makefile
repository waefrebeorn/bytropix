CC = gcc
NVCC = nvcc
CFLAGS = -O2 -Wall -Wextra -Wno-unused-parameter -I include -fopenmp
LDFLAGS = -lm -fopenmp
NVCC_FLAGS = -O2 -I include -arch=sm_120
CUDA_LIBS = -lcublas -lcudart
CUDA_INC = -I/usr/local/cuda-13.1/include

.PHONY: all clean

all: test_ssm load_model test_gpu test_model test_cpu_timing

# Object files
CORE_OBJ = src/wubu_ssm.o src/wubu_mobius.o src/wubu_moe.o src/wubu_moe_backward.o src/gguf_reader.o src/qlearner.o
MODEL_OBJ = src/wubu_model.o $(CORE_OBJ)
CUDA_OBJ = src/cuda_kernels.o
RSGD_OBJ = src/rsgd.o

src/qlearner.o: src/qlearner.c include/qlearner.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_ssm.o: src/wubu_ssm.c include/wubu_ssm.h include/wubu_mobius.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_mobius.o: src/wubu_mobius.c include/wubu_mobius.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_moe.o: src/wubu_moe.c include/wubu_moe.h include/wubu_ssm.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_moe_backward.o: src/wubu_moe_backward.c include/wubu_moe.h include/wubu_ssm.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/gguf_reader.o: src/gguf_reader.c include/gguf_reader.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_model.o: src/wubu_model.c include/wubu_model.h include/wubu_ssm.h include/wubu_moe.h include/gguf_reader.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/cuda_kernels.o: src/cuda_kernels.cu include/cuda_kernels.h
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

src/rsgd.o: src/rsgd.c include/rsgd.h include/gguf_reader.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/bench.o: src/bench.c include/bench.h include/cuda_kernels.h include/wubu_ssm.h
	$(CC) $(CFLAGS) $(CUDA_INC) -c -o $@ $<

# Test binaries
test_ssm: test_ssm_forward.c $(CORE_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_poincare_ssm: test_poincare_ssm.c $(CORE_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_tokenizer: tools/test_tokenizer.c src/wubu_tokenizer.o src/gguf_reader.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_model: tools/test_model.c $(MODEL_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_moe: tools/test_moe.c $(CORE_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

load_model: tools/load_model_layer.c $(CORE_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_gpu: tools/test_gpu.c $(CORE_OBJ) $(CUDA_OBJ)
	$(CC) $(CFLAGS) $(CUDA_INC) -o $@ $^ $(LDFLAGS) $(CUDA_LIBS) -L/usr/local/cuda/lib64 -lstdc++

bench_e2e: tools/bench_e2e.c src/bench.o $(CORE_OBJ) $(CUDA_OBJ)
	$(CC) $(CFLAGS) $(CUDA_INC) -o $@ $^ $(LDFLAGS) $(CUDA_LIBS) -L/usr/local/cuda/lib64 -lstdc++

test_parallel_scan: tools/test_parallel_scan.c $(CORE_OBJ) $(CUDA_OBJ)
	$(CC) $(CFLAGS) $(CUDA_INC) -o $@ $^ $(LDFLAGS) $(CUDA_LIBS) -L/usr/local/cuda/lib64 -lstdc++

test_fused: tools/test_fused.c $(CORE_OBJ) $(CUDA_OBJ)
	$(CC) $(CFLAGS) $(CUDA_INC) -o $@ $^ $(LDFLAGS) $(CUDA_LIBS) -L/usr/local/cuda/lib64 -lstdc++

test_fused_vs_old: tools/test_fused_vs_old.c $(CORE_OBJ) $(CUDA_OBJ)
	$(CC) $(CFLAGS) $(CUDA_INC) -o $@ $^ $(LDFLAGS) $(CUDA_LIBS) -L/usr/local/cuda/lib64 -lstdc++

# Training & tools
train_stub: tools/train_stub.c
	$(CC) $(CFLAGS) -o $@ $< -lm -O0 -g

tokenize_corpus: tools/tokenize_corpus.c src/wubu_tokenizer.o src/gguf_reader.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

train_real: tools/train_real.c $(MODEL_OBJ) src/wubu_tokenizer.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

train_backprop: tools/train_backprop.c $(MODEL_OBJ) src/wubu_tokenizer.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

train_gpu: tools/train_gpu.c src/bench.o $(MODEL_OBJ) $(CUDA_OBJ) src/wubu_tokenizer.o
	$(CC) $(CFLAGS) $(CUDA_INC) -o $@ $^ $(LDFLAGS) $(CUDA_LIBS) -L/usr/local/cuda/lib64 -lstdc++

dump_mmproj: tools/dump_mmproj.c src/gguf_reader.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

verify_iq2s: tools/verify_iq2s.c src/gguf_reader.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

verify_dequant: tools/verify_dequant.c src/gguf_reader.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

check_forward: tools/check_forward.c $(MODEL_OBJ) src/wubu_tokenizer.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Test runners
test: test_ssm
	./test_ssm

test_poincare: test_poincare_ssm
	./test_poincare_ssm

test_gpu_run: test_gpu
	./test_gpu

bench_e2e_run: bench_e2e
	./bench_e2e

train_stub_run: train_stub
	./train_stub

test_gpu_poincare: tools/test_gpu_poincare.c $(CORE_OBJ) $(CUDA_OBJ) src/bench.o
	$(CC) $(CFLAGS) $(CUDA_INC) -o $@ $^ $(LDFLAGS) $(CUDA_LIBS) -L/usr/local/cuda/lib64 -lstdc++

test_rsgd: tools/test_rsgd.c $(RSGD_OBJ) src/gguf_reader.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_backward: tools/test_backward.c $(CORE_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_bwd_model: tools/test_bwd_model.c $(MODEL_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_backward_simple: tools/test_backward_simple.c $(CORE_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# CPU timing + hedged spec tests (from tailslayer pattern)
test_cpu_timing: tools/test_cpu_timing.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) -lpthread

clean:
	rm -f test_ssm test_poincare_ssm load_model test_model test_gpu tokenize_corpus test_moe train_real bench_e2e verify_iq2s inspect_iq2s inspect_model train_backprop train_gpu test_gpu_poincare test_rsgd test_backward test_cpu_timing src/*.o tools/*.o
