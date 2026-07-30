CC = gcc
CXX = g++
# CUDA on this WSL box is the Debian nvidia-cuda-toolkit package (V12.0):
#   nvcc -> /usr/bin/nvcc, headers in /usr/include, libs in /usr/lib/x86_64-linux-gnu
# (NOT the NVIDIA .run layout /usr/local/cuda-X.Y/targets/...). Auto-detect and
# fall back to the Debian FHS paths so the same Makefile works on both layouts.
NVCC = $(or $(shell which nvcc 2>/dev/null),/usr/bin/nvcc)
# Derive /usr from /usr/bin/nvcc using a single shell command (avoid nesting
# Make $(...) inside $(shell ...), which Make mis-expands).
CUDA_HOME = $(shell d=$(NVCC); d=$${d%/*}; d=$${d%/*}; echo $$d)
CUDA_INC = -I$(CUDA_HOME)/include
CUDA_LIBDIR = $(shell if [ -d $(CUDA_HOME)/lib/x86_64-linux-gnu ]; then echo $(CUDA_HOME)/lib/x86_64-linux-gnu; else echo $(CUDA_HOME)/lib64; fi)
CFLAGS = -O3 -march=native -ffast-math -funroll-loops -ftree-vectorize -Wall -Wextra -Wno-unused-parameter -I include $(CUDA_INC) -fopenmp
LDFLAGS = -lm -fopenmp -L$(CUDA_LIBDIR) -lcudart -lcublas -lpthread -lssl -lcrypto
NVCC_FLAGS = -O3 -I include -arch=sm_86
CUDA_INCS = $(CUDA_INC)
CUDA_LIBS = -L$(CUDA_LIBDIR) -lcublas -lcudart
CUDA_LIB = -L$(CUDA_LIBDIR) -lcudart

.PHONY: all clean

all: test_nested_ssm test_nested_ssm_backward load_model test_model test_cpu_timing test_model_adapter infer_moe infer_moe_lazy infer_unified infer_vision infer_poincare infer_vision_gpu test_256k test_kv_cache infer_vision_text test_poincare_gqa test_tst test_moe_hyperbolic test_mobius_linear test_hyperbolic_output_proj train_integrated test_chunked_ssm api_server test_st_bridge test_btl3_lora

api_server: tools/api_server.c
	$(CC) -O2 -g -Wall -I include -o $@ $< -lssl -lcrypto -lm

# Object files
CORE_OBJ = src/wubu_model.o src/wubu_dims.o src/wubu_dims_gpu.o src/wubu_ssm.o src/wubu_ssm_workspace.o src/wubu_ssm_chunked.o src/wubu_mobius.o src/wubu_nested_ssm.o src/wubu_nested_ssm_backward.o src/wubu_moe.o src/wubu_moe_backward.o src/wubu_moe_hyperbolic.o src/wubu_poincare_ssm_backward.o src/wubu_poincare_gqa.o src/wubu_poincare_gqa_backward.o src/wubu_mobius_linear.o src/wubu_hyperbolic_output_proj.o src/wubu_vision.o src/gguf_reader.o src/qlearner.o src/rsgd.o src/wubu_tst.o src/dequant_iq2_xxs.o src/quantized_matmul.o src/quantized_dot_generic.o src/safetensors_reader.o src/wubu_repetition.o src/wubu_lora.o src/wubu_model_adapter.o src/wubu_model_safetensors_bridge.o src/wubu_safetensors_shard.o src/wubu_ssd_moe.o src/wubu_gemm.o src/wubu_kvcache_quant.o src/wubu_ssm_scan.o src/wubu_roofline.o src/wubu_kv_select.o src/wubu_kv_runtime.o src/wubu_gemv_tune.o src/wubu_affinity.o src/wubu_rotate.o src/wubu_flashdecode.o src/wubu_kvvq.o src/wubu_spec_decode.o src/wubu_generate.o src/wubu_ternary.o src/wubu_smoothquant.o src/wubu_arena.o src/wubu_prefix_cache.o src/wubu_paged_kv.o src/wubu_q4k_m.o src/wubu_delta_net.o src/wubu_scheduler.o src/wubu_ngram.o src/wubu_self_cascade.o src/wubu_spec_cascade.o src/wubu_spawn.o src/wubu_kv_styx.o src/wubu_kv_tier.o src/wubu_attn_gate.o
MODEL_OBJ = $(CORE_OBJ)
CUDA_OBJ = src/cuda_kernels.o src/gpu_output_proj.o src/flash_attn_q4_0_opt.o src/flash_attn_q4_0_prefill_opt.o
GPU_OBJ = src/wubu_model_gpu.o src/gpu_quant_matmul.o src/gpu_quant_matmul_row_major.o src/gpu_moe_kernel.o src/gpu_ssm_recurrence.o src/wubu_kv_runtime.o src/wubu_gemv_tune.o src/wubu_affinity.o src/wubu_rotate.o src/wubu_flashdecode.o src/wubu_kvvq.o src/wubu_spec_decode.o src/wubu_generate.o src/wubu_ternary.o src/wubu_smoothquant.o src/wubu_arena.o
RSGD_OBJ = src/rsgd.o

# New Colonel-model support (safetensors + LoRA + repetition + adapter)
NEW_OBJ = src/safetensors_reader.o src/wubu_repetition.o src/wubu_lora.o \
          src/wubu_model_adapter.o src/wubu_model_safetensors_bridge.o \
          src/wubu_safetensors_shard.o src/wubu_ssd_moe.o

# WuBuOS EDR layer (linked directly into the agent gauntlet; no daemon).
# Path to the WuBuOS checkout; override on the command line if checked out elsewhere.
WUBUOS ?= /home/wubu/wubuos
EDR_INC = -I$(WUBUOS)/src/runtime -I$(WUBUOS)/src/runtime/edr
EDR_SRC  = $(WUBUOS)/src/runtime/wubu_edr.c \
           $(WUBUOS)/src/runtime/edr/edr_core.c \
           $(WUBUOS)/src/runtime/edr/edr_proc_pin.c \
           $(WUBUOS)/src/runtime/edr/edr_fanotify.c \
           $(WUBUOS)/src/runtime/edr/edr_poller.c

src/wubu_ssd_moe.o: src/wubu_ssd_moe.c include/wubu_ssd_moe.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/safetensors_reader.o: src/safetensors_reader.c include/safetensors_reader.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_safetensors_shard.o: src/wubu_safetensors_shard.c include/wubu_safetensors_shard.h include/safetensors_reader.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_dims.o: src/wubu_dims.c include/wubu_dims.h
	$(CC) $(CFLAGS) -c $< -o $@

src/wubu_dims_gpu.o: src/wubu_dims_gpu.cu include/wubu_dims.h
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

src/wubu_model_safetensors_bridge.o: src/wubu_model_safetensors_bridge.c include/wubu_model_safetensors_bridge.h include/wubu_model.h include/wubu_lora.h include/wubu_model_adapter.h include/wubu_affinity.h include/wubu_rotate.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_repetition.o: src/wubu_repetition.c include/wubu_repetition.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_lora.o: src/wubu_lora.c include/wubu_lora.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_model_adapter.o: src/wubu_model_adapter.c include/wubu_model_adapter.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_safetensors_model.o: src/wubu_safetensors_model.c include/wubu_safetensors_model.h include/safetensors_reader.h include/wubu_model_adapter.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/qlearner.o: src/qlearner.c include/qlearner.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_ssm.o: src/wubu_ssm.c include/wubu_ssm.h include/wubu_mobius.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_ssm_chunked.o: src/wubu_ssm_chunked.c include/wubu_ssm.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_nested_ssm.o: src/wubu_nested_ssm.c include/wubu_nested_ssm.h include/wubu_ssm.h include/wubu_mobius.h include/gguf_reader.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_nested_ssm_backward.o: src/wubu_nested_ssm_backward.c include/wubu_nested_ssm.h include/wubu_ssm.h include/wubu_mobius.h include/gguf_reader.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_mobius.o: src/wubu_mobius.c include/wubu_mobius.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_moe.o: src/wubu_moe.c include/wubu_moe.h include/wubu_ssm.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_moe_hyperbolic.o: src/wubu_moe_hyperbolic.c include/wubu_moe_hyperbolic.h include/wubu_moe.h include/wubu_mobius.h include/wubu_ssm.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_moe_backward.o: src/wubu_moe_backward.c include/wubu_moe.h include/wubu_ssm.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_poincare_ssm_backward.o: src/wubu_poincare_ssm_backward.c include/wubu_ssm.h include/wubu_mobius.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_poincare_gqa.o: src/wubu_poincare_gqa.c include/wubu_poincare_gqa.h include/wubu_ssm.h include/wubu_mobius.h include/gguf_reader.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_poincare_gqa_backward.o: src/wubu_poincare_gqa_backward.c include/wubu_poincare_gqa.h include/wubu_ssm.h include/wubu_mobius.h include/gguf_reader.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_prefix_cache.o: src/wubu_prefix_cache.c include/wubu_prefix_cache.h include/wubu_paged_kv.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_paged_kv.o: src/wubu_paged_kv.c include/wubu_paged_kv.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_q4k_m.o: src/wubu_q4k_m.c include/wubu_q4k_m.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_delta_net.o: src/wubu_delta_net.c include/wubu_delta_net.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_ngram.o: src/wubu_ngram.c include/wubu_ngram.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_spec_cascade.o: src/wubu_spec_cascade.c include/wubu_spec_cascade.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_scheduler.o: src/wubu_scheduler.c include/wubu_scheduler.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_mobius_linear.o: src/wubu_mobius_linear.c include/wubu_mobius_linear.h include/wubu_mobius.h include/gguf_reader.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_mobius_gyrate.o: src/wubu_mobius_gyrate.c include/wubu_mobius.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_moe_hyperbolic_backward.o: src/wubu_moe_hyperbolic_backward.c include/wubu_moe_hyperbolic.h include/wubu_mobius.h include/gguf_reader.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_hyperbolic_output_proj.o: src/wubu_hyperbolic_output_proj.c include/wubu_hyperbolic_output_proj.h include/wubu_mobius_linear.h include/gguf_reader.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_vision.o: src/wubu_vision.c include/wubu_vision.h include/wubu_ssm.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/gguf_reader.o: src/gguf_reader.c include/gguf_reader.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_generate.o: src/wubu_generate.c include/wubu_generate.h include/wubu_model.h include/wubu_spec_decode.h
src/wubu_kvvq.o: src/wubu_kvvq.c include/wubu_kvvq.h
src/wubu_arena.o: src/wubu_arena.c include/wubu_arena.h
src/wubu_smoothquant.o: src/wubu_smoothquant.c include/wubu_smoothquant.h
src/wubu_ternary.o: src/wubu_ternary.c include/wubu_ternary.h
src/wubu_spec_decode.o: src/wubu_spec_decode.c include/wubu_spec_decode.h
src/wubu_flashdecode.o: src/wubu_flashdecode.c include/wubu_flashdecode.h
src/wubu_rotate.o: src/wubu_rotate.c include/wubu_rotate.h
src/wubu_model.o: src/wubu_model.c include/wubu_model.h include/wubu_ssm.h include/wubu_moe.h include/gguf_reader.h include/wubu_kv_select.h include/wubu_kv_runtime.h include/wubu_affinity.h include/wubu_rotate.h
src/wubu_gemm.o: src/wubu_gemm.c include/wubu_gemm.h
src/wubu_gemv_tune.o: src/wubu_gemv_tune.c include/wubu_gemv_tune.h include/wubu_roofline.h
src/quantized_matmul.o: src/quantized_matmul.c include/wubu_gemm.h include/wubu_gemv_tune.h include/gguf_reader.h include/wubu_ssm.h include/wubu_safetensors_shard.h
src/wubu_kv_runtime.o: src/wubu_kv_runtime.c include/wubu_kv_runtime.h include/wubu_kv_select.h include/wubu_roofline.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/cuda_kernels.o: src/cuda_kernels.cu include/cuda_kernels.h
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

src/gpu_output_proj.o: src/gpu_output_proj.cu include/gpu_output_proj.h
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

src/wubu_model_gpu.o: src/wubu_model_gpu.cu include/wubu_model.h include/cuda_kernels.h include/bench.h
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

src/gpu_quant_matmul.o: src/gpu_quant_matmul.cu include/gpu_quant_matmul.h include/gguf_reader.h
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

src/gpu_quant_matmul_row_major.o: src/gpu_quant_matmul_row_major.cu include/gpu_quant_matmul.h include/gguf_reader.h
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

src/gpu_moe_kernel.o: src/gpu_moe_kernel.cu include/gpu_moe_kernel.h include/gguf_reader.h src/iq2xxs_grid_data.inc src/iq3xxs_grid.inc
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

src/gpu_ssm_recurrence.o: src/gpu_ssm_recurrence.cu include/gpu_ssm_recurrence.h
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

src/flash_attn_q4_0_opt.o: src/flash_attn_q4_0_opt.cu include/cuda_kernels.h
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

src/flash_attn_q4_0_prefill_opt.o: src/flash_attn_q4_0_prefill_opt.cu include/cuda_kernels.h
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

src/gpu_gemma4.o: src/gpu_gemma4.cu include/gpu_gemma4.h include/wubu_gemma4.h
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

src/gpu_gemma4_forward.o: src/gpu_gemma4_forward.cu include/gpu_gemma4.h include/wubu_gemma4.h
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

src/rsgd.o: src/rsgd.c include/rsgd.h include/gguf_reader.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_tst.o: src/wubu_tst.c include/wubu_tst.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/bench.o: src/bench.c include/bench.h include/cuda_kernels.h include/wubu_ssm.h
	$(CC) $(CFLAGS) $(CUDA_INC) -c -o $@ $<

src/dequant_iq2_xxs.o: src/dequant_iq2_xxs.c include/gguf_reader.h
	$(CC) $(CFLAGS) -c -o $@ $<

# Test binaries

test_chunked_ssm: tools/test_chunked_ssm.c src/wubu_moe_cpu.o $(filter-out src/wubu_moe.o,$(CORE_OBJ))
	$(CC) $(CFLAGS) -o $@ tools/test_chunked_ssm.c src/wubu_moe_cpu.o $(filter-out src/wubu_moe.o,$(CORE_OBJ)) $(LDFLAGS)

test_decode_path: tools/test_decode_path.c $(MODEL_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_nested_ssm: tools/test_nested_ssm.c $(CORE_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_nested_ssm_backward: tools/test_nested_ssm_backward.c $(CORE_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)


test_poincare_gqa: tools/test_poincare_gqa.c $(CORE_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_poincare_kv_cache: tools/test_poincare_kv_cache.c $(CORE_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_pga_backward: tools/test_pga_backward.c $(CORE_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_mobius_linear: tools/test_mobius_linear.c $(CORE_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_hyperbolic_output_proj: tools/test_hyperbolic_output_proj.c $(CORE_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_gpu_layers: tools/test_gpu_layers.c $(CORE_OBJ) $(CUDA_OBJ) src/bench.o
	$(CC) $(CFLAGS) $(CUDA_INC) -o $@ $^ $(LDFLAGS) $(CUDA_LIBS) -L$(CUDA_LIBDIR) -lstdc++

test_gyrate: tools/test_gyrate.c src/wubu_mobius.o src/wubu_mobius_gyrate.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_poincare_router_backward: tools/test_poincare_router_backward.c $(CORE_OBJ) src/wubu_moe_hyperbolic_backward.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_nested_moe_router_backward: tools/test_nested_moe_router_backward.c $(CORE_OBJ) src/wubu_moe_hyperbolic_backward.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# ---- New Colonel-model unit tests ----
gen_fixture_safetensors: tools/gen_fixture_safetensors.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

test_kv_styx: tools/test_kv_styx.c src/wubu_kv_styx.o src/wubu_spawn.o
	$(CC) $(CFLAGS) -o $@ tools/test_kv_styx.c src/wubu_kv_styx.o src/wubu_spawn.o $(LDFLAGS)
	./$@

test_kivi_roundtrip: tools/test_kivi_roundtrip.c src/wubu_kvcache_quant.o
	$(CC) $(CFLAGS) -o test_kivi_roundtrip $< src/wubu_kvcache_quant.o $(LDFLAGS) -lm
	./test_kivi_roundtrip

test_kv_bw: tools/test_kv_bw.c src/wubu_kvcache_quant.o
	$(CC) $(CFLAGS) -o test_kv_bw $< src/wubu_kvcache_quant.o $(LDFLAGS) -lm
	./test_kv_bw

test_safetensors: tools/test_safetensors.c src/safetensors_reader.o gen_fixture_safetensors
	$(CC) $(CFLAGS) -o $@ $< src/safetensors_reader.o $(LDFLAGS)
	./gen_fixture_safetensors
	./$@

test_repetition: tools/test_repetition.c src/wubu_repetition.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	./$@

test_lora: tools/test_lora.c src/wubu_lora.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	./$@

test_model_adapter: tools/test_model_adapter.c src/wubu_model_adapter.o $(CORE_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	./$@

test_new_models: test_safetensors test_repetition test_lora test_model_adapter test_st_bridge
	@echo "=== new Colonel-model unit tests PASSED ==="

gen_fixture_safetensors_model: tools/gen_fixture_safetensors_model.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

gen_fixture_btl3_lora: tools/gen_fixture_btl3_lora.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

test_st_bridge: tools/test_st_bridge.c $(MODEL_OBJ) gen_fixture_safetensors_model
	$(CC) $(CFLAGS) -o $@ tools/test_st_bridge.c $(MODEL_OBJ) $(LDFLAGS)
	./gen_fixture_safetensors_model
	./$@

test_btl3_lora: tools/test_btl3_lora.c $(MODEL_OBJ) gen_fixture_safetensors_model gen_fixture_btl3_lora
	$(CC) $(CFLAGS) -o $@ tools/test_btl3_lora.c $(MODEL_OBJ) $(LDFLAGS)
	./gen_fixture_safetensors_model
	./gen_fixture_btl3_lora
	./$@

test_real_load: tools/test_real_load.c src/wubu_model_adapter.o $(MODEL_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	./$@

test_ssd_moe: tools/test_ssd_moe.c src/wubu_ssd_moe.o src/wubu_safetensors_shard.o src/safetensors_reader.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# ── 100-improvement modules (Areas A/B/C/D/F/H/I/J/K) ───────────────
test_spec_decode: tools/test_spec_decode.c src/wubu_spec_decode.o src/wubu_ngram.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	./$@

test_kvquant: tools/test_kvquant.c src/wubu_kvquant.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	./$@

test_paged_kv: tools/test_paged_kv.c src/wubu_paged_kv.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	./$@

test_moe_grouped: tools/test_moe_grouped.c src/wubu_moe_grouped.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	./$@

test_ssm_scan: tools/test_ssm_scan.c src/wubu_ssm_scan.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	./$@

test_q8: tools/test_q8.c src/wubu_q8.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	./$@

test_cuda_graph: tools/test_cuda_graph.c src/wubu_cuda_graph.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	./$@

# ── Round-2 (cross-disciplinary) modules (Areas L/M/N/O) ───────
test_roofline: tools/test_roofline.c src/wubu_roofline.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	./$@

test_cache_advice: tools/test_cache_advice.c src/wubu_cache_advice.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	./$@

test_kereq: tools/test_kereq.c src/wubu_kereq.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	./$@

test_pd_split: tools/test_pd_split.c src/wubu_pd_split.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	./$@

# ── Round-3 (new-model architecture: hybrid/recurrent, mHC, CLA, MEGA, YaRN) ──
test_delta_net: tools/test_delta_net.c src/wubu_delta_net.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	./$@

test_mhc: tools/test_mhc.c src/wubu_mhc.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	./$@

test_cla: tools/test_cla.c src/wubu_cla.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	./$@

test_mega: tools/test_mega.c src/wubu_mega.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	./$@

test_yarn: tools/test_yarn.c src/wubu_yarn.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	./$@

# ── Round-4 (Kimi K3: KDA, AttnRes, Stable LatentMoE, MXFP4/MXFP8) ──
test_kda: tools/test_kda.c src/wubu_kda.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	./$@

test_attnres: tools/test_attnres.c src/wubu_attnres.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	./$@

test_latentmoe: tools/test_latentmoe.c src/wubu_latentmoe.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	./$@

test_mxfp4: tools/test_mxfp4.c src/wubu_mxfp4.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	./$@

test_scheduler: tools/test_scheduler.c src/wubu_scheduler.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	./$@

test_affinity: tools/test_affinity.c src/wubu_affinity.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	./$@

# Aggregate of ALL 400 improvement unit tests (Round-1 + Round-2 + Round-3 + Round-4).
test_400: test_spec_decode test_kvquant test_paged_kv test_moe_grouped \
          test_ssm_scan test_q8 test_cuda_graph test_scheduler test_affinity \
          test_roofline test_cache_advice test_kereq test_pd_split \
          test_delta_net test_mhc test_cla test_mega test_yarn \
          test_kda test_attnres test_latentmoe test_mxfp4
	@echo "ALL 400-IMPROVEMENT UNIT TESTS PASSED"

# Aggregate of ALL 300 improvement unit tests (Round-1 + Round-2 + Round-3).
test_300: test_spec_decode test_kvquant test_paged_kv test_moe_grouped \
          test_ssm_scan test_q8 test_cuda_graph test_scheduler test_affinity \
          test_roofline test_cache_advice test_kereq test_pd_split \
          test_delta_net test_mhc test_cla test_mega test_yarn
	@echo "ALL 300-IMPROVEMENT UNIT TESTS PASSED"

# Aggregate of ALL 200 improvement unit tests (Round-1 + Round-2).
test_200: test_spec_decode test_kvquant test_paged_kv test_moe_grouped \
          test_ssm_scan test_q8 test_cuda_graph test_scheduler test_affinity \
          test_roofline test_cache_advice test_kereq test_pd_split
	@echo "ALL 200-IMPROVEMENT UNIT TESTS PASSED"

test_model_config: tools/test_model_config.c src/wubu_model_adapter.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	./$@

test_tokenizer: tools/test_tokenizer.c src/wubu_tokenizer.o src/gguf_reader.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_model: tools/test_model.c $(MODEL_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# ── Agent tool gauntlet (4 Colonel models × 3 tools, EDR fan-out) ──────────
# Links the wubuwizard engine + WuBuOS EDR layer. The EDR sources are
# standalone (lock-free ring + worker thread); no daemon required.
gauntlet: tools/agent_gauntlet/agent_gauntlet.c tools/agent_gauntlet/gauntlet_run.c \
          tools/agent_gauntlet/agent_gauntlet.h $(MODEL_OBJ) $(EDR_SRC) \
          src/wubu_tokenizer_hf.o src/wubu_tokenizer.o
	$(CC) $(CFLAGS) $(EDR_INC) -o $@ tools/agent_gauntlet/agent_gauntlet.c \
		tools/agent_gauntlet/gauntlet_run.c $(MODEL_OBJ) $(EDR_SRC) \
		src/wubu_tokenizer_hf.o src/wubu_tokenizer.o $(LDFLAGS) -lpthread

test_gauntlet: tools/agent_gauntlet/agent_gauntlet.c tools/agent_gauntlet/test_gauntlet.c \
               tools/agent_gauntlet/agent_gauntlet.h $(MODEL_OBJ) $(EDR_SRC) \
               src/wubu_tokenizer_hf.o src/wubu_tokenizer.o
	$(CC) $(CFLAGS) $(EDR_INC) -o $@ tools/agent_gauntlet/agent_gauntlet.c \
		tools/agent_gauntlet/test_gauntlet.c $(MODEL_OBJ) $(EDR_SRC) \
		src/wubu_tokenizer_hf.o src/wubu_tokenizer.o $(LDFLAGS) -lpthread
	./$@

# Verify the principled GDN chunkwise-parallel recurrence vs the sequential
# scalar reference (must match to ~1e-2 at every chunk size C).
test_gdn_chunk: tools/agent_gauntlet/test_gdn_chunk.c $(MODEL_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	./$@

# Our-own-kernel GEMM benchmark: baseline scalar vs tiled AVX2/AVX512-FMA.
bench_gemm: tools/bench_gemm.c src/wubu_gemm.o
	$(CC) $(CFLAGS) -mavx2 -mfma -o $@ $^ -lm -fopenmp
	./$@

# WuBuOS-agnostic kernel dispatch: prove device-backend registry works.
test_kernel_dispatch: tools/test_kernel_dispatch.c src/wubu_gemm.o
	$(CC) $(CFLAGS) -mavx2 -mfma -I include -o $@ $^ -lm -fopenmp
	./$@

bench_gemm_run: bench_gemm

# Projection accuracy: quantized_matmul GEMV vs unambiguous scalar oracle
# on REAL Qwen layer-0 gate_proj weights. Catches the transpose-layout bug.
# CPU-only link (exclude CUDA/GPU objects that need -lcuda).
CPU_OBJ = $(filter-out src/wubu_dims_gpu.o src/cuda_kernels.o src/gpu_output_proj.o src/flash_attn_q4_0_opt.o src/flash_attn_q4_0_prefill_opt.o src/wubu_model_gpu.o src/gpu_quant_matmul.o src/gpu_quant_matmul_row_major.o src/gpu_moe_kernel.o src/gpu_ssm_recurrence.o,$(CORE_OBJ)) src/wubu_dims_gpu_stub.o
test_proj_accuracy: tools/test_proj_accuracy.c $(CPU_OBJ)
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm -fopenmp
	./$@

# KV-cache quantization integration (engine's real kv_cache_read/write_head
# under each scheme) + roofline-driven wubu_kv_select. Three builds:
#  -DWAIT no — builds with KV_CACHE_OUR_Q8, KV_CACHE_KIVI, and default.
test_kv_cache_q8: tools/test_kv_cache_integration.c $(CPU_OBJ) src/wubu_kv_select.o
	$(CC) $(CFLAGS) -DKV_CACHE_OUR_Q8 -I include -o $@ $^ -lm -fopenmp
	./$@

test_kv_cache_kivi: tools/test_kv_cache_integration.c $(CPU_OBJ) src/wubu_kv_select.o
	$(CC) $(CFLAGS) -DKV_CACHE_KIVI -I include -o $@ $^ -lm -fopenmp
	./$@

# KV-cache quant unit test (Q8_0 + KIVI K!=V axes). Round-trip + edge.
# Roofline-driven GEMV auto-tuner: tiled fp32 + int8 variants vs scalar oracle
# + tuner sanity. Real Qwen weight used for one probe (skip-safe if absent).
# B03: int4 weight GEMV vs fp32 oracle + autotune precedence
# doc 018/K01: n-gram speculative decoding generator (equivalence vs plain)
test_generate_spec: tools/test_generate_spec.c src/wubu_generate.o src/wubu_spec_decode.o $(CPU_OBJ)
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm -fopenmp
	./$@

# doc 006: arena allocator for per-request + KV buffers
test_arena: tools/test_arena.c src/wubu_arena.o $(CPU_OBJ)
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm -fopenmp -lssl -lcrypto
	./$@

# doc 005: SmoothQuant activation outlier migration
test_smoothquant: tools/test_smoothquant.c src/wubu_smoothquant.o $(CPU_OBJ)
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm -fopenmp -lssl -lcrypto
	./$@

# doc 004: BitNet 1.58 ternary {-1,0,+1} GEMV
test_ternary: tools/test_ternary.c src/wubu_ternary.o $(CPU_OBJ)
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm -fopenmp
	./$@

# doc 014: sub-4-bit KV vector quantization (CommVQ/TurboQuant)
test_kvvq: tools/test_kvvq.c src/wubu_kvvq.o $(CPU_OBJ)
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm -fopenmp
	./$@

# doc 002: KV multi-tier storage (hot/warm/cold)
test_kv_tier: tools/test_kv_tier.c src/wubu_kv_tier.o $(CPU_OBJ)
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm -fopenmp
	./$@

# doc 011: attention-sink-free gated attention
test_attn_gate: tools/test_attn_gate.c src/wubu_attn_gate.o $(CPU_OBJ)
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm -fopenmp
	./$@

# doc 015: FlashDecoding-style parallel KV-load decode attention
test_flashdecode: tools/test_flashdecode.c src/wubu_flashdecode.o $(CPU_OBJ)
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm -fopenmp
	./$@

# doc 013: QuaRot/Hadamard rotation invariance + outlier-suppression
test_rotate: tools/test_rotate.c src/wubu_rotate.o $(CPU_OBJ)
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm -fopenmp
	./$@

test_gemv_int4: tools/test_gemv_int4.c $(CPU_OBJ)
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm -fopenmp
	./$@

test_gemv_tune: tools/test_gemv_tune.c $(CPU_OBJ)
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm -fopenmp
	./$@

test_kvcache_quant: tools/test_kvcache_quant.c $(CPU_OBJ)
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm -fopenmp
	./$@

test_moe: tools/test_moe.c $(CORE_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_moe_hyperbolic: tools/test_moe_hyperbolic.c $(CORE_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_full_moe: tools/test_full_moe.c $(MODEL_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_rope_t2: tools/test_rope_t2.c $(MODEL_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

gen_text: tools/gen_text.c $(MODEL_OBJ) src/wubu_tokenizer.o src/wubu_tokenizer_hf.o
	$(CC) $(CFLAGS) -o $@ $< $(MODEL_OBJ) src/wubu_tokenizer.o src/wubu_tokenizer_hf.o $(LDFLAGS)
# CPU-only gen_text (recompiles wubu_model + wubu_moe without GPU_SUPPORT)
gen_text_cpu: CFLAGS_FILTERED = $(filter-out -I$(CUDA_INC),$(CFLAGS))
gen_text_cpu: src/wubu_model_cpu.o src/wubu_moe_cpu.o $(filter-out src/wubu_moe.o,$(CORE_OBJ)) src/wubu_tokenizer.o
	$(CC) $(CFLAGS_FILTERED) -o $@ tools/gen_text.c src/wubu_model_cpu.o src/wubu_moe_cpu.o $(filter-out src/wubu_moe.o,$(CORE_OBJ)) src/wubu_tokenizer.o $(LDFLAGS)
	@echo "gen_text_cpu built (CPU-only, no GPU support)"

src/wubu_model_cpu.o: src/wubu_model.c include/wubu_model.h include/wubu_ssm.h include/wubu_moe.h include/gguf_reader.h
	$(CC) $(CFLAGS) -o $@ -c $<

# CPU-only wubu_moe (no GPU_SUPPORT)
src/wubu_moe_cpu.o: src/wubu_moe.c include/wubu_moe.h include/wubu_ssm.h
	$(CC) $(CFLAGS) -o $@ -c $<

run_bos: tools/run_bos.c $(MODEL_OBJ)
	$(CC) $(CFLAGS) -o $@ tools/run_bos.c $(MODEL_OBJ) $(LDFLAGS)

# Debug build (gdb/ASAN). Compiles gen_text + model objects with -g -O0,
# no GPU_SUPPORT, single-file objects (no _cpu variant clash).
gen_text_dbg: CFLAGS_DBG = -g -O0 -I include $(CUDA_INC) -fopenmp -Wall
gen_text_dbg: tools/gen_text.c $(MODEL_OBJ)
	$(CC) $(CFLAGS_DBG) -o $@ $< $(MODEL_OBJ) src/wubu_tokenizer.o src/wubu_tokenizer_hf.o $(LDFLAGS)

gen_text_asan: CFLAGS_ASAN = -g -O1 -fsanitize=address -I include $(CUDA_INC) -fopenmp
gen_text_asan: tools/gen_text.c $(MODEL_OBJ)
	$(CC) $(CFLAGS_ASAN) -o $@ $< $(MODEL_OBJ) src/wubu_tokenizer.o src/wubu_tokenizer_hf.o $(LDFLAGS)

# Probe: load real Qwen3.6-27B (MAX_LAYERS=1) and print layer-0 weight
# pointers + state buffers, to diagnose SSM forward crashes.
test_probe_qwen: tools/test_probe_qwen.c $(MODEL_OBJ)
	$(CC) $(CFLAGS) -o $@ $< $(MODEL_OBJ) src/wubu_tokenizer.o $(LDFLAGS)

# Verify the ds4-ssd MoE decode bank pages real KAT experts from the source
# checkpoint shards (no sidecar copy).
test_kat_decode_bank: tools/test_kat_decode_bank.c $(MODEL_OBJ)
	$(CC) $(CFLAGS) -I include -o $@ $< $(MODEL_OBJ) src/wubu_tokenizer.o $(LDFLAGS)

# ASAN variant for pinning SSM-forward heap bugs.
test_probe_qwen_asan: CFLAGS_ASAN = -O1 -g -fsanitize=address -mavx2 -mfma -I include $(CUDA_INC) -fopenmp
test_probe_qwen_asan: tools/test_probe_qwen.c $(MODEL_OBJ) src/wubu_tokenizer.o
	$(CC) $(CFLAGS_ASAN) -o $@ $< $(MODEL_OBJ) src/wubu_tokenizer.o $(LDFLAGS)

gen_text_mtp: tools/gen_text_mtp.c $(MODEL_OBJ) src/wubu_tokenizer.o
	$(CC) $(CFLAGS) -o $@ $(filter %.c %.o,$^) $(LDFLAGS)

gen_text_gpu: tools/gen_text.c $(MODEL_OBJ) src/wubu_tokenizer.o src/wubu_repetition.o $(CUDA_OBJ) $(GPU_OBJ)
	$(CXX) $(CFLAGS) -DGPU_SUPPORT -o $@ tools/gen_text.c $(MODEL_OBJ) src/wubu_tokenizer.o $(CUDA_OBJ) $(GPU_OBJ) $(LDFLAGS) -L$(CUDA_LIBDIR) -lcublas -lcudart

test_tok_debug: tools/test_tok_debug.c src/wubu_tokenizer.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

ref_dumper: tools/ref_dumper.cpp
	$(CXX) $(CFLAGS) -std=c++17 -I $(HOME)/llama.cpp/include -I $(HOME)/llama.cpp/ggml/include -o $@ $^ $(LDFLAGS) $(HOME)/llama.cpp/build/bin/libllama.so $(HOME)/llama.cpp/build/bin/libggml.so $(HOME)/llama.cpp/build/bin/libggml-cpu.so $(HOME)/llama.cpp/build/bin/libggml-base.so -Wl,-rpath,$(HOME)/llama.cpp/build/bin

ref_dumper_mtp: tools/ref_dumper_mtp.cpp
	$(CXX) $(CFLAGS) -std=c++17 -I $(HOME)/llama.cpp/include -I $(HOME)/llama.cpp/src -I $(HOME)/llama.cpp/ggml/include -o $@ $^ $(LDFLAGS) $(HOME)/llama.cpp/build/bin/libllama.so $(HOME)/llama.cpp/build/bin/libggml.so $(HOME)/llama.cpp/build/bin/libggml-cpu.so $(HOME)/llama.cpp/build/bin/libggml-base.so -Wl,-rpath,$(HOME)/llama.cpp/build/bin

test_quantized_matmul: tools/test_quantized_matmul.c $(CORE_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_vec_dot_types: tools/test_vec_dot_types.c $(CORE_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_iq_dot: tools/test_iq_dot.c $(CORE_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

load_model: tools/load_model_layer.c $(CORE_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_gpu: tools/test_gpu.c $(CORE_OBJ) $(CUDA_OBJ)
	$(CC) $(CFLAGS) $(CUDA_INC) -o $@ $^ $(LDFLAGS) $(CUDA_LIBS) -L$(CUDA_LIBDIR) -lstdc++

bench_e2e: tools/bench_e2e.c src/bench.o $(CORE_OBJ) $(CUDA_OBJ)
	$(CC) $(CFLAGS) $(CUDA_INC) -o $@ $^ $(LDFLAGS) $(CUDA_LIBS) -L$(CUDA_LIBDIR) -lstdc++

test_parallel_scan: tools/test_parallel_scan.c $(CORE_OBJ) $(CUDA_OBJ)
	$(CC) $(CFLAGS) $(CUDA_INC) -o $@ $^ $(LDFLAGS) $(CUDA_LIBS) -L$(CUDA_LIBDIR) -lstdc++

test_fused: tools/test_fused.c $(CORE_OBJ) $(CUDA_OBJ)
	$(CC) $(CFLAGS) $(CUDA_INC) -o $@ $^ $(LDFLAGS) $(CUDA_LIBS) -L$(CUDA_LIBDIR) -lstdc++

test_fused_vs_old: tools/test_fused_vs_old.c $(CORE_OBJ) $(CUDA_OBJ)
	$(CC) $(CFLAGS) $(CUDA_INC) -o $@ $^ $(LDFLAGS) $(CUDA_LIBS) -L$(CUDA_LIBDIR) -lstdc++

debug_beta_layout: tools/debug_beta_layout.c src/gguf_reader.o src/dequant_iq2_xxs.o src/cuda_kernels.o
	$(CC) $(CFLAGS) $(CUDA_INC) -o $@ $^ $(LDFLAGS) $(CUDA_LIBS) -L$(CUDA_LIBDIR) -lstdc++

verify_phase26: tools/verify_phase26_fusions.c $(CORE_OBJ) $(CUDA_OBJ)
	$(CC) $(CFLAGS) $(CUDA_INC) -o $@ $^ $(LDFLAGS) $(CUDA_LIBS) -L$(CUDA_LIBDIR) -lstdc++

train_integrated: tools/train_integrated.c $(MODEL_OBJ) src/wubu_tokenizer.o $(CUDA_OBJ) src/bench.o $(GPU_OBJ)
	$(CC) $(CFLAGS) $(CUDA_INC) -o $@ $^ $(LDFLAGS) $(CUDA_LIBS) -L$(CUDA_LIBDIR) -lstdc++

infer_text: tools/infer_text.c $(MODEL_OBJ) src/wubu_tokenizer.o src/bench.o $(CUDA_OBJ)
	$(CC) $(CFLAGS) $(CUDA_INC) -o $@ $^ $(LDFLAGS) $(CUDA_LIBS) -L$(CUDA_LIBDIR) -lstdc++

infer_text_gpu: tools/infer_text_gpu.c $(MODEL_OBJ) src/wubu_tokenizer.o src/bench.o $(CUDA_OBJ) $(GPU_OBJ)
	$(CC) $(CFLAGS) $(CUDA_INC) -o $@ $^ $(LDFLAGS) $(CUDA_LIBS) -L$(CUDA_LIBDIR) -lstdc++

test_cuda_kernels: tools/test_cuda_kernels.c $(CORE_OBJ) $(CUDA_OBJ)
	$(CC) $(CFLAGS) $(CUDA_INC) -o $@ $^ $(LDFLAGS) $(CUDA_LIBS) -L$(CUDA_LIBDIR) -lstdc++

# Compare our logits vs llama.cpp
compare_logits: tools/compare_logits.c $(MODEL_OBJ) src/wubu_tokenizer.o
	g++ -std=c++11 -O2 -I include -I /home/wubu/llama.cpp/include -I /home/wubu/llama.cpp/ggml/include \
		-o $@ $^ \
		-L /home/wubu/llama.cpp/build/bin -lllama -lggml-base -lggml-cpu -lggml \
		-lm -fopenmp -Wl,-rpath,/home/wubu/llama.cpp/build/bin

# Per-layer cos-sim comparison tool
layer_cos_sim: tools/layer_cos_sim.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# Hidden state dumper (standalone, no llama deps)
dump_hidden: tools/dump_hidden.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# Training & tools
train_stub: tools/train_stub.c
	$(CC) -O0 -g -Wall -Wextra -Wno-unused-parameter -I include -fopenmp -o $@ $< -lm -fopenmp

# Inference engines
infer_moe: tools/infer_moe.c $(CORE_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

infer_moe_lazy: tools/infer_moe_lazy.c $(CORE_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

infer_unified: tools/infer_unified.c $(MODEL_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

infer_vision: tools/infer_vision.c $(CORE_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

infer_vision_text: tools/infer_vision_text.c $(MODEL_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_vision_real: tools/test_vision_real.c $(MODEL_OBJ) $(CUDA_OBJ) $(GPU_OBJ)
	$(CXX) $(CFLAGS) -DGPU_SUPPORT -o $@ tools/test_vision_real.c $(MODEL_OBJ) $(CUDA_OBJ) $(GPU_OBJ) $(LDFLAGS) -L$(CUDA_LIBDIR) -lcublas -lcudart
	@echo "test_vision_real built (GPU vision + text)"

infer_vision_text_gpu: tools/infer_vision_text_gpu_nvcc.o $(MODEL_OBJ) $(CUDA_OBJ) src/cuda_vision.o $(GPU_OBJ)
	$(CXX) $(CFLAGS) $(CUDA_INC) -DGPU_SUPPORT -o $@ $^ $(LDFLAGS) $(CUDA_LIBS) -L$(CUDA_LIBDIR) -lcublas -lcudart -lstdc++

tools/infer_vision_text_gpu_nvcc.o: tools/infer_vision_text_gpu.cu include/cuda_vision.h include/wubu_vision.h
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

infer_poincare: tools/infer_poincare.c src/bench.o $(CORE_OBJ) $(CUDA_OBJ)
	$(CC) $(CFLAGS) $(CUDA_INC) -o $@ $^ $(LDFLAGS) $(CUDA_LIBS) -L$(CUDA_LIBDIR) -lstdc++

tailslayer: tools/tailslayer.c $(MODEL_OBJ) src/wubu_tokenizer.o src/bench.o $(CUDA_OBJ)
	$(CC) $(CFLAGS) $(CUDA_INC) -o $@ $^ $(LDFLAGS) $(CUDA_LIBS) -L$(CUDA_LIBDIR) -lstdc++

infer_vision_gpu: tools/infer_vision_gpu.o $(CORE_OBJ) src/cuda_vision.o
	$(CC) $(CFLAGS) $(CUDA_INC) -o $@ $^ $(LDFLAGS) $(CUDA_LIBS) -L$(CUDA_LIBDIR) -lstdc++

tools/infer_vision_gpu.o: tools/infer_vision_gpu.cu include/cuda_vision.h
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

src/cuda_vision.o: src/cuda_vision.cu include/cuda_vision.h include/wubu_vision.h
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

test_256k: tools/test_256k.c $(CORE_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_256k_context: tools/test_256k_context.c $(CORE_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_256k_forward: tools/test_256k_forward.c $(MODEL_OBJ) gen_fixture_safetensors_model
	$(CC) $(CFLAGS) -o $@ tools/test_256k_forward.c $(MODEL_OBJ) $(LDFLAGS)
	./gen_fixture_safetensors_model
	./$@

# Chunked 256K prefill proof: builds the binary; run explicitly (heavy 256K).
test_256k_chunked: tools/test_256k_chunked.c $(MODEL_OBJ) gen_fixture_safetensors_model
	$(CC) $(CFLAGS) -o $@ tools/test_256k_chunked.c $(MODEL_OBJ) $(LDFLAGS)
	./gen_fixture_safetensors_model

test_kv_cache: tools/test_kv_cache.c $(CORE_OBJ) $(CUDA_OBJ)
	$(CC) $(CFLAGS) $(CUDA_INC) -o $@ $^ $(LDFLAGS) $(CUDA_LIBS) -L$(CUDA_LIBDIR) -lstdc++

tokenize_corpus: tools/tokenize_corpus.c src/wubu_tokenizer.o src/gguf_reader.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

train_real: tools/train_real.c $(MODEL_OBJ) src/wubu_tokenizer.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

train_backprop: tools/train_backprop.c $(MODEL_OBJ) src/wubu_tokenizer.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

train_gpu: tools/train_gpu.c src/bench.o $(MODEL_OBJ) $(CUDA_OBJ) src/wubu_tokenizer.o $(GPU_OBJ)
	$(CC) $(CFLAGS) $(CUDA_INC) -o $@ $^ $(LDFLAGS) $(CUDA_LIBS) -L$(CUDA_LIBDIR) -lstdc++

dump_mmproj: tools/dump_mmproj.c src/gguf_reader.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

verify_iq2s: tools/verify_iq2s.c src/gguf_reader.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

check_iq2xxs_stride: tools/check_iq2xxs_stride.c src/gguf_reader.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

verify_dequant: tools/verify_dequant.c src/gguf_reader.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_iq2_dequant: tools/test_iq2_dequant.c src/gguf_reader.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_dequant: tools/test_dequant.c $(MODEL_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

check_forward: tools/check_forward.c $(MODEL_OBJ) src/wubu_tokenizer.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_iq2_xxs_dot: tools/test_iq2_xxs_dot.c src/gguf_reader.o src/dequant_iq2_xxs.o src/wubu_moe.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Test runners
test: test_ssm
	./test_ssm

test_nested_ssm_run: test_nested_ssm
	./test_nested_ssm

test_poincare_gqa_run: test_poincare_gqa
	./test_poincare_gqa

test_tst: tools/test_tst.c $(CORE_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_tst_run: test_tst
	./test_tst

test_gpu_run: test_gpu
	./test_gpu

bench_e2e_run: bench_e2e
	./bench_e2e

train_stub_run: train_stub
	./train_stub

test_regression: tools/test_regression.c $(MODEL_OBJ) src/wubu_tokenizer.o $(CUDA_OBJ)
	$(CC) $(CFLAGS) $(CUDA_INC) -o $@ $^ $(LDFLAGS) $(CUDA_LIBS) -L$(CUDA_LIBDIR) -lstdc++

test_gpu_poincare: tools/test_gpu_poincare.c $(CORE_OBJ) $(CUDA_OBJ) src/bench.o
	$(CC) $(CFLAGS) $(CUDA_INC) -o $@ $^ $(LDFLAGS) $(CUDA_LIBS) -L$(CUDA_LIBDIR) -lstdc++

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

check_weights: tools/check_weights.c $(MODEL_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

check_ssm_a: tools/check_ssm_a.c $(MODEL_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Gemma 4 12B test binaries
test_gemma4: tools/test_gemma4.c $(MODEL_OBJ) src/wubu_gemma4_model.o
	$(CC) $(CFLAGS) -o $@ tools/test_gemma4.c $(MODEL_OBJ) src/wubu_gemma4_model.o $(LDFLAGS)

test_gemma4_gpu: tools/test_gemma4.c $(MODEL_OBJ) src/wubu_gemma4_model.o src/gpu_gemma4.o src/gpu_gemma4_forward.o src/gpu_quant_matmul.o src/gpu_quant_matmul_row_major.o src/cuda_kernels.o
	$(CXX) $(CFLAGS) $(CUDA_INC) -DGPU_SUPPORT -o $@ tools/test_gemma4.c $(MODEL_OBJ) src/wubu_gemma4_model.o src/gpu_gemma4.o src/gpu_gemma4_forward.o src/gpu_quant_matmul.o src/gpu_quant_matmul_row_major.o src/cuda_kernels.o $(LDFLAGS) -L$(CUDA_LIBDIR) -lcublas -lcudart -lstdc++

gen_text_gemma4: tools/gen_text_gemma4.c $(MODEL_OBJ) src/wubu_gemma4_model.o
	$(CC) $(CFLAGS) -o $@ tools/gen_text_gemma4.c $(MODEL_OBJ) src/wubu_gemma4_model.o $(LDFLAGS)

clean:
	rm -f test_nested_ssm test_poincare_ssm test_poincare_gqa load_model test_model test_gpu tokenize_corpus test_moe test_moe_hyperbolic train_real bench_e2e verify_iq2s inspect_iq2s inspect_model train_backprop train_gpu test_gpu_poincare test_rsgd test_backward test_cpu_timing infer_moe infer_moe_lazy infer_unified infer_vision infer_poincare infer_vision_gpu test_256k test_kv_cache test_tst test_nested_moe_router_backward tailslayer test_iq2_dequant test_iq2_xxs_dot test_gemma4 test_gemma4_gpu gen_text_gemma4 src/*.o tools/*.o
