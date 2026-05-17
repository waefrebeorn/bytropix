#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

int main(int argc, char **argv) {
    const char *model_path = argc > 1 ? argv[1] : "/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf";
    const char *tensor_name = argc > 2 ? argv[2] : "blk.0.ffn_gate_exps.weight";
    int expert_id = argc > 3 ? atoi(argv[3]) : 0;
    
    ggml_backend_load_all();
    
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    llama_model *model = llama_model_load_from_file(model_path, mparams);
    if (!model) return 1;
    
    int n_layer = llama_model_n_layer(model);
    int n_embd = llama_model_n_embd(model);
    
    // Access raw GGUF data through llama_model
    // Try to find the tensor via ggml
    fprintf(stderr, "Model: %d layers, %d embd\n", n_layer, n_embd);
    
    // We can't directly get tensor data through the public API.
    // Need to access the raw GGUF context.
    // Let's use a different approach: find the tensor offset in GGUF
    
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 4;
    cparams.n_batch = 1;
    cparams.embeddings = false;
    
    llama_context *ctx = llama_init_from_model(model, cparams);
    if (!ctx) return 1;
    
    // The llama.cpp internal has the GGUF context stored in the model
    // Unfortunately it's not exposed through the public API
    
    fprintf(stderr, "Cannot access raw tensor data through public API\n");
    
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
