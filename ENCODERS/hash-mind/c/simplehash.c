/**
 * simplehash.c — Pure C port of SimpleHashV1-V3 (WuBu HashMind encoder)
 * No JAX, no numpy, no Python. Pure C.
 *
 * Architecture:
 *   - Tokenizer: ASCII char → int mapping
 *   - RollingHash: Rabin-Karp sliding window hash
 *   - HashMind: character embedding + hash embedding → dual-source prediction
 *   - Forward pass only (inference), no training (use the GPU training for that)
 *
 * Compile:
 *   gcc -O3 -lm -o simplehash simplehash.c wubu_math.c
 *
 * To add CUDA support later, see ggml-cuda/ pattern.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ─── Constants ─── */
#define VOCAB_SIZE      72     /* printable ASCII subset + specials */
#define D_MODEL         64     /* embedding dimension */
#define N_HEADS          4
#define N_LAYERS         4
#define D_HEAD           (D_MODEL / N_HEADS)  /* 16 */
#define D_FF             (D_MODEL * 4)         /* 256 */
#define CONTEXT_LEN      16
#define HASH_WINDOW       3
#define MODULUS          1000000007U
#define LEARNING_RATE    5e-4f
#define MAX_SEQ          1024

/* ─── ASCII Tokenizer ─── */
typedef struct {
    int char_to_idx[256];     /* char → token id */
    char idx_to_char[VOCAB_SIZE];
    int vocab_size;
} Tokenizer;

void tokenizer_init(Tokenizer* tok) {
    memset(tok->char_to_idx, 0, sizeof(tok->char_to_idx));
    
    /* Digits 0-9 */
    int idx = 0;
    for (char c = '0'; c <= '9'; c++) {
        tok->char_to_idx[(unsigned char)c] = idx;
        tok->idx_to_char[idx] = c;
        idx++;
    }
    /* Uppercase A-Z */
    for (char c = 'A'; c <= 'Z'; c++) {
        tok->char_to_idx[(unsigned char)c] = idx;
        tok->idx_to_char[idx] = c;
        idx++;
    }
    /* Lowercase a-z */
    for (char c = 'a'; c <= 'z'; c++) {
        tok->char_to_idx[(unsigned char)c] = idx;
        tok->idx_to_char[idx] = c;
        idx++;
    }
    /* Special chars */
    const char special[] = {' ', '.', ',', '!', '?', '\n', '\t'};
    for (int i = 0; i < (int)(sizeof(special)/sizeof(special[0])); i++) {
        tok->char_to_idx[(unsigned char)special[i]] = idx;
        tok->idx_to_char[idx] = special[i];
        idx++;
    }
    tok->vocab_size = idx;
}

int tokenizer_encode(Tokenizer* tok, const char* text, int* output, int max_len) {
    int len = 0;
    for (int i = 0; text[i] && len < max_len; i++) {
        int idx = tok->char_to_idx[(unsigned char)text[i]];
        output[len++] = (idx == 0 && text[i] != '0') ? 0 : idx; /* default to 0 */
    }
    return len;
}

/* ─── Rolling Hash ─── */
unsigned int rolling_hash(const int* values, int len, int window_size) {
    if (len < window_size) return 0;
    const unsigned int base = 31;
    unsigned int base_pow = 1;
    for (int i = 0; i < window_size - 1; i++)
        base_pow = (base_pow * base) % MODULUS;
    
    unsigned int hash = 0;
    for (int i = 0; i < window_size; i++)
        hash = (hash * base + (unsigned int)(values[i] % MODULUS)) % MODULUS;
    
    return hash;
}

void rolling_hashes_all(const int* values, int len, int window_size,
                        unsigned int* output, int* out_len) {
    if (len < window_size) { *out_len = 0; return; }
    const unsigned int base = 31;
    unsigned int base_pow = 1;
    for (int i = 0; i < window_size - 1; i++)
        base_pow = (base_pow * base) % MODULUS;
    
    unsigned int hash = 0;
    for (int i = 0; i < window_size; i++)
        hash = (hash * base + (unsigned int)(values[i] % MODULUS)) % MODULUS;
    output[0] = hash;
    
    int count = 1;
    for (int i = 1; i < len - window_size + 1; i++) {
        unsigned int old_val = (unsigned int)(values[i-1] % MODULUS);
        unsigned int new_val = (unsigned int)(values[i + window_size - 1] % MODULUS);
        hash = ((hash + MODULUS - (old_val * base_pow) % MODULUS) * base + new_val) % MODULUS;
        output[count++] = hash;
    }
    *out_len = count;
}

/* ─── Neural Network Ops (forward only, single precision) ─── */
static inline float layer_norm(float x, float mean, float var, float gamma, float beta) {
    return gamma * (x - mean) / sqrtf(var + 1e-5f) + beta;
}

static inline float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

static inline float softmax_partial(float x) {
    return expf(x);
}

/* ─── HashMind Model (forward pass only) ─── */
typedef struct {
    /* Embeddings */
    float token_embed[VOCAB_SIZE][D_MODEL];  /* char embedding table */
    float hash_projector[D_MODEL];            /* hash → embed projection */
    
    /* Transformer blocks */
    struct {
        /* Attention */
        float qkv_proj[D_MODEL][D_MODEL * 3];
        float out_proj[D_MODEL][D_MODEL];
        /* FFN */
        float ffn1[D_MODEL][D_FF];
        float ffn2[D_FF][D_MODEL];
        /* Layer norm */
        float norm1_gamma[D_MODEL], norm1_beta[D_MODEL];
        float norm2_gamma[D_MODEL], norm2_beta[D_MODEL];
    } blocks[N_LAYERS];
    
    /* Output */
    float output_proj[D_MODEL][VOCAB_SIZE];
} HashMind;

void hashmind_init(HashMind* hm) {
    srand(42);
    /* Random init with small scale */
    float scale = 0.02f;
    for (int i = 0; i < VOCAB_SIZE; i++)
        for (int j = 0; j < D_MODEL; j++)
            hm->token_embed[i][j] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale;
    
    for (int j = 0; j < D_MODEL; j++)
        hm->hash_projector[j] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale;
    
    for (int l = 0; l < N_LAYERS; l++) {
        float s2 = sqrtf(2.0f / D_MODEL);
        float s_ff = sqrtf(2.0f / D_MODEL);
        for (int i = 0; i < D_MODEL; i++) {
            for (int j = 0; j < D_MODEL * 3; j++)
                hm->blocks[l].qkv_proj[i][j] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * s2;
            for (int j = 0; j < D_MODEL; j++)
                hm->blocks[l].out_proj[i][j] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * s2;
            for (int j = 0; j < D_FF; j++)
                hm->blocks[l].ffn1[i][j] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * s_ff;
        }
        for (int i = 0; i < D_FF; i++)
            for (int j = 0; j < D_MODEL; j++)
                hm->blocks[l].ffn2[i][j] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f / sqrtf(D_FF);
        
        for (int j = 0; j < D_MODEL; j++) {
            hm->blocks[l].norm1_gamma[j] = 1.0f;
            hm->blocks[l].norm1_beta[j] = 0.0f;
            hm->blocks[l].norm2_gamma[j] = 1.0f;
            hm->blocks[l].norm2_beta[j] = 0.0f;
        }
    }
    
    /* Output projection */
    float s_out = sqrtf(2.0f / D_MODEL);
    for (int i = 0; i < D_MODEL; i++)
        for (int j = 0; j < VOCAB_SIZE; j++)
            hm->output_proj[i][j] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * s_out;
}

/* ─── Forward pass: character indices → logits ─── */
/* Uses dual-source: char_embed + hash_embed */
void hashmind_forward(HashMind* hm,
                      const unsigned int* context_hashes, int num_hashes,
                      const int* context_indices, int context_len,
                      float* logits_out)  /* [VOCAB_SIZE] */
{
    (void)num_hashes; /* unused — context_len derived from hashes */
    float x[D_MODEL];
    float residual[D_MODEL];
    float norm_out[D_MODEL];
    float attn_out[D_MODEL];
    
    /* ─── Dual-source embedding ─── */
    /* Char embedding from token_embed lookup */
    for (int j = 0; j < D_MODEL; j++)
        x[j] = hm->token_embed[context_indices[0]][j];
    
    /* Hash embedding from linear projection */
    float hash_val = (float)context_hashes[0] / (float)MODULUS;
    for (int j = 0; j < D_MODEL; j++)
        x[j] += hash_val * hm->hash_projector[j];
    
    /* ─── Positional Encoding (sinusoidal) ─── */
    for (int j = 0; j < D_MODEL; j += 2) {
        float div = expf(-(float)j * logf(10000.0f) / D_MODEL);
        float pos = (float)context_len;
        x[j] += sinf(pos * div);
        if (j + 1 < D_MODEL)
            x[j + 1] += cosf(pos * div);
    }
    
    /* ─── Transformer blocks ─── */
    for (int l = 0; l < N_LAYERS; l++) {
        /* ── Attention sub-layer ── */
        /* Layer norm */
        float mean = 0.0f, var = 0.0f;
        for (int j = 0; j < D_MODEL; j++) mean += x[j];
        mean /= D_MODEL;
        for (int j = 0; j < D_MODEL; j++) var += (x[j] - mean) * (x[j] - mean);
        var /= D_MODEL;
        
        for (int j = 0; j < D_MODEL; j++)
            norm_out[j] = layer_norm(x[j], mean, var,
                                     hm->blocks[l].norm1_gamma[j],
                                     hm->blocks[l].norm1_beta[j]);
        
        /* QKV projection (single token: batch=1, T=1) */
        float q[D_HEAD], k[D_HEAD], v[D_HEAD];
        for (int h = 0; h < N_HEADS; h++) {
            for (int d = 0; d < D_HEAD; d++) {
                int q_idx = h * D_HEAD + d;
                int k_idx = D_MODEL + h * D_HEAD + d;
                int v_idx = 2 * D_MODEL + h * D_HEAD + d;
                q[d] = 0; k[d] = 0; v[d] = 0;
                for (int j = 0; j < D_MODEL; j++) {
                    q[d] += norm_out[j] * hm->blocks[l].qkv_proj[j][q_idx];
                    k[d] += norm_out[j] * hm->blocks[l].qkv_proj[j][k_idx];
                    v[d] += norm_out[j] * hm->blocks[l].qkv_proj[j][v_idx];
                }
            }
            
            /* Self-attention score (single token → score = 1) */
            float score = 0.0f;
            for (int d = 0; d < D_HEAD; d++)
                score += q[d] * k[d];
            score /= sqrtf((float)D_HEAD);
            float attn_w = expf(score);  /* softmax over single token = 1 */
            
            /* Value aggregation */
            for (int d = 0; d < D_HEAD; d++)
                attn_out[h * D_HEAD + d] = attn_w * v[d];
        }
        
        /* Output projection */
        for (int j = 0; j < D_MODEL; j++) {
            attn_out[j] = 0;
            for (int i = 0; i < D_MODEL; i++)
                attn_out[j] += x[i] * hm->blocks[l].out_proj[i][j];
        }
        
        /* Residual */
        for (int j = 0; j < D_MODEL; j++)
            residual[j] = x[j] + attn_out[j];
        
        /* ── FFN sub-layer ── */
        /* Layer norm */
        mean = 0; var = 0;
        for (int j = 0; j < D_MODEL; j++) mean += residual[j];
        mean /= D_MODEL;
        for (int j = 0; j < D_MODEL; j++) var += (residual[j] - mean) * (residual[j] - mean);
        var /= D_MODEL;
        
        for (int j = 0; j < D_MODEL; j++)
            norm_out[j] = layer_norm(residual[j], mean, var,
                                     hm->blocks[l].norm2_gamma[j],
                                     hm->blocks[l].norm2_beta[j]);
        
        /* FFN: ReLU + linear */
        float ffn_hidden[D_FF];
        for (int j = 0; j < D_FF; j++) {
            ffn_hidden[j] = 0;
            for (int i = 0; i < D_MODEL; i++)
                ffn_hidden[j] += norm_out[i] * hm->blocks[l].ffn1[i][j];
            if (ffn_hidden[j] < 0) ffn_hidden[j] = 0;  /* ReLU */
        }
        
        for (int j = 0; j < D_MODEL; j++) {
            float ffn_out_j = 0;
            for (int i = 0; i < D_FF; i++)
                ffn_out_j += ffn_hidden[i] * hm->blocks[l].ffn2[i][j];
            x[j] = residual[j] + ffn_out_j;
        }
    }
    
    /* ─── Output projection ─── */
    for (int j = 0; j < VOCAB_SIZE; j++) {
        logits_out[j] = 0;
        for (int i = 0; i < D_MODEL; i++)
            logits_out[j] += x[i] * hm->output_proj[i][j];
    }
}

/* ─── Generation ─── */
/* Generate next token given context */
int hashmind_generate(HashMind* hm, const int* indices, int len,
                      const unsigned int* hashes, float temperature) {
    /* Prepare context: last CONTEXT_LEN tokens/hashes */
    int ctx_len = len < CONTEXT_LEN ? len : CONTEXT_LEN;
    
    float logits[VOCAB_SIZE];
    hashmind_forward(hm, hashes + len - ctx_len, ctx_len,
                     indices + len - ctx_len, ctx_len, logits);
    
    /* Temperature sampling */
    if (temperature > 0) {
        float max_logit = logits[0];
        for (int i = 1; i < VOCAB_SIZE; i++)
            if (logits[i] > max_logit) max_logit = logits[i];
        
        float sum = 0;
        float probs[VOCAB_SIZE];
        for (int i = 0; i < VOCAB_SIZE; i++) {
            probs[i] = expf((logits[i] - max_logit) / temperature);
            sum += probs[i];
        }
        
        float r = (float)rand() / RAND_MAX;
        float cum = 0;
        for (int i = 0; i < VOCAB_SIZE; i++) {
            cum += probs[i] / sum;
            if (r <= cum) return i;
        }
        return VOCAB_SIZE - 1;
    } else {
        /* Greedy */
        int best = 0;
        for (int i = 1; i < VOCAB_SIZE; i++)
            if (logits[i] > logits[best]) best = i;
        return best;
    }
}

/* ─── Main ─── */
int main() {
    Tokenizer tok;
    tokenizer_init(&tok);
    
    HashMind hm;
    hashmind_init(&hm);
    
    /* Test text */
    const char* prompt = "Hello World";
    int indices[MAX_SEQ];
    int len = tokenizer_encode(&tok, prompt, indices, MAX_SEQ);
    
    printf("=== SimpleHash (Pure C) — HashMind Inference ===\n");
    printf("Prompt: %s\n", prompt);
    printf("Tokens: ");
    for (int i = 0; i < len; i++) printf("%d ", indices[i]);
    printf("\n");
    
    /* Compute rolling hashes */
    unsigned int hashes[MAX_SEQ];
    int num_hashes;
    rolling_hashes_all(indices, len, HASH_WINDOW, hashes, &num_hashes);
    printf("Hashes: ");
    for (int i = 0; i < num_hashes; i++) printf("%u ", hashes[i]);
    printf("\n\n");
    
    /* Forward pass */
    float logits[VOCAB_SIZE];
    int ctx_start = num_hashes < CONTEXT_LEN ? 0 : num_hashes - CONTEXT_LEN;
    hashmind_forward(&hm, hashes + ctx_start, num_hashes - ctx_start,
                     indices + ctx_start, num_hashes - ctx_start, logits);
    
    printf("Logits (first 10): ");
    for (int i = 0; i < 10 && i < VOCAB_SIZE; i++)
        printf("%.4f ", logits[i]);
    printf("...\n\n");
    
    /* Generate some text */
    srand(time(NULL));
    printf("Generating with temperature=0.8:\n");
    printf("%s", prompt);
    
    int gen_indices[MAX_SEQ];
    memcpy(gen_indices, indices, len * sizeof(int));
    int gen_len = len;
    unsigned int gen_hashes[MAX_SEQ];
    int gen_num_hashes;
    
    for (int step = 0; step < 50; step++) {
        rolling_hashes_all(gen_indices, gen_len, HASH_WINDOW, gen_hashes, &gen_num_hashes);
        int next = hashmind_generate(&hm, gen_indices, gen_len, gen_hashes, 0.8f);
        printf("%c", tok.idx_to_char[next]);
        gen_indices[gen_len++] = next;
        if (gen_len >= MAX_SEQ) break;
    }
    printf("\n");
    
    return 0;
}
