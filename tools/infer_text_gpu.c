/**
 * infer_text_gpu.c — GPU-accelerated text generation v5
 *
 * Chunked prefill + persistent KV cache + incremental decode.
 * No full-sequence re-evaluation. Supports 256K context.
 *
 * Build: make infer_text_gpu
 * Usage: ./infer_text_gpu [gguf] [prompt] [max_tok]
 * Env:  MOE=1  VERBOSE=1  CHUNK=512
 */
#include "wubu_model.h"
#include "wubu_ssm.h"
#include "wubu_moe.h"
#include "wubu_tokenizer.h"
#include "gguf_reader.h"
#include "bench.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <signal.h>

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
static int greedy(const float *l, int vs) {
    int b=0;float bv=l[0];for(int i=1;i<vs;i++)if(l[i]>bv){bv=l[i];b=i;}return b;
}
static int sample(float *logits, int vs, float temperature, int top_k, float top_p) {
    if (temperature <= 0.0f) return greedy(logits, vs);
    float *probs = (float *)malloc(vs * 4);
    float max_l = logits[0];
    for (int i = 1; i < vs; i++) if (logits[i] > max_l) max_l = logits[i];
    for (int i = 0; i < vs; i++) probs[i] = (logits[i] - max_l) / temperature;
    if (top_k > 0 && top_k < vs) {
        float *vals = (float *)malloc(vs * 4);
        memcpy(vals, probs, vs * 4);
        for (int i = 0; i < top_k; i++) {
            int best = i;
            for (int j = i+1; j < vs; j++) if (vals[j] > vals[best]) best = j;
            float t = vals[i]; vals[i] = vals[best]; vals[best] = t;
        }
        float thr = vals[top_k - 1]; free(vals);
        for (int i = 0; i < vs; i++) if (probs[i] < thr) probs[i] = -1e30f;
    }
    max_l = probs[0];
    for (int i = 1; i < vs; i++) if (probs[i] > max_l) max_l = probs[i];
    float sum = 0.0f;
    for (int i = 0; i < vs; i++) { probs[i] = expf(probs[i] - max_l); sum += probs[i]; }
    float inv = 1.0f / (sum + 1e-30f);
    for (int i = 0; i < vs; i++) probs[i] *= inv;
    if (top_p > 0.0f && top_p < 1.0f) {
        typedef struct { float p; int i; } pt;
        pt *ps = (pt *)malloc(vs * sizeof(pt));
        for (int i = 0; i < vs; i++) { ps[i].p = probs[i]; ps[i].i = i; }
        for (int i = 0; i < vs; i++) {
            int b = i;
            for (int j = i+1; j < vs; j++) if (ps[j].p > ps[b].p) b = j;
            pt t = ps[i]; ps[i] = ps[b]; ps[b] = t;
        }
        float cum = 0.0f; int cut = vs;
        for (int i = 0; i < vs; i++) { if (ps[i].p <= 0.0f) break; cum += ps[i].p; if (cum >= top_p) { cut = i+1; break; } }
        for (int i = cut; i < vs; i++) probs[ps[i].i] = 0.0f;
        free(ps);
        sum = 0.0f; for (int i = 0; i < vs; i++) sum += probs[i];
        if (sum > 1e-30f) { inv = 1.0f / sum; for (int i = 0; i < vs; i++) probs[i] *= inv; }
    }
    float r = (float)rand() / RAND_MAX; float cum = 0.0f;
    for (int i = 0; i < vs; i++) { cum += probs[i]; if (r <= cum) { free(probs); return i; } }
    free(probs); return vs - 1;
}
static volatile int stop=0;
static void handler(int s){(void)s;stop=1;}

// --- Persistent KV cache for one GQA layer ---
typedef struct {
    float *d_k;  // [maxT, kv_dim] on GPU
    float *d_v;  // [maxT, kv_dim] on GPU
    int cap;
} kv_cache_t;

static kv_cache_t *kv_cache_init(int nL, int maxT) {
    int kv_dim = GQA_KV_HEADS * GQA_HEAD_DIM;
    kv_cache_t *c = calloc(nL, sizeof(kv_cache_t));
    for (int l = 0; l < nL; l++) {
        c[l].d_k = wubu_cuda_alloc((size_t)maxT * kv_dim * 4);
        c[l].d_v = wubu_cuda_alloc((size_t)maxT * kv_dim * 4);
        c[l].cap = maxT;
    }
    return c;
}
static void kv_cache_free(kv_cache_t *c, int nL) {
    for (int l = 0; l < nL; l++) { cudaFree(c[l].d_k); cudaFree(c[l].d_v); }
    free(c);
}

// --- Lazy MoE (unchanged) ---
typedef struct { int eid; float *gate,*up,*down; } lex_t;
typedef struct {
    lex_t *exps; int n,cap;
    float *sh_gate,*sh_up,*sh_down,*router;
    const uint8_t *qg,*qu,*qd;
    int64_t rs,rsd; int ty_ge,ty_gi,ty_gs; bool has;
} lm_t;
static void lm_init(lm_t *m) { memset(m,0,sizeof(*m)); }
static void lm_free(lm_t *m) {
    for(int i=0;i<m->n;i++){free(m->exps[i].gate);free(m->exps[i].up);free(m->exps[i].down);}
    free(m->exps);free(m->sh_gate);free(m->sh_up);free(m->sh_down);free(m->router);memset(m,0,sizeof(*m));
}
static float* find_ex(lm_t *m, int eid, int w) {
    for(int i=0;i<m->n;i++) if(m->exps[i].eid==eid)
        return w==0?m->exps[i].gate:w==1?m->exps[i].up:m->exps[i].down;
    return NULL;
}
static void moe_exp_fwd(const float*x,const float*gw,const float*uw,const float*dw,float*tmp,float*out){
    float*go=tmp,*uo=tmp+D_FF,*ao=tmp+D_FF*2;
    for(int j=0;j<D_FF;j++){float s=0;for(int k=0;k<D_MODEL;k++)s+=x[k]*gw[k+j*D_MODEL];go[j]=s;}
    for(int j=0;j<D_FF;j++){float s=0;for(int k=0;k<D_MODEL;k++)s+=x[k]*uw[k+j*D_MODEL];uo[j]=s;}
    for(int j=0;j<D_FF;j++){float g=go[j];ao[j]=((g<-80.0f)?0.0f:g/(1.0f+expf(-g)))*uo[j];}
    for(int j=0;j<D_MODEL;j++){float s=0;for(int k=0;k<D_FF;k++)s+=ao[k]*dw[k+j*D_FF];out[j]=s;}
}
static void moe_fwd_1tok(const float*x,lm_t*mc,float*out){
    float scores[256]; wubu_moe_router(x,1,1,mc->router,scores);
    float sv[256]; for(int e=0;e<N_EXPERTS;e++)sv[e]=1.0f/(1.0f+expf(-scores[e]));
    int tki[8]; float tkw[8];
    for(int k=0;k<N_ACTIVE_EXPTS;k++){int bi=-1;float bv=-1e30f;
        for(int e=0;e<N_EXPERTS;e++){int u=0;for(int pk=0;pk<k;pk++)if(tki[pk]==e){u=1;break;}if(!u&&sv[e]>bv){bv=sv[e];bi=e;}}
        tki[k]=bi; tkw[k]=bv;}
    float sw=0; for(int k=0;k<N_ACTIVE_EXPTS;k++)sw+=tkw[k];
    if(sw>1e-30f){float iw=1.0f/sw;for(int k=0;k<N_ACTIVE_EXPTS;k++)tkw[k]*=iw;}
    int uid[8],nu=0;
    for(int k=0;k<N_ACTIVE_EXPTS;k++){int e=tki[k];if(e<0)continue;int u=0;for(int pk=0;pk<k;pk++)if(uid[pk]==e){u=1;break;}if(!u)uid[nu++]=e;}
    int ch=(nu!=mc->n); if(!ch){for(int u=0;u<nu;u++)if(uid[u]!=mc->exps[u].eid){ch=1;break;}}
    if(ch){
        for(int i=0;i<mc->n;i++){free(mc->exps[i].gate);free(mc->exps[i].up);free(mc->exps[i].down);}mc->n=0;
        if(!mc->exps||mc->cap<nu){free(mc->exps);mc->exps=malloc(nu*sizeof(lex_t));mc->cap=nu;}
        for(int u=0;u<nu;u++){int64_t ne=(int64_t)D_MODEL*D_FF,nd=(int64_t)D_FF*D_MODEL;int e=uid[u];
            mc->exps[u].eid=e;mc->exps[u].gate=malloc(ne*4);mc->exps[u].up=malloc(ne*4);mc->exps[u].down=malloc(nd*4);
            gguf_dequantize(mc->qg+(int64_t)e*mc->rs,mc->ty_ge,ne,mc->exps[u].gate);
            gguf_dequantize(mc->qu+(int64_t)e*mc->rs,mc->ty_ge,ne,mc->exps[u].up);
            gguf_dequantize(mc->qd+(int64_t)e*mc->rsd,mc->ty_ge,nd,mc->exps[u].down);}mc->n=nu;
    }
    float*scr=malloc(D_FF*3*4);
    if(mc->sh_gate){
        float*sg=scr,*su=scr+D_FF,*sa=scr+D_FF*2;
        for(int j=0;j<SHARED_D_FF;j++){float s2=0;for(int k=0;k<D_MODEL;k++)s2+=x[k]*mc->sh_gate[k+j*D_MODEL];sg[j]=s2;}
        for(int j=0;j<SHARED_D_FF;j++){float s2=0;for(int k=0;k<D_MODEL;k++)s2+=x[k]*mc->sh_up[k+j*D_MODEL];su[j]=s2;}
        for(int j=0;j<SHARED_D_FF;j++){float g=sg[j];sa[j]=((g<-80.0f)?0.0f:g/(1.0f+expf(-g)))*su[j];}
        for(int j=0;j<D_MODEL;j++){float s2=0;for(int k=0;k<SHARED_D_FF;k++)s2+=sa[k]*mc->sh_down[k+j*SHARED_D_FF];out[j]=s2;}
    }else memset(out,0,D_MODEL*4);
    for(int kk=0;kk<N_ACTIVE_EXPTS;kk++){int e=tki[kk];float wgt=tkw[kk];if(e<0||wgt<1e-30f)continue;
        float*gw=find_ex(mc,e,0),*uw=find_ex(mc,e,1),*dw=find_ex(mc,e,2);
        if(!gw||!uw||!dw)continue;float eo[2048];moe_exp_fwd(x,gw,uw,dw,scr,eo);
        for(int j=0;j<D_MODEL;j++)out[j]+=wgt*eo[j];}
    free(scr);
}

int main(int argc,char**argv){
    const char*path=(argc>1&&strlen(argv[1]))?argv[1]:"/home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf";
    const char*prompt=argc>2?argv[2]:"The meaning of life is";
    int max_tok=argc>3?atoi(argv[3]):32;
    int verb=getenv("VERBOSE")?1:0;
    int moe_on=getenv("MOE")?atoi(getenv("MOE")):0;
    int chunk_sz=getenv("CHUNK")?atoi(getenv("CHUNK")):256;
    signal(SIGINT,handler);srand(time(NULL));
    float temperature = getenv("TEMP") ? atof(getenv("TEMP")) : 1.0f;
    int samp_top_k = getenv("TOP_K") ? atoi(getenv("TOP_K")) : 20;
    float top_p = getenv("TOP_P") ? atof(getenv("TOP_P")) : 0.95f;
    printf("=== infer_text_gpu v5 (KV cache + chunked prefill) ===\nModel: %s\nPrompt: \"%s\" | max=%d | MOE=%d | CHUNK=%d\n",path,prompt,max_tok,moe_on,chunk_sz);
    printf("Sampling: temp=%.1f top_k=%d top_p=%.2f\n",temperature,samp_top_k,top_p);
    double T0=now_s();

    // Load model
    gguf_ctx*ctx=gguf_open(path);if(!ctx)return 1;gguf_buffer_data(ctx);
    wubu_tokenizer_t tok;if(!wubu_tokenizer_init(&tok,path))return 1;
    wubu_model_t mdl;if(!wubu_model_init(&mdl,path))return 1;
    if(mdl.gguf_ctx)gguf_close(mdl.gguf_ctx);mdl.gguf_ctx=ctx;
    int vs=0;float*embd=NULL;
    {gguf_tensor_info*t=gguf_find_tensor(ctx,"token_embd.weight");
     if(t){int64_t ne=1;for(int i=0;i<t->n_dims;i++)ne*=t->dims[i];vs=(int)(ne/D_MODEL);embd=malloc(ne*4);gguf_read_tensor_f32(ctx,t,embd,ne);}}
    if(!embd)return 1;
    int pids[262144],np=wubu_tokenizer_encode(&tok,prompt,pids+1,262143);
    if(np<=0)return 1;
    // Prepend BOS token (required by Qwen)
    pids[0]=tok.bos_id; np++;
    int nL=mdl.n_layers;
    int maxT=np+max_tok+1; if(maxT<4096)maxT=4096; if(maxT>262144)maxT=262144;
    printf("Prompt: %d tok | Layers: %d | maxT=%d\n",np,nL,maxT);

    // CUDA init
    cublasHandle_t ch;cudaStream_t st;
    cublasCreate(&ch);cudaStreamCreate(&st);
    printf("Uploading weights...\n");
    gpu_gqa_weights*gqa_w=calloc(nL,sizeof(gpu_gqa_weights));
    gpu_ssm_weights*ssm_w=calloc(nL,sizeof(gpu_ssm_weights));
    for(int l=0;l<nL;l++){
        if(mdl.layers[l].is_ssm){if(!gpu_load_ssm_layer(ctx,l,&ssm_w[l],st))return 1;}
        else{if(!gpu_load_gqa_layer(ctx,l,&gqa_w[l],st))return 1;}
    }

    // SSM persistent states
    float**d_ss=calloc(nL,sizeof(float*));float**d_cs=calloc(nL,sizeof(float*));
    for(int l=0;l<nL;l++){if(!mdl.layers[l].is_ssm)continue;
        d_ss[l]=wubu_cuda_alloc((size_t)SSM_V_HEADS*SSM_D_STATE*SSM_D_STATE*4);
        d_cs[l]=wubu_cuda_alloc((size_t)(CONV_KERNEL-1)*CONV_DIM*4);
        cudaMemset(d_ss[l],0,(size_t)SSM_V_HEADS*SSM_D_STATE*SSM_D_STATE*4);
        cudaMemset(d_cs[l],0,(size_t)(CONV_KERNEL-1)*CONV_DIM*4);}

    // GPU scratch buffers (sized for one chunk)
    int qd=GQA_Q_HEADS*GQA_HEAD_DIM,kd=GQA_KV_HEADS*GQA_HEAD_DIM;
    float*d_X=wubu_cuda_alloc((size_t)chunk_sz*D_MODEL*4);
    float*d_Scr=wubu_cuda_alloc((size_t)chunk_sz*qd*2*4); // fused Q+gate
    float*d_Ktmp=wubu_cuda_alloc((size_t)chunk_sz*kd*4);
    float*d_Vtmp=wubu_cuda_alloc((size_t)chunk_sz*kd*4);
    float*d_GOut=wubu_cuda_alloc((size_t)chunk_sz*D_MODEL*4);

    // SSM scratch
    int qkdim=KEY_DIM*2+VALUE_DIM;
    float*d_qkv=wubu_cuda_alloc((size_t)chunk_sz*qkdim*4);
    float*d_z=wubu_cuda_alloc((size_t)chunk_sz*VALUE_DIM*4);
    float*d_beta=wubu_cuda_alloc((size_t)chunk_sz*DT_RANK*4);
    float*d_alpha=wubu_cuda_alloc((size_t)chunk_sz*DT_RANK*4);
    float*d_bsig=wubu_cuda_alloc((size_t)chunk_sz*DT_RANK*4);
    float*d_abi=wubu_cuda_alloc((size_t)chunk_sz*DT_RANK*4);
    float*d_gate=wubu_cuda_alloc((size_t)chunk_sz*DT_RANK*4);
    float*d_ci=wubu_cuda_alloc((size_t)chunk_sz*CONV_DIM*4);
    float*d_co=wubu_cuda_alloc((size_t)chunk_sz*CONV_DIM*4);
    float*d_qc=wubu_cuda_alloc((size_t)chunk_sz*KEY_DIM*4);
    float*d_kc=wubu_cuda_alloc((size_t)chunk_sz*KEY_DIM*4);
    float*d_vc=wubu_cuda_alloc((size_t)chunk_sz*VALUE_DIM*4);
    float*d_qn=wubu_cuda_alloc((size_t)chunk_sz*KEY_DIM*4);
    float*d_kn=wubu_cuda_alloc((size_t)chunk_sz*KEY_DIM*4);
    float*d_del=wubu_cuda_alloc((size_t)chunk_sz*VALUE_DIM*4);
    float*d_zs=wubu_cuda_alloc((size_t)chunk_sz*VALUE_DIM*4);
    float*d_SOut=wubu_cuda_alloc((size_t)chunk_sz*D_MODEL*4);

    // Output projection
    float*d_ow=NULL,*d_hid=NULL,*d_log=NULL;
    if(mdl.output_weight){
        d_ow=gpu_upload_output_weight(ch,mdl.output_weight,vs,st);
        d_hid=wubu_cuda_alloc(D_MODEL*4);
        d_log=wubu_cuda_alloc(vs*4);
    }

    // RoPE sin/cos table (maxT)
    int ro_dim = ROTARY_DIM;
    float *h_sc = malloc((size_t)maxT * ro_dim * sizeof(float));
    for (int p = 0; p < maxT; p++) {
        for (int i = 0; i < ro_dim / 2; i++) {
            float theta = powf(ROPE_THETA, -2.0f * i / ro_dim);
            float angle = (float)p * theta * 0.25f; // 4× extrapolation
            h_sc[p * ro_dim + i * 2]     = cosf(angle);
            h_sc[p * ro_dim + i * 2 + 1] = sinf(angle);
        }
    }
    float *d_sc = wubu_cuda_alloc((size_t)maxT * ro_dim * sizeof(float));
    cudaMemcpyAsync(d_sc, h_sc, (size_t)maxT * ro_dim * sizeof(float), cudaMemcpyHostToDevice, st);
    free(h_sc);
    cudaStreamSynchronize(st);

    // Persistent KV cache for GQA layers
    kv_cache_t *kvc = kv_cache_init(nL, maxT);

    // Score scratch for chunked attention (tiled: only [C, ATTEN_TILE] + M/L)
    size_t score_bytes = wubu_cuda_chunked_attn_query_scratch(chunk_sz, maxT);
    float *d_score_scratch = wubu_cuda_alloc(score_bytes);

    // MoE setup
    lm_t*lm=NULL;if(moe_on){
        lm=calloc(nL,sizeof(lm_t));
        for(int l=0;l<nL;l++){lm_init(&lm[l]);char nm[256];snprintf(nm,256,"blk.%d.ffn_gate_inp.weight",l);
            if(!gguf_find_tensor(ctx,nm))continue;lm[l].has=true;
            snprintf(nm,256,"blk.%d.ffn_gate_exps.weight",l);gguf_tensor_info*t=gguf_find_tensor(ctx,nm);
            if(!t){lm[l].has=false;continue;}lm[l].ty_ge=t->ggml_type;
            lm[l].rs=gguf_raw_size(t->ggml_type,(int64_t)D_MODEL*D_FF);
            lm[l].rsd=gguf_raw_size(t->ggml_type,(int64_t)D_FF*D_MODEL);
            lm[l].qg=(const uint8_t*)ctx->data_blob+t->data_offset;
            snprintf(nm,256,"blk.%d.ffn_up_exps.weight",l);t=gguf_find_tensor(ctx,nm);if(t)lm[l].qu=(const uint8_t*)ctx->data_blob+t->data_offset;
            snprintf(nm,256,"blk.%d.ffn_down_exps.weight",l);t=gguf_find_tensor(ctx,nm);if(t)lm[l].qd=(const uint8_t*)ctx->data_blob+t->data_offset;
            snprintf(nm,256,"blk.%d.ffn_gate_inp.weight",l);t=gguf_find_tensor(ctx,nm);lm[l].ty_gi=t->ggml_type;
            lm[l].router=malloc((size_t)D_MODEL*N_EXPERTS*4);gguf_dequantize((const uint8_t*)ctx->data_blob+t->data_offset,t->ggml_type,(int64_t)D_MODEL*N_EXPERTS,lm[l].router);
            snprintf(nm,256,"blk.%d.ffn_gate_shexp.weight",l);t=gguf_find_tensor(ctx,nm);
            if(t){lm[l].ty_gs=t->ggml_type;int64_t sn=(int64_t)D_MODEL*SHARED_D_FF,sd=(int64_t)SHARED_D_FF*D_MODEL;
                lm[l].sh_gate=malloc((size_t)sn*4);lm[l].sh_up=malloc((size_t)sn*4);lm[l].sh_down=malloc((size_t)sd*4);
                gguf_dequantize((const uint8_t*)ctx->data_blob+t->data_offset,t->ggml_type,sn,lm[l].sh_gate);
                snprintf(nm,256,"blk.%d.ffn_up_shexp.weight",l);t=gguf_find_tensor(ctx,nm);if(t)gguf_dequantize((const uint8_t*)ctx->data_blob+t->data_offset,t->ggml_type,sn,lm[l].sh_up);
                snprintf(nm,256,"blk.%d.ffn_down_shexp.weight",l);t=gguf_find_tensor(ctx,nm);if(t)gguf_dequantize((const uint8_t*)ctx->data_blob+t->data_offset,t->ggml_type,sd,lm[l].sh_down);}}
        printf("MoE setup: %d layers\n",nL);}
    // GPU MoE scratch buffers
    float *d_moe_in = NULL, *d_moe_out = NULL, *d_moe_gw = NULL;
    float *d_moe_uw = NULL, *d_moe_silu = NULL, *d_moe_dnw = NULL, *d_moe_contrib = NULL;
    if (moe_on) {
        size_t mw = (size_t)D_MODEL * D_FF; // 67M floats per weight matrix
        d_moe_in  = wubu_cuda_alloc(D_MODEL * 4);
        d_moe_out = wubu_cuda_alloc(D_MODEL * 4);
        d_moe_gw  = wubu_cuda_alloc(mw * 4);
        d_moe_uw  = wubu_cuda_alloc(mw * 4);
        d_moe_silu = wubu_cuda_alloc(D_FF * 4);
        d_moe_dnw = wubu_cuda_alloc((size_t)D_FF * D_MODEL * 4);
        d_moe_contrib = wubu_cuda_alloc(D_MODEL * 4);
        printf("GPU MoE buffers: %zu MB\n", (mw*3 + (size_t)D_FF + D_MODEL*2) * 4 / 1048576);
    }
    printf("GPU init: %.1f s\n",now_s()-T0);

    // Host buffers
    float*h_x =malloc((size_t)chunk_sz*D_MODEL*4);
    float*h_n =malloc((size_t)chunk_sz*D_MODEL*4);
    float*h_a =malloc((size_t)chunk_sz*D_MODEL*4);
    float*h_mo=malloc((size_t)chunk_sz*D_MODEL*4);
    float*h_Xtmp=malloc((size_t)maxT*D_MODEL*4); // residual stream for all tokens

    // Copy prompt embeddings to residual
    for(int i=0;i<np;i++) memcpy(h_Xtmp+i*D_MODEL,embd+pids[i]*D_MODEL,D_MODEL*4);

    // ===== CHUNKED PREFILL =====
    printf("--- Chunked prefill (%d tok, chunk=%d) ---\n",np,chunk_sz);
    double tp=now_s();
    int cache_pos = 0; // running position in KV cache

    int n_chunks = (np + chunk_sz - 1) / chunk_sz;
    for (int ch = 0; ch < n_chunks; ch++) {
        int c_start = ch * chunk_sz;
        int c_end = c_start + chunk_sz;
        if (c_end > np) c_end = np;
        int C = c_end - c_start;
        if (C <= 0) break;

        // Copy chunk embeddings from residual
        memcpy(h_x, h_Xtmp + c_start * D_MODEL, (size_t)C * D_MODEL * 4);

        for (int l = 0; l < nL; l++) {
            // RMS norm over chunk
            wubu_rms_norm(1, C, D_MODEL, h_x, mdl.layers[l].attn_norm_weight, 1e-6f, h_n);

            if (mdl.layers[l].is_ssm) {
                // SSM: process chunk (state persists in d_ss[l], d_cs[l])
                cudaMemcpyAsync(d_X, h_n, (size_t)C*D_MODEL*4, cudaMemcpyHostToDevice, st);
                gpu_ssm_forward(ch, st, d_X, 1, C,
                    ssm_w[l].d_attn_qkv, ssm_w[l].d_attn_gate,
                    ssm_w[l].d_ssm_beta, ssm_w[l].d_ssm_alpha,
                    ssm_w[l].d_ssm_dt_bias, ssm_w[l].d_ssm_a,
                    ssm_w[l].d_ssm_conv1d, ssm_w[l].d_ssm_norm, ssm_w[l].d_ssm_out,
                    d_ss[l], d_cs[l], d_SOut,
                    d_qkv, d_z, d_beta, d_alpha, d_bsig, d_abi, d_gate,
                    d_ci, d_co, d_qc, d_kc, d_vc, d_qn, d_kn, d_del, d_zs);
                cudaMemcpyAsync(h_a, d_SOut, (size_t)C*D_MODEL*4, cudaMemcpyDeviceToHost, st);
                cudaStreamSynchronize(st);
            } else {
                // GQA: QKV projection → RMSNorm → RoPE → cache → attention
                int T_cache = cache_pos; // cached tokens before this chunk

                // Upload chunk normed to GPU
                cudaMemcpyAsync(d_X, h_n, (size_t)C*D_MODEL*4, cudaMemcpyHostToDevice, st);
                cudaStreamSynchronize(st);

                // Q (fused Q+gate), K, V projections
                wubu_cuda_matmul(ch, d_X, C, D_MODEL, gqa_w[l].d_attn_q, qd*2, d_Scr, 1.0f, 0.0f);
                wubu_cuda_matmul(ch, d_X, C, D_MODEL, gqa_w[l].d_attn_k, kd, d_Ktmp, 1.0f, 0.0f);
                wubu_cuda_matmul(ch, d_X, C, D_MODEL, gqa_w[l].d_attn_v, kd, d_Vtmp, 1.0f, 0.0f);
                cudaStreamSynchronize(st);

                // RMSNorm Q per head (in-place on d_Scr — Q is first qd elements per row)
                // d_Scr layout: [C, qd*2], Q is at offset 0, gate at offset qd
                // RMSNorm operates on [C * n_q, hd]
                wubu_cuda_rms_norm_heads(C * GQA_Q_HEADS, GQA_HEAD_DIM,
                    d_Scr, gqa_w[l].d_q_norm_w, 1e-6f, d_Scr, st);

                // RMSNorm K per head (in-place on d_Ktmp)
                wubu_cuda_rms_norm_heads(C * GQA_KV_HEADS, GQA_HEAD_DIM,
                    d_Ktmp, gqa_w[l].d_k_norm_w, 1e-6f, d_Ktmp, st);
                cudaStreamSynchronize(st);

                // Apply RoPE to Q and K (in-place on d_Scr for Q, d_Ktmp for K)
                // Q layout: [C, n_q*hd], K layout: [C, n_kv*hd]
                wubu_cuda_apply_rotary_to_qk(d_Scr, d_Ktmp,
                    C, C, GQA_Q_HEADS, GQA_KV_HEADS, GQA_HEAD_DIM,
                    d_sc + (size_t)c_start * ROTARY_DIM, st);
                cudaStreamSynchronize(st);

                // Append K, V to persistent cache at position cache_pos
                int kv_dim = GQA_KV_HEADS * GQA_HEAD_DIM;
                cudaMemcpyAsync(kvc[l].d_k + (size_t)cache_pos * kv_dim, d_Ktmp,
                    (size_t)C * kv_dim * sizeof(float), cudaMemcpyDeviceToDevice, st);
                cudaMemcpyAsync(kvc[l].d_v + (size_t)cache_pos * kv_dim, d_Vtmp,
                    (size_t)C * kv_dim * sizeof(float), cudaMemcpyDeviceToDevice, st);
                cudaStreamSynchronize(st);

                // Chunked attention: Q against all cached K,V (T_cache + C total)
                // Gate is second half of d_Scr: d_Scr + qd
                wubu_cuda_chunked_attn(ch, st, C, T_cache + C,
                    d_Scr,                    // Q (RMSNorm'd + RoPE'd)
                    kvc[l].d_k,               // K_cache
                    kvc[l].d_v,               // V_cache
                    d_Scr + qd,               // gate (raw, pre-sigmoid)
                    gqa_w[l].d_attn_out_w,    // output projection weight
                    d_GOut,                   // output [C, D_MODEL]
                    d_score_scratch);
                cudaMemcpyAsync(h_a, d_GOut, (size_t)C*D_MODEL*4, cudaMemcpyDeviceToHost, st);
                cudaStreamSynchronize(st);
            }

            // Add attention output to residual (in h_x)
            for (int i = 0; i < C * D_MODEL; i++) h_x[i] += h_a[i];

            // Post-attention RMSNorm
            wubu_rms_norm(1, C, D_MODEL, h_x, mdl.layers[l].post_attn_norm_weight, 1e-6f, h_n);

            // MoE for chunk tokens
            if (moe_on && lm[l].has) {
                for (int i = 0; i < C; i++)
                    moe_fwd_1tok(h_n + i*D_MODEL, &lm[l], h_mo + i*D_MODEL);
                for (int i = 0; i < C; i++) {
                    float *src = h_mo + i * D_MODEL;
                    float *dst = h_x + i * D_MODEL;
                    for (int j = 0; j < D_MODEL; j++) dst[j] += src[j];
                }
            } else {
                for (int i = 0; i < C; i++) {
                    float *src = h_n + i * D_MODEL;
                    float *dst = h_x + i * D_MODEL;
                    for (int j = 0; j < D_MODEL; j++) dst[j] += src[j];
                }
            }
        } // layers

        // Save chunk residual back to global residual buffer
        memcpy(h_Xtmp + c_start * D_MODEL, h_x, (size_t)C * D_MODEL * 4);
        cache_pos = c_end; // advance cache position
        if (verb) printf("chunk %d/%d (%d tok, cache=%d)\n", ch+1, n_chunks, C, cache_pos);
    }

    // Final normalization
    if(mdl.norm_weight){
        wubu_rms_norm(1, np, D_MODEL, h_Xtmp, mdl.norm_weight, 1e-6f, h_n);
        memcpy(h_Xtmp, h_n, (size_t)np * D_MODEL * 4);
    }

    // First token output
    float*logits=malloc((size_t)vs*4);
    cudaMemcpyAsync(d_hid, h_Xtmp+(np-1)*D_MODEL, D_MODEL*4, cudaMemcpyHostToDevice, st);
    gpu_output_projection(ch,st,d_hid,1,1,d_ow,vs,d_log);
    cudaMemcpyAsync(logits,d_log,(size_t)vs*4,cudaMemcpyDeviceToHost,st);cudaStreamSynchronize(st);
    int tid=greedy(logits,vs);pids[np]=tid;int n_total=np+1;
    char out_buf[262144];int po=wubu_tokenizer_decode(&tok,pids,n_total,out_buf,262144);
    if(po>0){out_buf[po]=0;printf("%s",out_buf);fflush(stdout);}
    printf("\nPrefill: %.2f s (%.0f tok/s)\n",now_s()-tp,np/(now_s()-tp));

    // ===== INCREMENTAL DECODE (with KV cache) =====
    printf("--- Decode (incremental) ---\n");
    double td=now_s();int gen=0;

    while(gen<max_tok&&!stop){
        double ts=now_s();
        int T = n_total; // current sequence length (total tokens so far)
        int pos = T - 1; // position of new token being processed

        // Embed new token
        memcpy(h_Xtmp + pos * D_MODEL, embd + pids[pos] * D_MODEL, D_MODEL * 4);
        memcpy(h_x, h_Xtmp + pos * D_MODEL, D_MODEL * 4);

        for (int l = 0; l < nL; l++) {
            // RMS norm single token
            wubu_rms_norm(1, 1, D_MODEL, h_x, mdl.layers[l].attn_norm_weight, 1e-6f, h_n);

            if (mdl.layers[l].is_ssm) {
                // SSM: process 1 token (state persists)
                cudaMemcpyAsync(d_X, h_n, D_MODEL*4, cudaMemcpyHostToDevice, st);
                gpu_ssm_forward(ch, st, d_X, 1, 1,
                    ssm_w[l].d_attn_qkv, ssm_w[l].d_attn_gate,
                    ssm_w[l].d_ssm_beta, ssm_w[l].d_ssm_alpha,
                    ssm_w[l].d_ssm_dt_bias, ssm_w[l].d_ssm_a,
                    ssm_w[l].d_ssm_conv1d, ssm_w[l].d_ssm_norm, ssm_w[l].d_ssm_out,
                    d_ss[l], d_cs[l], d_SOut,
                    d_qkv, d_z, d_beta, d_alpha, d_bsig, d_abi, d_gate,
                    d_ci, d_co, d_qc, d_kc, d_vc, d_qn, d_kn, d_del, d_zs);
                cudaMemcpyAsync(h_a, d_SOut, D_MODEL*4, cudaMemcpyDeviceToHost, st);
                cudaStreamSynchronize(st);
            } else {
                // GQA: QKV proj for new token, append to cache, attention
                cudaMemcpyAsync(d_X, h_n, D_MODEL*4, cudaMemcpyHostToDevice, st);
                cudaStreamSynchronize(st);

                // QKV projections
                wubu_cuda_matmul(ch, d_X, 1, D_MODEL, gqa_w[l].d_attn_q, qd*2, d_Scr, 1.0f, 0.0f);
                wubu_cuda_matmul(ch, d_X, 1, D_MODEL, gqa_w[l].d_attn_k, kd, d_Ktmp, 1.0f, 0.0f);
                wubu_cuda_matmul(ch, d_X, 1, D_MODEL, gqa_w[l].d_attn_v, kd, d_Vtmp, 1.0f, 0.0f);
                cudaStreamSynchronize(st);

                // RMSNorm Q per head
                wubu_cuda_rms_norm_heads(1 * GQA_Q_HEADS, GQA_HEAD_DIM,
                    d_Scr, gqa_w[l].d_q_norm_w, 1e-6f, d_Scr, st);
                // RMSNorm K per head
                wubu_cuda_rms_norm_heads(1 * GQA_KV_HEADS, GQA_HEAD_DIM,
                    d_Ktmp, gqa_w[l].d_k_norm_w, 1e-6f, d_Ktmp, st);
                cudaStreamSynchronize(st);

                // RoPE for Q and K at position pos
                wubu_cuda_apply_rotary_to_qk(d_Scr, d_Ktmp,
                    1, 1, GQA_Q_HEADS, GQA_KV_HEADS, GQA_HEAD_DIM,
                    d_sc + (size_t)pos * ROTARY_DIM, st);
                cudaStreamSynchronize(st);

                // Append K, V to cache at position pos
                int kv_dim = GQA_KV_HEADS * GQA_HEAD_DIM;
                cudaMemcpyAsync(kvc[l].d_k + (size_t)pos * kv_dim, d_Ktmp,
                    (size_t)kv_dim * sizeof(float), cudaMemcpyDeviceToDevice, st);
                cudaMemcpyAsync(kvc[l].d_v + (size_t)pos * kv_dim, d_Vtmp,
                    (size_t)kv_dim * sizeof(float), cudaMemcpyDeviceToDevice, st);
                cudaStreamSynchronize(st);

                // Chunked attention: C=1 query against all T cached tokens
                wubu_cuda_chunked_attn(ch, st, 1, T,
                    d_Scr,            // Q (RMSNorm'd + RoPE'd)
                    kvc[l].d_k,       // K_cache
                    kvc[l].d_v,       // V_cache
                    d_Scr + qd,       // gate (raw)
                    gqa_w[l].d_attn_out_w,
                    d_GOut,
                    d_score_scratch);
                cudaMemcpyAsync(h_a, d_GOut, D_MODEL*4, cudaMemcpyDeviceToHost, st);
                cudaStreamSynchronize(st);
            }

            // Add attention to residual (single token)
            for (int i = 0; i < D_MODEL; i++) h_x[i] += h_a[i];

            // Post-attention RMSNorm
            wubu_rms_norm(1, 1, D_MODEL, h_x, mdl.layers[l].post_attn_norm_weight, 1e-6f, h_n);

            // MoE for this token (GPU)
            if (moe_on && lm[l].has) {
                // Build expert pointer arrays from lazy cache
                float *gw_ptrs[8], *uw_ptrs[8], *dw_ptrs[8];
                int eids[8]; float ewgts[8];
                // Use existing routing logic — re-run moe_fwd_1tok's routing section
                float scores[256]; wubu_moe_router(h_n,1,1,lm[l].router,scores);
                float sv[256]; for(int e=0;e<N_EXPERTS;e++)sv[e]=1.0f/(1.0f+expf(-scores[e]));
                int n_active=0;
                for(int k=0;k<N_ACTIVE_EXPTS;k++){
                    int bi=-1;float bv=-1e30f;
                    for(int e=0;e<N_EXPERTS;e++){int u=0;for(int pk=0;pk<k;pk++)if(eids[pk]==e){u=1;break;}if(!u&&sv[e]>bv){bv=sv[e];bi=e;}}
                    eids[n_active]=bi; ewgts[n_active]=bv; n_active++;
                }
                float sw=0; for(int k=0;k<n_active;k++)sw+=ewgts[k];
                if(sw>1e-30f){float iw=1.0f/sw;for(int k=0;k<n_active;k++)ewgts[k]*=iw;}

                // Check cache + get pointers
                int uid[8],nu=0;
                for(int k=0;k<n_active;k++){int e=eids[k];if(e<0)continue;int seen=0;for(int u=0;u<nu;u++)if(uid[u]==e){seen=1;break;}if(!seen)uid[nu++]=e;}
                int cache_ch=(nu!=lm[l].n); if(!cache_ch){for(int u=0;u<nu;u++)if(uid[u]!=lm[l].exps[u].eid){cache_ch=1;break;}}
                if(cache_ch){
                    for(int i=0;i<lm[l].n;i++){free(lm[l].exps[i].gate);free(lm[l].exps[i].up);free(lm[l].exps[i].down);}lm[l].n=0;
                    if(!lm[l].exps||lm[l].cap<nu){free(lm[l].exps);lm[l].exps=malloc((size_t)nu*sizeof(lex_t));lm[l].cap=nu;}
                    for(int u=0;u<nu;u++){int64_t ne=(int64_t)D_MODEL*D_FF,nd=(int64_t)D_FF*D_MODEL;int e=uid[u];
                        lm[l].exps[u].eid=e;lm[l].exps[u].gate=malloc((size_t)ne*4);lm[l].exps[u].up=malloc((size_t)ne*4);lm[l].exps[u].down=malloc((size_t)nd*4);
                        gguf_dequantize(lm[l].qg+(int64_t)e*lm[l].rs,lm[l].ty_ge,ne,lm[l].exps[u].gate);
                        gguf_dequantize(lm[l].qu+(int64_t)e*lm[l].rs,lm[l].ty_ge,ne,lm[l].exps[u].up);
                        gguf_dequantize(lm[l].qd+(int64_t)e*lm[l].rsd,lm[l].ty_ge,nd,lm[l].exps[u].down);}lm[l].n=nu;
                }
                // Build pointer arrays
                for(int k=0;k<n_active;k++){
                    gw_ptrs[k]=find_ex(&lm[l],eids[k],0);
                    uw_ptrs[k]=find_ex(&lm[l],eids[k],1);
                    dw_ptrs[k]=find_ex(&lm[l],eids[k],2);
                }

                // Upload input to GPU + run GPU MoE
                cudaMemcpyAsync(d_moe_in, h_n, D_MODEL*4, cudaMemcpyHostToDevice, st);
                cudaStreamSynchronize(st);
                wubu_cuda_moe_fwd_1tok(ch, st, d_moe_in,
                    (const float**)gw_ptrs, (const float**)uw_ptrs, (const float**)dw_ptrs,
                    eids, ewgts, n_active,
                    d_moe_out, d_moe_gw, d_moe_uw, d_moe_silu, d_moe_dnw, d_moe_contrib);
                cudaMemcpyAsync(h_mo, d_moe_out, D_MODEL*4, cudaMemcpyDeviceToHost, st);
                cudaStreamSynchronize(st);
                for (int j = 0; j < D_MODEL; j++) h_x[j] += h_mo[j];
            } else {
                for (int j = 0; j < D_MODEL; j++) h_x[j] += h_n[j];
            }

            // Save to global residual
            memcpy(h_Xtmp + pos * D_MODEL, h_x, D_MODEL * 4);
        }

        // Final normalization + output projection
        if(mdl.norm_weight){
            wubu_rms_norm(1, 1, D_MODEL, h_x, mdl.norm_weight, 1e-6f, h_n);
            memcpy(h_x, h_n, D_MODEL * 4);
        }
        cudaMemcpyAsync(d_hid, h_x, D_MODEL*4, cudaMemcpyHostToDevice, st);
        gpu_output_projection(ch,st,d_hid,1,1,d_ow,vs,d_log);
        cudaMemcpyAsync(logits,d_log,(size_t)vs*4,cudaMemcpyDeviceToHost,st);cudaStreamSynchronize(st);
        tid=sample(logits,vs,temperature,samp_top_k,top_p);pids[n_total]=tid;n_total++;
        int no=wubu_tokenizer_decode(&tok,pids,n_total,out_buf,262144);
        if(no>po){out_buf[no]=0;printf("%s",out_buf+po);fflush(stdout);po=no;}
        gen++;
        if(verb)printf("\n[T+%d: %.0f ms] T=%d",gen,(now_s()-ts)*1000,T);
        if(tid==tok.eos_id&&gen>2)break;
    }
    double dt=now_s()-td;
    printf("\n\n=== Summary ===\n");
    printf("Prefill: %d tok in %.2f s (%.0f tok/s)\n",np,now_s()-tp,np/(now_s()-tp));
    printf("Decode:  %d tok in %.2f s (%.1f tok/s)\n",gen,dt,gen/dt);
    printf("Total:   %.2f s\n",now_s()-T0);

    // Cleanup
    for(int l=0;l<nL;l++){gpu_free_gqa_weights(&gqa_w[l]);gpu_free_ssm_weights(&ssm_w[l]);cudaFree(d_ss[l]);cudaFree(d_cs[l]);}
    free(gqa_w);free(ssm_w);free(d_ss);free(d_cs);
    cudaFree(d_X);cudaFree(d_Scr);cudaFree(d_Ktmp);cudaFree(d_Vtmp);cudaFree(d_GOut);
    cudaFree(d_qkv);cudaFree(d_z);cudaFree(d_beta);cudaFree(d_alpha);cudaFree(d_bsig);cudaFree(d_abi);cudaFree(d_gate);
    cudaFree(d_ci);cudaFree(d_co);cudaFree(d_qc);cudaFree(d_kc);cudaFree(d_vc);cudaFree(d_qn);cudaFree(d_kn);cudaFree(d_del);cudaFree(d_zs);
    cudaFree(d_SOut);
    cudaFree(d_sc);cudaFree(d_score_scratch);
    cudaFree(d_ow);cudaFree(d_hid);cudaFree(d_log);
    if(moe_on){cudaFree(d_moe_in);cudaFree(d_moe_out);cudaFree(d_moe_gw);cudaFree(d_moe_uw);cudaFree(d_moe_silu);cudaFree(d_moe_dnw);cudaFree(d_moe_contrib);}
    kv_cache_free(kvc, nL);
    cublasDestroy(ch);cudaStreamDestroy(st);
    free(h_x);free(h_n);free(h_a);free(h_mo);free(h_Xtmp);free(logits);free(embd);
    if(lm){for(int l=0;l<nL;l++)lm_free(&lm[l]);free(lm);}
    wubu_model_free(&mdl);wubu_tokenizer_free(&tok);
    printf("=== PASS ===\n");return 0;
}
