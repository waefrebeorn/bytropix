/**
 * infer_text_gpu.c — GPU-accelerated text generation v3
 *
 * Correct pipeline with GPU compute per layer + CPU residual chain.
 * Each layer: upload normed input → GPU GQA/SSM → download → CPU residual+post-norm
 *
 * Build: make infer_text_gpu
 * Usage: ./infer_text_gpu [gguf] [prompt] [max_tok]
 * Env:  VERBOSE=1
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
static volatile int stop=0;
static void handler(int s){(void)s;stop=1;}

// ============ Lazy MoE (expert cache) ============
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
static void moe_expert_fwd_lazy(const float*x,const float*gw,const float*uw,const float*dw,float*tmp,float*out){
    float*go=tmp,*uo=tmp+D_FF,*ao=tmp+D_FF*2;
    for(int j=0;j<D_FF;j++){float s=0;for(int k=0;k<D_MODEL;k++)s+=x[k]*gw[k*D_FF+j];go[j]=s;}
    for(int j=0;j<D_FF;j++){float s=0;for(int k=0;k<D_MODEL;k++)s+=x[k]*uw[k*D_FF+j];uo[j]=s;}
    for(int j=0;j<D_FF;j++){float g=go[j];ao[j]=((g<-80.0f)?0.0f:g/(1.0f+expf(-g)))*uo[j];}
    for(int j=0;j<D_MODEL;j++){float s=0;for(int k=0;k<D_FF;k++)s+=ao[k]*dw[k*D_MODEL+j];out[j]=s;}
}
static void lazy_moe_fwd(const float*x,int B,int T,const uint8_t*qg,const uint8_t*qu,const uint8_t*qd,lm_t*mc,float*out){
    int N=B*T;float scores[256];wubu_moe_router(x,B,T,mc->router,scores);
    int tki[N*N_ACTIVE_EXPTS];float tkw[N*N_ACTIVE_EXPTS];
    for(int s=0;s<N;s++){float*sc=scores+s*N_EXPERTS,mx=sc[0];
        for(int e=1;e<N_EXPERTS;e++)if(sc[e]>mx)mx=sc[e];float se=0;for(int e=0;e<N_EXPERTS;e++)se+=expf(sc[e]-mx);
        float iv=1.0f/(se+1e-30f),sm[256];for(int e=0;e<N_EXPERTS;e++)sm[e]=expf(sc[e]-mx)*iv;
        int*is=tki+s*N_ACTIVE_EXPTS;float*ws=tkw+s*N_ACTIVE_EXPTS;
        for(int k=0;k<N_ACTIVE_EXPTS;k++){int bi=-1;float bv=-1e30f;for(int e=0;e<N_EXPERTS;e++){int u=0;for(int pk=0;pk<k;pk++)if(is[pk]==e){u=1;break;}if(!u&&sm[e]>bv){bv=sm[e];bi=e;}}is[k]=bi;ws[k]=bv;}
        float sw=0;for(int k=0;k<N_ACTIVE_EXPTS;k++)sw+=ws[k];if(sw>1e-30f){float iw=1.0f/sw;for(int k=0;k<N_ACTIVE_EXPTS;k++)ws[k]*=iw;}
    }
    int uid[N_ACTIVE_EXPTS*N],nu=0;for(int s=0;s<N;s++){int*is=tki+s*N_ACTIVE_EXPTS;for(int k=0;k<N_ACTIVE_EXPTS;k++){int e=is[k];if(e<0)continue;int se=0;for(int u=0;u<nu;u++)if(uid[u]==e){se=1;break;}if(!se)uid[nu++]=e;}}
    int ch=(nu!=mc->n);if(!ch){for(int u=0;u<nu;u++)if(uid[u]!=mc->exps[u].eid){ch=1;break;}}
    if(ch){for(int i=0;i<mc->n;i++){free(mc->exps[i].gate);free(mc->exps[i].up);free(mc->exps[i].down);}mc->n=0;
        if(!mc->exps||mc->cap<nu){free(mc->exps);mc->exps=malloc(nu*sizeof(lex_t));mc->cap=nu;}
        for(int u=0;u<nu;u++){int64_t ne=(int64_t)D_MODEL*D_FF,nd=(int64_t)D_FF*D_MODEL;int e=uid[u];mc->exps[u].eid=e;
            mc->exps[u].gate=malloc(ne*4);mc->exps[u].up=malloc(ne*4);mc->exps[u].down=malloc(nd*4);
            gguf_dequantize(qg+(int64_t)e*mc->rs,mc->ty_ge,ne,mc->exps[u].gate);
            gguf_dequantize(qu+(int64_t)e*mc->rs,mc->ty_ge,ne,mc->exps[u].up);
            gguf_dequantize(qd+(int64_t)e*mc->rsd,mc->ty_ge,nd,mc->exps[u].down);}mc->n=nu;
    }
    float*scr=malloc(D_FF*3*sizeof(float));
    for(int s=0;s<N;s++){const float*xs=x+s*D_MODEL;float*os=out+s*D_MODEL;int*is=tki+s*N_ACTIVE_EXPTS;float*ws=tkw+s*N_ACTIVE_EXPTS;
        if(mc->sh_gate){float*sg=scr,*su=scr+D_FF,*sa=scr+D_FF*2;
            for(int j=0;j<SHARED_D_FF;j++){float su2=0;for(int k=0;k<D_MODEL;k++)su2+=xs[k]*mc->sh_gate[k*SHARED_D_FF+j];sg[j]=su2;}
            for(int j=0;j<SHARED_D_FF;j++){float su2=0;for(int k=0;k<D_MODEL;k++)su2+=xs[k]*mc->sh_up[k*SHARED_D_FF+j];su[j]=su2;}
            for(int j=0;j<SHARED_D_FF;j++){float g=sg[j];sa[j]=((g<-80.0f)?0.0f:g/(1.0f+expf(-g)))*su[j];}
            for(int j=0;j<D_MODEL;j++){float su2=0;for(int k=0;k<SHARED_D_FF;k++)su2+=sa[k]*mc->sh_down[k*D_MODEL+j];os[j]=su2;}
        }else memset(os,0,D_MODEL*4);
        for(int kk=0;kk<N_ACTIVE_EXPTS;kk++){int e=is[kk];float wgt=ws[kk];if(e<0||wgt<1e-30f)continue;
            float*gw=find_ex(mc,e,0),*uw=find_ex(mc,e,1),*dw=find_ex(mc,e,2);
            if(!gw||!uw||!dw)continue;float eo[2048];moe_expert_fwd_lazy(xs,gw,uw,dw,scr,eo);
            for(int j=0;j<D_MODEL;j++)os[j]+=wgt*eo[j];}
    }free(scr);
}

int main(int argc,char**argv){
    const char*path=(argc>1&&strlen(argv[1]))?argv[1]:"/home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf";
    const char*prompt=argc>2?argv[2]:"The meaning of life is";
    int max_tok=argc>3?atoi(argv[3]):32;
    int verb=getenv("VERBOSE")?1:0;
    signal(SIGINT,handler);srand(time(NULL));
    printf("=== infer_text_gpu v3 ===\nModel: %s\nPrompt: \"%s\" | max=%d\n",path,prompt,max_tok);
    double T0=now_s();

    // Load
    gguf_ctx*ctx=gguf_open(path);if(!ctx)return 1;gguf_buffer_data(ctx);
    wubu_tokenizer_t tok;if(!wubu_tokenizer_init(&tok,path))return 1;
    wubu_model_t mdl;if(!wubu_model_init(&mdl,path))return 1;
    if(mdl.gguf_ctx)gguf_close(mdl.gguf_ctx);mdl.gguf_ctx=ctx;
    int vs=0;float*embd=NULL;
    {gguf_tensor_info*t=gguf_find_tensor(ctx,"token_embd.weight");
     if(t){int64_t ne=1;for(int i=0;i<t->n_dims;i++)ne*=t->dims[i];vs=(int)(ne/D_MODEL);embd=malloc(ne*4);gguf_read_tensor_f32(ctx,t,embd,ne);}}
    if(!embd)return 1;
    int pids[65536],np=wubu_tokenizer_encode(&tok,prompt,pids,65536);
    if(np<=0)return 1;
    int nL=mdl.n_layers;
    printf("Prompt: %d tok | Layers: %d\n",np,nL);

    // CUDA
    cublasHandle_t ch;cudaStream_t st;
    cublasCreate(&ch);cudaStreamCreate(&st);

    // Upload weights
    printf("Uploading weights...\n");
    gpu_gqa_weights*gqa_w=calloc(nL,sizeof(gpu_gqa_weights));
    gpu_ssm_weights*ssm_w=calloc(nL,sizeof(gpu_ssm_weights));
    for(int l=0;l<nL;l++){
        if(mdl.layers[l].is_ssm){if(!gpu_load_ssm_layer(ctx,l,&ssm_w[l],st))return 1;}
        else{if(!gpu_load_gqa_layer(ctx,l,&gqa_w[l],st))return 1;}
    }

    // SSM states (GPU, persist for decode)
    float**d_ss=calloc(nL,sizeof(float*));
    float**d_cs=calloc(nL,sizeof(float*));
    for(int l=0;l<nL;l++){
        if(!mdl.layers[l].is_ssm)continue;
        d_ss[l]=wubu_cuda_alloc(SSM_V_HEADS*SSM_D_STATE*SSM_D_STATE*sizeof(float));
        d_cs[l]=wubu_cuda_alloc((CONV_KERNEL-1)*CONV_DIM*sizeof(float));
        cudaMemset(d_ss[l],0,SSM_V_HEADS*SSM_D_STATE*SSM_D_STATE*sizeof(float));
        cudaMemset(d_cs[l],0,(CONV_KERNEL-1)*CONV_DIM*sizeof(float));
    }

    // Scratch for max context
    int maxT = np + max_tok + 1;
    if(maxT < 4096) maxT = 4096;

    int qd=GQA_Q_HEADS*GQA_HEAD_DIM,kd=GQA_KV_HEADS*GQA_HEAD_DIM;
    float*d_X=wubu_cuda_alloc((size_t)maxT*D_MODEL*sizeof(float)); // input for current layer
    float*d_Qf=wubu_cuda_alloc((size_t)maxT*qd*2*sizeof(float));
    float*d_Kbuf=wubu_cuda_alloc((size_t)maxT*kd*sizeof(float));
    float*d_Vbuf=wubu_cuda_alloc((size_t)maxT*kd*sizeof(float));
    float*d_Scr=wubu_cuda_alloc((size_t)maxT*qd*sizeof(float));
    float*d_GOut=wubu_cuda_alloc((size_t)maxT*D_MODEL*sizeof(float));

    int qkv_dim=KEY_DIM*2+VALUE_DIM;
    float*d_qkv=wubu_cuda_alloc((size_t)maxT*qkv_dim*sizeof(float));
    float*d_z=wubu_cuda_alloc((size_t)maxT*VALUE_DIM*sizeof(float));
    float*d_beta=wubu_cuda_alloc((size_t)maxT*DT_RANK*sizeof(float));
    float*d_alpha=wubu_cuda_alloc((size_t)maxT*DT_RANK*sizeof(float));
    float*d_bsig=wubu_cuda_alloc((size_t)maxT*DT_RANK*sizeof(float));
    float*d_abi=wubu_cuda_alloc((size_t)maxT*DT_RANK*sizeof(float));
    float*d_gate=wubu_cuda_alloc((size_t)maxT*DT_RANK*sizeof(float));
    float*d_ci=wubu_cuda_alloc((size_t)maxT*CONV_DIM*sizeof(float));
    float*d_co=wubu_cuda_alloc((size_t)maxT*CONV_DIM*sizeof(float));
    float*d_qc=wubu_cuda_alloc((size_t)maxT*KEY_DIM*sizeof(float));
    float*d_kc=wubu_cuda_alloc((size_t)maxT*KEY_DIM*sizeof(float));
    float*d_vc=wubu_cuda_alloc((size_t)maxT*VALUE_DIM*sizeof(float));
    float*d_qn=wubu_cuda_alloc((size_t)maxT*KEY_DIM*sizeof(float));
    float*d_kn=wubu_cuda_alloc((size_t)maxT*KEY_DIM*sizeof(float));
    float*d_del=wubu_cuda_alloc((size_t)maxT*VALUE_DIM*sizeof(float));
    float*d_zs=wubu_cuda_alloc((size_t)maxT*VALUE_DIM*sizeof(float));
    float*d_SOut=wubu_cuda_alloc((size_t)maxT*D_MODEL*sizeof(float));

    // Output buffers
    float*d_ow=NULL,*d_hid=NULL,*d_log=NULL;
    if(mdl.output_weight){
        d_ow=gpu_upload_output_weight(ch,mdl.output_weight,vs,st);
        d_hid=wubu_cuda_alloc(D_MODEL*sizeof(float));
        d_log=wubu_cuda_alloc(vs*sizeof(float));
    }

    cudaStreamSynchronize(st);

    // MoE setup (CPU lazy cache per layer)
    int moe_on = getenv("MOE") ? atoi(getenv("MOE")) : 0;
    lm_t *lm = NULL;
    if (moe_on) {
        lm = calloc(nL, sizeof(lm_t));
        for (int l = 0; l < nL; l++) {
            lm_init(&lm[l]);
            char nm[256];
            snprintf(nm,256,"blk.%d.ffn_gate_inp.weight",l);
            if (!gguf_find_tensor(ctx,nm)) continue;
            lm[l].has = true;
            snprintf(nm,256,"blk.%d.ffn_gate_exps.weight",l);
            gguf_tensor_info *t = gguf_find_tensor(ctx,nm);
            if (!t) { lm[l].has = false; continue; }
            lm[l].ty_ge = t->ggml_type;
            lm[l].rs = gguf_raw_size(t->ggml_type, (int64_t)D_MODEL * D_FF);
            lm[l].rsd = gguf_raw_size(t->ggml_type, (int64_t)D_FF * D_MODEL);
            lm[l].qg = (const uint8_t*)ctx->data_blob + t->data_offset;
            snprintf(nm,256,"blk.%d.ffn_up_exps.weight",l); t = gguf_find_tensor(ctx,nm); if (t) lm[l].qu = (const uint8_t*)ctx->data_blob + t->data_offset;
            snprintf(nm,256,"blk.%d.ffn_down_exps.weight",l); t = gguf_find_tensor(ctx,nm); if (t) lm[l].qd = (const uint8_t*)ctx->data_blob + t->data_offset;
            snprintf(nm,256,"blk.%d.ffn_gate_inp.weight",l); t = gguf_find_tensor(ctx,nm);
            lm[l].ty_gi = t->ggml_type;
            lm[l].router = malloc(D_MODEL * N_EXPERTS * 4);
            gguf_dequantize((const uint8_t*)ctx->data_blob + t->data_offset, t->ggml_type, D_MODEL * N_EXPERTS, lm[l].router);
            snprintf(nm,256,"blk.%d.ffn_gate_shexp.weight",l); t = gguf_find_tensor(ctx,nm);
            if (t) {
                lm[l].ty_gs = t->ggml_type;
                int64_t sn = (int64_t)D_MODEL * SHARED_D_FF, sd = (int64_t)SHARED_D_FF * D_MODEL;
                lm[l].sh_gate = malloc(sn * 4); lm[l].sh_up = malloc(sn * 4); lm[l].sh_down = malloc(sd * 4);
                gguf_dequantize((const uint8_t*)ctx->data_blob + t->data_offset, t->ggml_type, sn, lm[l].sh_gate);
                snprintf(nm,256,"blk.%d.ffn_up_shexp.weight",l); t = gguf_find_tensor(ctx,nm);
                if (t) gguf_dequantize((const uint8_t*)ctx->data_blob + t->data_offset, t->ggml_type, sn, lm[l].sh_up);
                snprintf(nm,256,"blk.%d.ffn_down_shexp.weight",l); t = gguf_find_tensor(ctx,nm);
                if (t) gguf_dequantize((const uint8_t*)ctx->data_blob + t->data_offset, t->ggml_type, sd, lm[l].sh_down);
            }
        }
        printf("MoE setup: %d layers\n", nL);
    }

    printf("GPU init: %.1f s\n", now_s()-T0);

    // Host buffers
    float*h_emb=malloc((size_t)maxT*D_MODEL*4);
    float*h_n=malloc((size_t)maxT*D_MODEL*4);
    float*h_a=malloc((size_t)maxT*D_MODEL*4);
    float*h_m=malloc((size_t)maxT*D_MODEL*4); // MoE output

    // ===== PREFILL =====
    printf("--- Prefill (%d tok) ---\n",np);
    double tp=now_s();
    for(int i=0;i<np;i++)memcpy(h_emb+i*D_MODEL,embd+pids[i]*D_MODEL,D_MODEL*4);
    float*res=h_emb;

    for(int l=0;l<nL;l++){
        double tl=now_s();

        // CPU RMSNorm (O(T×D) = T×2048, fast on CPU for moderate T)
        wubu_rms_norm(1,np,D_MODEL,res,mdl.layers[l].attn_norm_weight,1e-6f,h_n);
        cudaMemcpyAsync(d_X,h_n,np*D_MODEL*4,cudaMemcpyHostToDevice,st);

        if(mdl.layers[l].is_ssm){
            gpu_ssm_forward(ch,st,d_X,1,np,
                ssm_w[l].d_attn_qkv,ssm_w[l].d_attn_gate,
                ssm_w[l].d_ssm_beta,ssm_w[l].d_ssm_alpha,
                ssm_w[l].d_ssm_dt_bias,ssm_w[l].d_ssm_a,
                ssm_w[l].d_ssm_conv1d,ssm_w[l].d_ssm_norm,
                ssm_w[l].d_ssm_out,
                d_ss[l],d_cs[l],d_SOut,
                d_qkv,d_z,d_beta,d_alpha,d_bsig,d_abi,d_gate,
                d_ci,d_co,d_qc,d_kc,d_vc,d_qn,d_kn,d_del,d_zs);
            cudaMemcpyAsync(h_a,d_SOut,np*D_MODEL*4,cudaMemcpyDeviceToHost,st);
        }else{
            gpu_gqa_forward(ch,st,d_X,1,np,
                gqa_w[l].d_attn_q,gqa_w[l].d_attn_k,gqa_w[l].d_attn_v,
                gqa_w[l].d_attn_out_w,gqa_w[l].d_q_norm_w,gqa_w[l].d_k_norm_w,
                d_GOut,d_Qf,d_Kbuf,d_Vbuf,d_Scr);
            cudaMemcpyAsync(h_a,d_GOut,np*D_MODEL*4,cudaMemcpyDeviceToHost,st);
        }
        cudaStreamSynchronize(st);

        // Residual + post-norm on CPU
        for(int i=0;i<np*D_MODEL;i++)res[i]+=h_a[i];
        wubu_rms_norm(1,np,D_MODEL,res,mdl.layers[l].post_attn_norm_weight,1e-6f,h_n);
        // MoE
        if (moe_on && lm[l].has) {
            lazy_moe_fwd(h_n,1,np,lm[l].qg,lm[l].qu,lm[l].qd,&lm[l],h_m);
            for(int i=0;i<np*D_MODEL;i++)res[i]+=h_m[i];
        } else {
            for(int i=0;i<np*D_MODEL;i++)res[i]+=h_n[i];
        }

        if(verb)printf("  L%d: %.1f ms\n",l,(now_s()-tl)*1000);
    }

    // Final norm
    if(mdl.norm_weight){wubu_rms_norm(1,np,D_MODEL,res,mdl.norm_weight,1e-6f,h_n);memcpy(res,h_n,np*D_MODEL*4);}

    // Sample first
    float*logits=malloc(vs*4);
    const float*hl=res+(np-1)*D_MODEL;
    cudaMemcpyAsync(d_hid,hl,D_MODEL*4,cudaMemcpyHostToDevice,st);
    gpu_output_projection(ch,st,d_hid,1,1,d_ow,vs,d_log);
    cudaMemcpyAsync(logits,d_log,vs*4,cudaMemcpyDeviceToHost,st);cudaStreamSynchronize(st);
    int tid=greedy(logits,vs);
    pids[np]=tid;int n_total=np+1;
    char out_buf[1048576];
    int po=wubu_tokenizer_decode(&tok,pids,n_total,out_buf,1048576);
    if(po>0){out_buf[po]=0;printf("%s",out_buf);fflush(stdout);}
    printf("\nPrefill: %.2f s (%.0f tok/s)\n",now_s()-tp,np/(now_s()-tp));

    // ===== DECODE =====
    printf("--- Decode ---\n");
    double td=now_s();int gen=0;

    while(gen<max_tok&&!stop){
        double ts=now_s();

        // Reset SSM states to zero (we process all T tokens from scratch each step)
        for(int l=0;l<nL;l++){
            if(!mdl.layers[l].is_ssm)continue;
            cudaMemset(d_ss[l],0,SSM_V_HEADS*SSM_D_STATE*SSM_D_STATE*sizeof(float));
            cudaMemset(d_cs[l],0,(CONV_KERNEL-1)*CONV_DIM*sizeof(float));
        }

        // Build embedded sequence: previous tokens + new token
        int T = n_total;
        memcpy(h_emb+(T-1)*D_MODEL,embd+pids[T-1]*D_MODEL,D_MODEL*4);
        res = h_emb; // reset residual to embeddings

        for(int l=0;l<nL;l++){
            wubu_rms_norm(1,T,D_MODEL,res,mdl.layers[l].attn_norm_weight,1e-6f,h_n);

            // Upload ALL T tokens
            cudaMemcpyAsync(d_X,h_n,T*D_MODEL*4,cudaMemcpyHostToDevice,st);

            if(mdl.layers[l].is_ssm){
                gpu_ssm_forward(ch,st,d_X,1,T,
                    ssm_w[l].d_attn_qkv,ssm_w[l].d_attn_gate,
                    ssm_w[l].d_ssm_beta,ssm_w[l].d_ssm_alpha,
                    ssm_w[l].d_ssm_dt_bias,ssm_w[l].d_ssm_a,
                    ssm_w[l].d_ssm_conv1d,ssm_w[l].d_ssm_norm,
                    ssm_w[l].d_ssm_out,
                    d_ss[l],d_cs[l],d_SOut,
                    d_qkv,d_z,d_beta,d_alpha,d_bsig,d_abi,d_gate,
                    d_ci,d_co,d_qc,d_kc,d_vc,d_qn,d_kn,d_del,d_zs);
                cudaMemcpyAsync(h_a,d_SOut,T*D_MODEL*4,cudaMemcpyDeviceToHost,st);
            }else{
                gpu_gqa_forward(ch,st,d_X,1,T,
                    gqa_w[l].d_attn_q,gqa_w[l].d_attn_k,gqa_w[l].d_attn_v,
                    gqa_w[l].d_attn_out_w,gqa_w[l].d_q_norm_w,gqa_w[l].d_k_norm_w,
                    d_GOut,d_Qf,d_Kbuf,d_Vbuf,d_Scr);
                cudaMemcpyAsync(h_a,d_GOut,T*D_MODEL*4,cudaMemcpyDeviceToHost,st);
            }
            cudaStreamSynchronize(st);

            // CPU residual + post-norm
            for(int i=0;i<T*D_MODEL;i++)res[i]+=h_a[i];
            wubu_rms_norm(1,T,D_MODEL,res,mdl.layers[l].post_attn_norm_weight,1e-6f,h_n);
            // MoE
            if (moe_on && lm[l].has) {
                lazy_moe_fwd(h_n,1,T,lm[l].qg,lm[l].qu,lm[l].qd,&lm[l],h_m);
                for(int i=0;i<T*D_MODEL;i++)res[i]+=h_m[i];
            } else {
                for(int i=0;i<T*D_MODEL;i++)res[i]+=h_n[i];
            }
        }

        // Final norm
        if(mdl.norm_weight){wubu_rms_norm(1,T,D_MODEL,res,mdl.norm_weight,1e-6f,h_n);memcpy(res,h_n,T*D_MODEL*4);}

        // Output proj for last token
        const float*h_last=res+(T-1)*D_MODEL;
        cudaMemcpyAsync(d_hid,h_last,D_MODEL*4,cudaMemcpyHostToDevice,st);
        gpu_output_projection(ch,st,d_hid,1,1,d_ow,vs,d_log);
        cudaMemcpyAsync(logits,d_log,vs*4,cudaMemcpyDeviceToHost,st);cudaStreamSynchronize(st);

        tid=greedy(logits,vs);
        pids[n_total]=tid;n_total++;
        int no=wubu_tokenizer_decode(&tok,pids,n_total,out_buf,1048576);
        if(no>po){out_buf[no]=0;printf("%s",out_buf+po);fflush(stdout);po=no;}
        gen++;
        if(verb)printf("\n[T+%d: %.0f ms] %d tok",gen,(now_s()-ts)*1000,T);
        if(tid==tok.eos_id&&gen>1)break;
    }
    double dt=now_s()-td;
    printf("\n\n=== Summary ===\n");
    printf("Prefill: %d tok in %.2f s (%.0f tok/s)\n",np,now_s()-tp,np/(now_s()-tp));
    printf("Decode:  %d tok in %.2f s (%.1f tok/s)\n",gen,dt,gen/dt);
    printf("Total:   %.2f s\n",now_s()-T0);

    // Cleanup
    for(int l=0;l<nL;l++){gpu_free_gqa_weights(&gqa_w[l]);gpu_free_ssm_weights(&ssm_w[l]);cudaFree(d_ss[l]);cudaFree(d_cs[l]);}
    free(gqa_w);free(ssm_w);free(d_ss);free(d_cs);
    cudaFree(d_X);cudaFree(d_Qf);cudaFree(d_Kbuf);cudaFree(d_Vbuf);cudaFree(d_Scr);cudaFree(d_GOut);
    cudaFree(d_qkv);cudaFree(d_z);cudaFree(d_beta);cudaFree(d_alpha);
    cudaFree(d_bsig);cudaFree(d_abi);cudaFree(d_gate);
    cudaFree(d_ci);cudaFree(d_co);cudaFree(d_qc);cudaFree(d_kc);cudaFree(d_vc);
    cudaFree(d_qn);cudaFree(d_kn);cudaFree(d_del);cudaFree(d_zs);cudaFree(d_SOut);
    cudaFree(d_ow);cudaFree(d_hid);cudaFree(d_log);
    cublasDestroy(ch);cudaStreamDestroy(st);
    free(h_emb);free(h_n);free(h_a);free(h_m);free(logits);free(embd);
    if (lm) { for(int l=0;l<nL;l++) lm_free(&lm[l]); free(lm); }
    wubu_model_free(&mdl);wubu_tokenizer_free(&tok);
    printf("=== PASS ===\n");return 0;
}
