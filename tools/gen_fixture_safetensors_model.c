/*
 * gen_fixture_safetensors_model.c -- write a tiny F32 safetensors model
 * using the REAL published HF tensor names (model.language_model.layers.N...)
 * at SSM-VALID dims, so test_st_bridge can verify the bridge maps them into
 * wubuwizard's SSM/GQA/MoE F32 forward and it RUNS (no stub).
 *
 * The SSM recurrence is dimension-locked to the invariant SSM_D_STATE=128,
 * SSM_K_HEADS=16 (so KEY_DIM = 128*16 = 2048) and requires VALUE_DIM to be a
 * multiple of 128 (>=1 SSM value head). We therefore use REAL SSM geometry:
 *   D_MODEL=256, VALUE_DIM=128 (1 v-head), KEY_DIM=2048, CONV_DIM=4192,
 *   DT_RANK=32, SSM_D_STATE=128, n_layers=2, small GQA heads, vocab=64.
 * This keeps weights small (~tens of MB) while exercising the actual SSM
 * forward (not a degenerate v_heads=0 path).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

static float rnd(void) { static unsigned s = 12345; s = s*1664525u+1013904223u; return (float)((int)(s&0xffff)-(int)32768)/32768.0f; }

typedef struct { char name[128]; int shape[4]; int nd; } Tdef;
static Tdef T[256]; static int Tn=0;
static void add(const char*fmt,int l,int d0,int d1){
    Tdef*t=&T[Tn++]; snprintf(t->name,128,fmt,l);
    t->shape[0]=d0;t->shape[1]=d1;t->nd=(d1>0?2:1);
}
static void add0(const char*fmt,int d0,int d1){
    Tdef*t=&T[Tn++]; snprintf(t->name,128,"%s",fmt);
    t->shape[0]=d0;t->shape[1]=d1;t->nd=(d1>0?2:1);
}

int main(void){
    const int D=256, dff=512, DT=32, hd=4, kv=2, qh=4;
    const int SSMDS=128, SSMKH=16, VD=128;          /* SSM invariants (real-model geometry) */
    const int KD=SSMDS*SSMKH;                        /* KEY_DIM = 2048 (invariant) */
    const int CONVD=KD*2+VD;                         /* CONV_DIM = 4192 */
    const int CONV_KERNEL=4;                         /* SSM depthwise conv kernel */
    const int nL=2, vocab=64;
    for(int l=0;l<nL;l++){
        add("model.language_model.layers.%d.self_attn.q_proj.weight",l,qh*hd,D);
        add("model.language_model.layers.%d.self_attn.k_proj.weight",l,kv*hd,D);
        add("model.language_model.layers.%d.self_attn.v_proj.weight",l,kv*hd,D);
        add("model.language_model.layers.%d.self_attn.o_proj.weight",l,D,D);
        add("model.language_model.layers.%d.linear_attn.in_proj_qkv.weight",l,CONVD,D);
        add("model.language_model.layers.%d.linear_attn.in_proj_z.weight",l,VD,D);
        add("model.language_model.layers.%d.linear_attn.in_proj_a.weight",l,DT,D);
        add("model.language_model.layers.%d.linear_attn.in_proj_b.weight",l,DT,D);
        add("model.language_model.layers.%d.linear_attn.A_log.weight",l,DT,0);
        add("model.language_model.layers.%d.linear_attn.dt_bias.weight",l,DT,0);
        add("model.language_model.layers.%d.linear_attn.convNd.weight",l,CONV_KERNEL,CONVD);
        add("model.language_model.layers.%d.linear_attn.norm.weight",l,SSMDS,0);
        add("model.language_model.layers.%d.linear_attn.out_proj.weight",l,D,VD);
        add("model.language_model.layers.%d.mlp.gate_proj.weight",l,dff,D);
        add("model.language_model.layers.%d.mlp.up_proj.weight",l,dff,D);
        add("model.language_model.layers.%d.mlp.down_proj.weight",l,D,dff);
        add("model.language_model.layers.%d.input_layernorm.weight",l,D,0);
        add("model.language_model.layers.%d.post_attention_layernorm.weight",l,D,0);
    }
    add0("model.language_model.embed_tokens.weight",vocab,D);
    add0("model.language_model.norm.weight",D,0);
    add0("lm_head.weight",vocab,D);

    /* header JSON */
    char hdr[8192]; int hp=0;
    hp+=snprintf(hdr+hp,sizeof(hdr)-hp,"{");
    int base=0;
    for(int i=0;i<Tn;i++){
        int n=(T[i].nd==2)?T[i].shape[0]*T[i].shape[1]:T[i].shape[0];
        char dim2[16]; dim2[0]=0;
        if(T[i].nd==2) snprintf(dim2,sizeof(dim2),",%d",T[i].shape[1]);
        hp+=snprintf(hdr+hp,sizeof(hdr)-hp,"\"%s\":{\"dtype\":\"F32\",\"shape\":[%d%s],\"data_offsets\":[%d,%d]}%s",
            T[i].name,T[i].shape[0],dim2,base,base+n, (i<Tn-1?",":""));
        base+=n;
    }
    hp+=snprintf(hdr+hp,sizeof(hdr)-hp,"}");
    int hlen=(int)strlen(hdr);
    int raw_off=((8+hlen+7)/8)*8;
    long total=hlen; for(int i=0;i<Tn;i++){int n=(T[i].nd==2)?T[i].shape[0]*T[i].shape[1]:T[i].shape[0]; total+=n*4;}
    FILE*f=fopen("fixture_model.safetensors","wb");
    uint64_t off=hlen; fwrite(&off,8,1,f);
    fwrite(hdr,1,hlen,f);
    while((int)ftell(f)<raw_off) fputc(0,f);
    for(int i=0;i<Tn;i++){
        int n=(T[i].nd==2)?T[i].shape[0]*T[i].shape[1]:T[i].shape[0];
        float*buf=malloc((size_t)n*4); for(int j=0;j<n;j++)buf[j]=rnd();
        fwrite(buf,4,n,f); free(buf);
    }
    fclose(f);
    printf("wrote fixture_model.safetensors hdr=%d raw_off=%d total=%ld\n",hlen,raw_off,(long)total);
    return 0;
}
