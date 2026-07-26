/*
 * gen_fixture_btl3_lora.c -- write a tiny BTL-3 LoRA adapter safetensors on
 * top of the fixture base (D=256). Emits __metadata__ with
 * base_model_name_or_path + lora_A/lora_B for q/k/v/o_proj at rank 32 so
 * wubu_model_apply_lora() can load + apply it. This is a SYNTHETIC adapter
 * (random deltas) -- it exercises the two-step orchestration (base load ->
 * delta apply), not a trained BTL-3.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

static float rnd(void) { static unsigned s = 99173; s = s*1664525u+1013904223u; return (float)((int)(s&0xffff)-(int)32768)/32768.0f; }

typedef struct { char name[160]; int shape[3]; int nd; } Tdef;
static Tdef T[256]; static int Tn=0;
static void add(const char*fmt,int l,int d0,int d1){
    Tdef*t=&T[Tn++]; snprintf(t->name,160,fmt,l);
    t->shape[0]=d0;t->shape[1]=d1;t->nd=2;
}

int main(void){
    const int D=256, rank=32, nL=2;
    const int SSMDS=128, SSMKH=16, VD=128;
    const int qh=4, kv=2, hd=4;       /* qh*hd=16 (q/o out), kv*hd=8 (k/v out) */
    const int q_out=qh*hd, kv_out=kv*hd;
    for(int l=0;l<nL;l++){
        add("model.language_model.layers.%d.self_attn.q_proj.lora_A.weight",l,rank,D);
        add("model.language_model.layers.%d.self_attn.q_proj.lora_B.weight",l,q_out,rank);
        add("model.language_model.layers.%d.self_attn.k_proj.lora_A.weight",l,rank,D);
        add("model.language_model.layers.%d.self_attn.k_proj.lora_B.weight",l,kv_out,rank);
        add("model.language_model.layers.%d.self_attn.v_proj.lora_A.weight",l,rank,D);
        add("model.language_model.layers.%d.self_attn.v_proj.lora_B.weight",l,kv_out,rank);
        add("model.language_model.layers.%d.self_attn.o_proj.lora_A.weight",l,rank,D);
        add("model.language_model.layers.%d.self_attn.o_proj.lora_B.weight",l,q_out,rank);
    }

    /* header JSON: _metadata_ (so wubu_adapter_load detects is_lora) + tensors */
    char hdr[8192]; int hp=0;
    hp+=snprintf(hdr+hp,sizeof(hdr)-hp,
        "{\"__metadata__\":{\"base_model_name_or_path\":\"fixture_model.safetensors\"},");
    int base=0;
    for(int i=0;i<Tn;i++){
        int n=T[i].shape[0]*T[i].shape[1];
        hp+=snprintf(hdr+hp,sizeof(hdr)-hp,
            "\"%s\":{\"dtype\":\"F32\",\"shape\":[%d,%d],\"data_offsets\":[%d,%d]}%s",
            T[i].name,T[i].shape[0],T[i].shape[1],base,base+n,(i<Tn-1?",":""));
        base+=n;
    }
    hp+=snprintf(hdr+hp,sizeof(hdr)-hp,"}");
    int hlen=(int)strlen(hdr);
    int raw_off=((8+hlen+7)/8)*8;
    long total=hlen; for(int i=0;i<Tn;i++) total+=(long)T[i].shape[0]*T[i].shape[1]*4;
    FILE*f=fopen("fixture_btl3_lora.safetensors","wb");
    uint64_t off=hlen; fwrite(&off,8,1,f);
    fwrite(hdr,1,hlen,f);
    while((int)ftell(f)<raw_off) fputc(0,f);
    for(int i=0;i<Tn;i++){
        int n=T[i].shape[0]*T[i].shape[1];
        float*buf=malloc((size_t)n*4); for(int j=0;j<n;j++)buf[j]=rnd();
        fwrite(buf,4,n,f); free(buf);
    }
    fclose(f);
    printf("wrote fixture_btl3_lora.safetensors hdr=%d raw_off=%d total=%ld\n",hlen,raw_off,(long)total);
    return 0;
}
