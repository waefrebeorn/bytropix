#include "gguf_reader.h"
#include "gpu_quant_matmul.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define QK_K 256

// F32 reference: dequant weight to F32, then plain dot product
static double dot_f32(const float*x, const uint8_t*blk, int n){
    float d,f; memcpy(&d,blk,2); memcpy(&f,blk+2,2);
    auto f2=[](uint16_t h)->float{int s=(h>>15)&1,e=(h>>10)&0x1F,m=h&0x3FF;
        if(e==0)return 0.0f;if(e==31)return s?-INFINITY:INFINITY;
        return ldexpf(1.0f+(float)m/1024.0f,e-15)*(s?-1:1);};
    float dd=f2(*(const uint16_t*)blk),dm=f2(*(const uint16_t*)(blk+2));
    const uint8_t*sc=blk+4,*qh=blk+16,*qs=blk+48;int is=0;double s=0;
    for(int j=0;j<QK_K&&j<n;j+=64){
        uint8_t s1,m1,s2,m2;
        auto gs=[](int jj,const uint8_t*q,uint8_t*d,uint8_t*m){
            if(jj<4){*d=q[jj]&63;*m=q[jj+4]&63;}
            else{*d=(q[jj+4]&0xF)|((q[jj-4]>>6)<<4);*m=(q[jj+4]>>4)|((q[jj]>>6)<<4);}};
        gs(is+0,sc,&s1,&m1);gs(is+1,sc,&s2,&m2);
        float d1=dd*s1,ml1=dm*m1,d2=dd*s2,ml2=dm*m2;int qb=j/2,ci=j/64;
        for(int l=0;l<32&&j+l<n;l++){
            uint8_t lo=qs[qb+l];
            int h1=(qh[l]>>(ci*2))&1,h2=(qh[l]>>(ci*2+1))&1;
            s+=(double)x[j+l]*((double)d1*((lo&0xF)+(h1?16:0))-(double)ml1);
            if(j+32+l<n)s+=(double)x[j+32+l]*((double)d2*((lo>>4)+(h2?16:0))-(double)ml2);
        }is+=2;
    }return s;
}
#define QK_K 256
int main(){
    gguf_ctx*ctx=gguf_open("/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf");gguf_buffer_data(ctx);
    const uint8_t*blob=(const uint8_t*)ctx->data_blob;
    gguf_tensor_info*t=gguf_find_tensor(ctx,"blk.0.attn_qkv.weight");
    int D=2048,C=8192,BPC=(D+QK_K-1)/QK_K,stride=BPC*176;
    float*x=(float*)malloc(D*sizeof(float));
    for(int i=0;i<D;i++)x[i]=((float)rand()/RAND_MAX-0.5f)*2.0f;
    
    cudaStream_t st;cudaStreamCreate(&st);
    int64_t raw=gguf_raw_size(t->ggml_type,(int64_t)D*C);
    uint8_t*d_W;float*d_x,*d_y;
    cudaMalloc((void**)&d_W,(size_t)raw);cudaMalloc((void**)&d_x,D*4);cudaMalloc((void**)&d_y,4);
    cudaMemcpy(d_W,blob+t->data_offset,(size_t)raw,cudaMemcpyHostToDevice);
    cudaMemcpy(d_x,x,D*4,cudaMemcpyHostToDevice);
    
    printf("GPU vs F32-dequant reference (first 8 columns):\n");
    double dt=0,n1=0,n2=0,me=0;int mi=0;
    for(int col=0;col<8;col++){
        // F32 reference: sum 8 blocks
        double f32=0;
        for(int b=0;b<BPC;b++)f32+=dot_f32(x+b*QK_K,blob+t->data_offset+col*stride+b*176, QK_K);
        
        // GPU
        uint8_t*d_col=d_W+(int64_t)col*stride;
        wubu_cuda_quant_matmul(d_x,d_col,t->ggml_type,D,1,d_y,NULL,0,st);
        cudaStreamSynchronize(st);
        float yg;cudaMemcpy(&yg,d_y,4,cudaMemcpyDeviceToHost);
        double de=(double)f32-yg;if(fabs(de)>me){me=fabs(de);mi=col;}
        dt+=f32*yg;n1+=f32*f32;n2+=yg*yg;
        printf("  col%d: f32=%.6f gpu=%.6f diff=%.6f\n",col,(float)f32,yg,(float)de);
    }
    printf("cos-sim=%.8f max_diff=%.6f@col%d\n",dt/(sqrt(n1)*sqrt(n2)),me,mi);
    
    cudaFree(d_W);cudaFree(d_x);cudaFree(d_y);cudaStreamDestroy(st);
    free(x);gguf_close(ctx);return 0;
}
