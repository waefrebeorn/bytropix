#include "gguf_reader.h"
#include "gpu_quant_matmul.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define QK_K 256

// Proper F16→F32 matching CPU f16_to_f32
static float f16_to_f32_proper(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x03FF;
    if (exp == 0) {
        uint32_t normal_f32 = (sign << 31) | ((1 + 112) << 23) | (mant << 13);
        float normal_val;
        memcpy(&normal_val, &normal_f32, 4);
        return sign ? normal_val + 6.103515625e-5f : normal_val - 6.103515625e-5f;
    }
    if (exp == 31) {
        uint32_t f32 = (sign << 31) | (0xFF << 23) | (mant << 13);
        float result;
        memcpy(&result, &f32, 4);
        return result;
    }
    uint32_t f32 = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
    float result;
    memcpy(&result, &f32, 4);
    return result;
}

// F32 reference with proper f16 handling
static double dot_f32_proper(const float*x, const uint8_t*blk, int n){
    float dd=f16_to_f32_proper(*(const uint16_t*)blk);
    float dm=f16_to_f32_proper(*(const uint16_t*)(blk+2));
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

// Old flush-to-zero reference (for comparison)
static double dot_f32_fz(const float*x, const uint8_t*blk, int n){
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

int main(){
    gguf_ctx*ctx=gguf_open("/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf");
    gguf_buffer_data(ctx);
    const uint8_t*blob=(const uint8_t*)ctx->data_blob;
    gguf_tensor_info*t=gguf_find_tensor(ctx,"blk.0.attn_qkv.weight");
    int D=2048,C=8192,BPC=(D+QK_K-1)/QK_K,stride=BPC*176;
    float*x=(float*)malloc(D*sizeof(float));
    for(int i=0;i<D;i++)x[i]=((float)rand()/RAND_MAX-0.5f)*2.0f;
    
    cudaStream_t st;cudaStreamCreate(&st);
    int64_t raw=gguf_raw_size(t->ggml_type,(int64_t)D*C);
    uint8_t*d_W;float*d_x,*d_y;
    cudaMalloc((void**)&d_W,(size_t)raw);
    cudaMalloc((void**)&d_x,D*4);
    cudaMalloc((void**)&d_y,4);
    cudaMemcpy(d_W,blob+t->data_offset,(size_t)raw,cudaMemcpyHostToDevice);
    cudaMemcpy(d_x,x,D*4,cudaMemcpyHostToDevice);
    
    printf("GPU vs F32-dequant reference (first 8 columns, blk.0.attn_qkv.weight):\n");
    printf("Column |  Proper F32 (CPU)  |  Flush-to-Zero  |  GPU (patched)  | delta_vs_proper | delta_vs_fz\n");
    printf("-------|--------------------|-----------------|-----------------|-----------------|--------------\n");
    double dt_p=0,n1_p=0,n2_p=0,dt_fz=0,n1_fz=0,n2_fz=0;
    double me_p=0,me_fz=0;int mi_p=0,mi_fz=0;
    for(int col=0;col<8;col++){
        double f32_proper=0, f32_fz=0;
        for(int b=0;b<BPC;b++){
            f32_proper+=dot_f32_proper(x+b*QK_K, blob+t->data_offset+col*stride+b*176, QK_K);
            f32_fz    +=dot_f32_fz(   x+b*QK_K, blob+t->data_offset+col*stride+b*176, QK_K);
        }
        uint8_t*d_col=d_W+(int64_t)col*stride;
        wubu_cuda_quant_matmul(d_x,d_col,t->ggml_type,D,1,d_y,NULL,0,st);
        cudaStreamSynchronize(st);
        float yg;cudaMemcpy(&yg,d_y,4,cudaMemcpyDeviceToHost);
        double dp=fabs(f32_proper-yg), df=fabs(f32_fz-yg);
        if(dp>me_p){me_p=dp;mi_p=col;}
        if(df>me_fz){me_fz=df;mi_fz=col;}
        dt_p+=f32_proper*yg;n1_p+=f32_proper*f32_proper;n2_p+=yg*yg;
        dt_fz+=f32_fz*yg;n1_fz+=f32_fz*f32_fz;n2_fz+=yg*yg;
        printf(" col %d | %18.8f | %15.8f | %15.8f | %15.8f | %13.8f\n",
               col,(float)f32_proper,(float)f32_fz,yg,(float)(f32_proper-yg),(float)(f32_fz-yg));
    }
    printf("\nGPU vs Proper F32:  cos-sim=%.10f max_diff=%.8f@col%d\n",
           dt_p/(sqrt(n1_p)*sqrt(n2_p)+1e-30),me_p,mi_p);
    printf("GPU vs FZ F32:     cos-sim=%.10f max_diff=%.8f@col%d\n",
           dt_fz/(sqrt(n1_fz)*sqrt(n2_fz)+1e-30),me_fz,mi_fz);
    
    // Also test with actual inference: compare dequantized rows
    printf("\n--- Checking first block denormal values ---\n");
    const uint8_t*blk=blob+t->data_offset;
    uint16_t db,dbmin;
    memcpy(&db,blk,2);memcpy(&dbmin,blk+2,2);
    printf("  d=0x%04X (exp=%d) proper=%e fz=0.0\n",db,(db>>10)&0x1F,f16_to_f32_proper(db));
    printf("  dmin=0x%04X (exp=%d) proper=%e fz=0.0\n",dbmin,(dbmin>>10)&0x1F,f16_to_f32_proper(dbmin));
    
    cudaFree(d_W);cudaFree(d_x);cudaFree(d_y);cudaStreamDestroy(st);
    free(x);gguf_close(ctx);return 0;
}
