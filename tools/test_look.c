#include "wubu_look.h"
#include <stdio.h>
#include <string.h>
static int fails=0, checks=0;
#define CHECK(c,m) do{checks++; if(!(c)){printf("  [FAIL] %s\n",m);fails++;} else printf("  [PASS] %s\n",m);}while(0)
int main(void){
  wubu_img a={0}, b={0};
  wubu_img_alloc(&a,128,128); wubu_img_fill(&a,30);
  wubu_img_alloc(&b,128,128); wubu_img_fill(&b,30);
  float reuse=0; wubu_look_roi r[8];
  CHECK(wubu_look(NULL,&b,16,32,4,r,&reuse)==0 && reuse==1.0f, "first frame: nothing to look at, full reuse");
  CHECK(wubu_look(&a,&b,16,32,4,r,&reuse)==0 && reuse>0.99f, "identical frames: full reuse");
  for(int y=48;y<80;y++) for(int x=48;x<80;x++) b.data[y*b.stride+x]=((x/2+y/2)&1)?220:30;
  int n=wubu_look(&a,&b,16,32,4,r,&reuse);
  printf("  rois=%d reuse=%.3f first=%d,%d %dx%d energy=%.0f\n",n,reuse,
         n?r[0].x:-1,n?r[0].y:-1,n?r[0].w:0,n?r[0].h:0,n?r[0].energy:0.0);
  CHECK(n>=1, "a changed region produces at least one ROI");
  CHECK(reuse>0.7f && reuse<1.0f, "reuse ratio drops but stays high for a small change");
  int covers=0; if(n) covers = r[0].x<80 && r[0].x+r[0].w>48 && r[0].y<80 && r[0].y+r[0].h>48;
  CHECK(covers, "the ROI overlaps the changed square");
  CHECK(wubu_look(&a,&b,0,32,4,r,&reuse)==-1, "patch 0 -> -1");
  wubu_img_free(&a); wubu_img_free(&b);
  printf("\n=== test_look: %d checks, %d failures ===\n",checks,fails);
  return fails?1:0;
}
