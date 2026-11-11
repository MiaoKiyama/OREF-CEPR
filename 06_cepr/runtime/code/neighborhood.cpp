#include <cstdint>
#include <cmath>
#include <cstddef>
extern "C" void alternatives(const uint8_t* pred,const float* conf,const uint8_t* current,
 const bool* eligible,uint8_t* d1,uint8_t* d2,uint8_t* r1,uint8_t* r2,int h,int w) {
 const int radii[4]={2,8,32,128};
 const int dy[8]={-1,0,0,1,-1,-1,1,1},dx[8]={0,-1,1,0,-1,1,-1,1};
 #pragma omp parallel for num_threads(2) schedule(static)
 for(int y=0;y<h;++y)for(int x=0;x<w;++x){
  size_t p=size_t(y)*w+x;d1[p]=d2[p]=r1[p]=r2[p]=0;
  if(!eligible[p])continue;
  float score[25]={};uint8_t votes[25]={},near[25];
  for(int k=0;k<25;++k)near[k]=4;
  for(int r=0;r<4;++r)for(int q=0;q<8;++q){
   int yy=y+dy[q]*radii[r],xx=x+dx[q]*radii[r];
   if(yy<0||yy>=h||xx<0||xx>=w)continue;
   size_t j=size_t(yy)*w+xx;uint8_t c=pred[j];
   score[c]+=conf[j]/float(std::sqrt(double(radii[r])));++votes[c];
   if(near[c]>r)near[c]=r;
  }
  for(int k=0;k<25;++k)if(k==0||k==current[p]||votes[k]<2)score[k]=-1;
  for(int rank=0;rank<2;++rank){
   int best=0;for(int k=1;k<25;++k)if(score[k]>score[best])best=k;
   uint8_t c=score[best]>0?best:0;
   (rank?d2:d1)[p]=c;(rank?r2:r1)[p]=c?near[c]:0;score[best]=-1;
  }
 }
}
