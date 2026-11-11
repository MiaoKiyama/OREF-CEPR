#include <cstdint>
#include <algorithm>

extern "C" void region_context_wide(const uint8_t* pred, const uint32_t* lab,
 const uint64_t* signature, const uint8_t* previous,
 uint64_t* area, int32_t* boxes, uint8_t* classes, uint32_t* votes,
 uint32_t* perimeter, uint32_t* border, uint64_t* confidence,int h,int w,int n){
 const int radius[4]={8,32,128,512},dy[4]={-1,1,0,0},dx[4]={0,0,-1,1};
 size_t stride=size_t(n+1)*25;
 for(int c=0;c<=n;++c){boxes[c*4]=h;boxes[c*4+1]=w;boxes[c*4+2]=-1;boxes[c*4+3]=-1;}
 for(int y=0;y<h;++y)for(int x=0;x<w;++x){
  size_t i=size_t(y)*w+x;uint32_t c=lab[i];++area[c];classes[c]=pred[i];
  boxes[c*4]=std::min(boxes[c*4],y);boxes[c*4+1]=std::min(boxes[c*4+1],x);
  boxes[c*4+2]=std::max(boxes[c*4+2],y);boxes[c*4+3]=std::max(boxes[c*4+3],x);
  ++votes[5*stride+size_t(c)*25+((signature[i]>>5)&31)];++votes[6*stride+size_t(c)*25+previous[i]];
  confidence[c*2]+=(signature[i]>>19)&15;confidence[c*2+1]+=(signature[i]>>23)&15;
  bool edge=false;
  for(int d=0;d<4;++d){int yy=y+dy[d],xx=x+dx[d];
   if(yy<0||yy>=h||xx<0||xx>=w){++perimeter[c];++border[c];edge=true;continue;}
   size_t j=size_t(yy)*w+xx;if(lab[j]!=c){++votes[size_t(c)*25+pred[j]];++perimeter[c];edge=true;}
  }
  if(!edge)continue;
  for(int r=0;r<4;++r)for(int d=0;d<4;++d){int yy=y+dy[d]*radius[r],xx=x+dx[d]*radius[r];
   if(yy<0||yy>=h||xx<0||xx>=w)continue;size_t j=size_t(yy)*w+xx;
   if(lab[j]!=c)++votes[(r+1)*stride+size_t(c)*25+pred[j]];
  }
 }
}
