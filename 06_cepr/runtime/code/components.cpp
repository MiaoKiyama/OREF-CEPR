#include <cstdint>
#include <vector>
#include <algorithm>
#include <exception>

extern "C" int label4(const uint8_t* a, uint32_t* out, int h, int w) {
  try {
    std::vector<uint32_t> parent(1, 0);
    std::vector<uint8_t> rank(1, 0);
    auto find = [&](uint32_t x) {
      uint32_t r=x;
      while (parent[r]!=r) r=parent[r];
      while (parent[x]!=x) { auto n=parent[x]; parent[x]=r; x=n; }
      return r;
    };
    for (int y=0;y<h;++y) for (int x=0;x<w;++x) {
      size_t i=size_t(y)*w+x;
      if (a[i]==0) { out[i]=0; continue; }
      uint32_t l=(x && a[i-1]==a[i]) ? out[i-1] : 0;
      uint32_t u=(y && a[i-w]==a[i]) ? out[i-w] : 0;
      if (!l && !u) {
        uint32_t id=uint32_t(parent.size()); parent.push_back(id);rank.push_back(0);out[i]=id;
      } else if (!l || !u) out[i]=l ? l:u;
      else {
        uint32_t p=find(l),q=find(u);
        if (p!=q) {
          if (rank[p]<rank[q]) std::swap(p,q);
          parent[q]=p;
          if (rank[p]==rank[q]) ++rank[p];
        }
        out[i]=p;
      }
    }
    std::vector<uint32_t> compact(parent.size(),0);uint32_t n=0;
    for (size_t i=0;i<size_t(h)*w;++i) if (out[i]) {
      uint32_t p=find(out[i]);
      if (!compact[p]) compact[p]=++n;
      out[i]=compact[p];
    }
    return int(n);
  } catch (const std::exception&) { return -1; }
}

extern "C" void component_stats(const uint8_t* a, const uint32_t* lab,
    uint64_t* area, int32_t* boxes, uint8_t* classes, uint32_t* contacts,
    int h, int w, int n) {
  for (int c=0;c<=n;++c) { boxes[c*4]=h;boxes[c*4+1]=w;boxes[c*4+2]=-1;boxes[c*4+3]=-1; }
  for (int y=0;y<h;++y) for (int x=0;x<w;++x) {
    size_t i=size_t(y)*w+x;uint32_t c=lab[i];
    ++area[c];classes[c]=a[i];
    boxes[c*4]=std::min(boxes[c*4],y);boxes[c*4+1]=std::min(boxes[c*4+1],x);
    boxes[c*4+2]=std::max(boxes[c*4+2],y);boxes[c*4+3]=std::max(boxes[c*4+3],x);
    if (x+1<w && lab[i+1]!=c) {
      ++contacts[size_t(c)*25+a[i+1]];++contacts[size_t(lab[i+1])*25+a[i]];
    }
    if (y+1<h && lab[i+w]!=c) {
      ++contacts[size_t(c)*25+a[i+w]];++contacts[size_t(lab[i+w])*25+a[i]];
    }
  }
}
