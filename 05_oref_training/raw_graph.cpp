#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {
constexpr int ORDER[7] = {2,4,6,3,5,0,1};
struct Object {
    uint32_t area=0, perimeter=0, boundary=0;
    int xmin, xmax=-1, ymin, ymax=-1, maxd=0, euler4=0;
    double sx=0, sy=0, sxx=0, syy=0, sxy=0, sd=0;
    bool edge=false;
    Object(int h,int w):xmin(w),ymin(h) {}
};
struct Nodes {
    std::vector<float> geometry;
    std::vector<uint8_t> labels, expert;
    std::vector<uint32_t> area;
};
using Edges = std::unordered_map<uint64_t,uint32_t>;
uint32_t find(std::vector<uint32_t>& p,uint32_t x) {
    while(p[x]!=x){p[x]=p[p[x]];x=p[x];} return x;
}
int euler(unsigned b) {
    int n=__builtin_popcount(b);
    return n==1?1:n==3?-1:(b==6||b==9)?2:0;
}
void add(Edges& edges,uint32_t a,uint32_t b,uint32_t amount=1) {
    if(a==b)return;
    if(a>b)std::swap(a,b);
    auto& count=edges[(uint64_t(a)<<32)|b];
    if(count>std::numeric_limits<uint32_t>::max()-amount)throw std::runtime_error("Edge weight overflow");
    count+=amount;
}
std::vector<uint32_t> components(const uint8_t* image,int h,int w,int expert,Nodes& nodes) {
    size_t P=size_t(h)*w;
    std::vector<uint32_t> ids(P), parents(1,0);
    for(int y=0;y<h;++y)for(int x=0;x<w;++x){
        size_t i=size_t(y)*w+x;
        if(image[i]<1||image[i]>24)throw std::runtime_error("Labels outside1..24");
        uint32_t a=x&&image[i-1]==image[i]?find(parents,ids[i-1]):0;
        uint32_t b=y&&image[i-w]==image[i]?find(parents,ids[i-w]):0;
        if(a&&b){parents[std::max(a,b)]=std::min(a,b);ids[i]=std::min(a,b);}
        else if(a||b)ids[i]=a?a:b;
        else {ids[i]=parents.size();parents.push_back(ids[i]);}
    }
    std::vector<uint32_t> dense(parents.size(),0);
    std::vector<Object> objs;objs.emplace_back(h,w);
    std::vector<uint8_t> classes(1,0);
    for(size_t i=0;i<P;++i){
        uint32_t r=find(parents,ids[i]);
        if(!dense[r]){dense[r]=objs.size();objs.emplace_back(h,w);classes.push_back(image[i]);}
        ids[i]=dense[r];
    }
    std::vector<uint32_t>().swap(parents);std::vector<uint32_t>().swap(dense);
    std::vector<uint16_t> distance(P,65535);
    for(int y=0;y<h;++y)for(int x=0;x<w;++x){
        size_t i=size_t(y)*w+x;Object& a=objs[ids[i]];
        ++a.area;a.xmin=std::min(a.xmin,x);a.xmax=std::max(a.xmax,x);
        a.ymin=std::min(a.ymin,y);a.ymax=std::max(a.ymax,y);
        a.sx+=x;a.sy+=y;a.sxx+=double(x)*x;a.syy+=double(y)*y;a.sxy+=double(x)*y;
        int edges=(x==0||image[i-1]!=image[i])+(x==w-1||image[i+1]!=image[i])
                 +(y==0||image[i-w]!=image[i])+(y==h-1||image[i+w]!=image[i]);
        a.perimeter+=edges;if(edges){++a.boundary;distance[i]=1;}
        a.edge|=x==0||y==0||x==w-1||y==h-1;
        if(x)distance[i]=std::min<int>(distance[i],int(distance[i-1])+1);
        if(y)distance[i]=std::min<int>(distance[i],int(distance[i-w])+1);
    }
    for(int y=h-1;y>=0;--y)for(int x=w-1;x>=0;--x){
        size_t i=size_t(y)*w+x;
        if(x+1<w)distance[i]=std::min<int>(distance[i],int(distance[i+1])+1);
        if(y+1<h)distance[i]=std::min<int>(distance[i],int(distance[i+w])+1);
        objs[ids[i]].sd+=distance[i];objs[ids[i]].maxd=std::max<int>(objs[ids[i]].maxd,distance[i]);
    }
    std::vector<uint16_t>().swap(distance);
    for(int y=0;y<=h;++y)for(int x=0;x<=w;++x){
        uint32_t a[4]={y&&x?ids[size_t(y-1)*w+x-1]:0,y&&x<w?ids[size_t(y-1)*w+x]:0,
                       y<h&&x?ids[size_t(y)*w+x-1]:0,y<h&&x<w?ids[size_t(y)*w+x]:0};
        for(int j=0;j<4;++j){
            if(!a[j])continue;bool first=true;
            for(int k=0;k<j;++k)first&=a[k]!=a[j];
            if(!first)continue;unsigned bits=0;
            for(int k=0;k<4;++k)if(a[k]==a[j])bits|=1u<<k;
            objs[a[j]].euler4+=euler(bits);
        }
    }
    uint32_t offset=nodes.area.size();
    for(size_t j=1;j<objs.size();++j){
        const Object& a=objs[j];
        if(a.euler4%4)throw std::runtime_error("Euler characteristic nonintegral");
        int holes=1-a.euler4/4;if(holes<0)throw std::runtime_error("Negative holes");
        double bw=a.xmax-a.xmin+1,bh=a.ymax-a.ymin+1;
        double mx=a.sx/a.area,my=a.sy/a.area;
        double vx=std::max(0.,a.sxx/a.area-mx*mx),vy=std::max(0.,a.syy/a.area-my*my);
        double cov=a.sxy/a.area-mx*my,spread=std::sqrt((vx-vy)*(vx-vy)+4*cov*cov);
        double high=std::max(0.,(vx+vy+spread)/2),low=std::max(0.,(vx+vy-spread)/2);
        std::array<float,12> f={
            float(std::log1p(a.area)/std::log1p(P)),float(bw/w),float(bh/h),
            float(std::tanh(std::log(bw/bh))),float(a.area/(bw*bh)),
            float(std::min(1.,4*std::acos(-1.)*a.area/(double(a.perimeter)*a.perimeter))),
            float(double(a.boundary)/a.area),float(high>1e-12?1-std::min(1.,low/high):0),
            float(std::log1p(a.sd/a.area)/std::log1p(std::min(h,w))),
            float(std::log1p(a.maxd)/std::log1p(std::min(h,w))),
            float(std::log1p(holes)/std::log1p(a.area)),float(a.edge)};
        nodes.geometry.insert(nodes.geometry.end(),f.begin(),f.end());
        nodes.labels.push_back(classes[j]);nodes.expert.push_back(expert);nodes.area.push_back(a.area);
    }
    for(auto& id:ids)id=offset+id-1;
    return ids;
}
std::vector<std::array<uint32_t,3>> sorted_edges(const Edges& source) {
    std::vector<std::array<uint32_t,3>> out;out.reserve(source.size());
    for(auto& kv:source)out.push_back({uint32_t(kv.first>>32),uint32_t(kv.first),kv.second});
    std::sort(out.begin(),out.end());return out;
}
template<class T>void save(std::ofstream& out,const std::vector<T>& a){
    out.write(reinterpret_cast<const char*>(a.data()),a.size()*sizeof(T));
}
}

extern "C" int build_graph(const uint8_t* hard,const int32_t* group,int h,int w,int n,
                          const char* path,char* error,int error_size){
    try{
        if(h<1||w<1||h>60000||w>60000||n<0||uint64_t(h)*w>=uint64_t(INT32_MAX))
            throw std::runtime_error("Invalid dimensions");
        size_t P=size_t(h)*w;Nodes nodes;
        std::vector<std::vector<uint32_t>> maps;
        std::vector<uint32_t> primary(size_t(n)*7,UINT32_MAX),mass(n,0);
        Edges extras;
        for(size_t i=0;i<P;++i){
            if(group[i]<-1||group[i]>=n)throw std::runtime_error("Invalid group");
            if(group[i]>=0)++mass[group[i]];
        }
        for(int e=0;e<7;++e){
            maps.push_back(components(hard+size_t(ORDER[e])*P,h,w,e,nodes));
            const auto& ids=maps.back();
            for(size_t i=0;i<P;++i)if(group[i]>=0){
                uint32_t row=uint32_t(group[i])*7+e;
                auto& first=primary[row];
                if(first==UINT32_MAX)first=ids[i];
                else if(first!=ids[i])++extras[(uint64_t(row)<<32)|ids[i]];
            }
        }
        for(auto v:mass)if(!v)throw std::runtime_error("Empty group");
        Edges contact,overlap,continuation;
        const int dx[4]={1,0,1,-1},dy[4]={0,1,1,1},steps[3]={2,4,8};
        for(int e=0;e<7;++e){
            const auto& ids=maps[e];const uint8_t* labels=hard+size_t(ORDER[e])*P;
            for(int y=0;y<h;++y)for(int x=0;x<w;++x){
                size_t i=size_t(y)*w+x;
                if(x+1<w)add(contact,ids[i],ids[i+1]);
                if(y+1<h)add(contact,ids[i],ids[i+w]);
                // One orientation for each undirected axis; support count is reproducible.
                for(int direction=0;direction<4;++direction){
                    int x1=x+dx[direction],y1=y+dy[direction];
                    if(x1<0||x1>=w||y1>=h||labels[size_t(y1)*w+x1]==labels[i])continue;
                    for(int step:steps){
                        int xx=x+dx[direction]*step,yy=y+dy[direction]*step;
                        if(xx<0||xx>=w||yy>=h)continue;
                        size_t j=size_t(yy)*w+xx;
                        if(labels[i]==labels[j]&&ids[i]!=ids[j])add(continuation,ids[i],ids[j]);
                    }
                }
            }
        }
        for(int a=0;a<7;++a)for(int b=a+1;b<7;++b)
            for(size_t i=0;i<P;++i)add(overlap,maps[a][i],maps[b][i]);
        std::vector<std::vector<uint32_t>>().swap(maps);
        auto ce=sorted_edges(contact),oe=sorted_edges(overlap),re=sorted_edges(continuation),ex=sorted_edges(extras);
        // extra is not a symmetric relation: packed high bits are original group-action row.
        uint64_t header[5]={nodes.area.size(),ce.size(),oe.size(),re.size(),ex.size()};
        std::ofstream out(path,std::ios::binary);
        if(!out)throw std::runtime_error("Cannot open temporary graph output");
        out.write(reinterpret_cast<char*>(header),sizeof(header));
        save(out,nodes.geometry);save(out,nodes.labels);save(out,nodes.expert);save(out,nodes.area);
        save(out,primary);save(out,mass);save(out,ex);save(out,ce);save(out,oe);save(out,re);
        if(!out)throw std::runtime_error("Graph temporary output failed");
        return 0;
    }catch(const std::exception& e){std::snprintf(error,error_size,"%s",e.what());return 1;}
}
