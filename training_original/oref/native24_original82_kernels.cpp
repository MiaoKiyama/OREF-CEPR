#include <algorithm>
#include <cstdint>
#include <vector>
#include <omp.h>

extern "C" void gains_cpu(const double* cm,const double* hist,const double* mass,
    const uint8_t* labels,const uint8_t* current,int64_t n,double* out,int threads) {
    double diag[24],uni[24];
    // Supplied cm summaries are calculated in NumPy to retain its exact reduction order.
    for(int y=0;y<24;++y){diag[y]=cm[y];uni[y]=cm[24+y];}
    #pragma omp parallel for num_threads(threads) schedule(static)
    for(int64_t i=0;i<n;++i){
        int old=labels[i*7+current[i]]-1;double tp=hist[i*24+old];
        double remove=(diag[old]-tp)/std::max(uni[old]-mass[i]+tp,1.0)-diag[old]/std::max(uni[old],1.0);
        for(int a=0;a<7;++a){
            int next=labels[i*7+a]-1;double t=hist[i*24+next];
            double add=(diag[next]+t)/std::max(uni[next]+mass[i]-t,1.0)-diag[next]/std::max(uni[next],1.0);
            out[i*7+a]=next==old?0.0:(remove+add)/24.0;
        }
    }
}

extern "C" void expected_cpu(const double* hist,const uint8_t* labels,const uint8_t* route,
    int64_t n,double* out,int threads){
    #pragma omp parallel for num_threads(threads) schedule(static)
    for(int y=0;y<24;++y){
        double row[24]={0};
        // Each cell sees precisely the same ascending group order as np.bincount.
        for(int64_t i=0;i<n;++i)row[labels[i*7+route[i]]-1]+=hist[i*24+y];
        for(int a=0;a<24;++a)out[y*24+a]=row[a];
    }
}

extern "C" void counts_cpu(const uint16_t* counts,const int64_t* bins,const uint8_t* pred,
    const uint8_t* source,int64_t n,int nb,int64_t* out,int threads){
    int64_t width=static_cast<int64_t>(nb)*25*25;
    std::vector<int64_t> partial(static_cast<size_t>(width)*threads,0);
    #pragma omp parallel num_threads(threads)
    {
        int tid=omp_get_thread_num(),nt=omp_get_num_threads();int64_t* local=partial.data()+tid*width;
        for(int64_t i=n*tid/nt;i<n*(tid+1)/nt;++i){
            int to=pred[i],from=source?source[i]:-1;if(to==from)continue;
            int64_t offset=(bins?bins[i]:0)*625;
            for(int y=0;y<25;++y){
                int value=counts[i*25+y];if(!value)continue;
                local[offset+y*25+to]+=value;
                if(from>=0)local[offset+y*25+from]-=value;
            }
        }
    }
    #pragma omp parallel for num_threads(threads) schedule(static)
    for(int64_t j=0;j<width;++j){int64_t sum=0;for(int t=0;t<threads;++t)sum+=partial[t*width+j];out[j]=sum;}
}
