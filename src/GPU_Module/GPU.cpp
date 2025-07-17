//u ovom fajlu ce biti i map i reduce i combiner namenjeni za gpu, tj k mer algoritam
#include <filesystem>
#include <CL/sycl.hpp>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <tuple>
#include <cstdlib> 
int constexpr MAXK=4;
namespace sycl = cl::sycl;
struct Mapped{
  char word[MAXK+1];
  int v;
   Mapped operator+(const Mapped& other) const {
        if (strcmp(word, other.word) != 0) {
            throw std::invalid_argument("Keys do not match for reduction!");
        }
        Mapped m;
        strcpy(m.word, word);
        m.v = v + other.v;
        return m;
    }
};
struct Reduced{
  char word[MAXK+1];
  int v;
      Reduced operator+(const Reduced& other) const {
        if (strcmp(word, other.word) != 0) {
            throw std::invalid_argument("Keys do not match for reduction!");
        }
        Reduced r;
        strcpy(r.word, word);
        r.v = v + other.v;
        return r;
          }
};
//usm
struct Map {
  char** data;
  std::size_t N;
sycl::queue& q;
int k;        
Map(char** _data, std::size_t _N, sycl::queue& _q,int _k)
    : data(_data), N(_N), q(_q), k(_k) {}
Mapped* mappedw = sycl::malloc_shared<Mapped>(N-k+1, q);
  void operator()(sycl::nd_item<1> it)  {
  /*  std::size_t gid = it.get_global_id(0);
       const char* src = data[gid];       
std::size_t n = strlen(src);      
//char* dest = mappedw[gid].word;            

for (std::size_t start = 0; start <= n - k; ++start) {
    for (std::size_t i = 0; i < k; ++i){
        mappedw[start].word[i] = src[start + i];
                  mappedw[start].v = 1;
*/

   std::size_t gid = it.get_global_id(0);
    std::size_t start = it.get_global_id(1);

        const char* src = data[gid];
        std::size_t n = strlen(src);

        if (start <= n - k) {
            for (std::size_t i = 0; i < k; ++i) {
                mappedw[start].word[i] = src[start + i];
            }
            mappedw[start].v = 1;
            mappedw[start].word[k]='\0';
        }
  }
 //   staviti null terminaciju posle

   
  

  

  void runkernel() const {
         


    std::size_t local_size  = 256;
    std::size_t global_size = ((N + local_size - 1) / local_size) * local_size;
    sycl::nd_range<1> ndr{ {global_size}, {local_size} };

    q.submit([&](sycl::handler& h) {
      h.parallel_for<Map>(
        ndr,
        *this             
      );
    }).wait();
  }
  ~Map() {
    //sycl::free(data, q);
    //delete[] mappedw;
    sycl::free(mappedw, q);

  }
};

//dodati sort fazu, ali to moze i kod cpu-a , tj radix i std sort odvojeno benchmarking, kakogod, to kad se bude implementirao cpu deo

struct Reduce{
     char** data;
    std::size_t N;
    sycl::queue& q;
    std::size_t k;

    Reduce(char** _data, std::size_t _N, sycl::queue& _q, std::size_t _k)
        : data(_data), N(_N), q(_q), k(_k) {}

void operator()(sycl::nd_item<1> it,
                    sycl::local_accessor<int, 1> shared,
                    int* result) const {

        constexpr int BLOCK_SIZE = 512;

        std::size_t gid = it.get_global_id(0);
        std::size_t lid = it.get_local_id(0);

        // compute string length or 0 if out of bounds
        int v = 0;
        if (gid < N) {
            const char* src = data[gid];
            v = static_cast<int>(strlen(src));
        }

        shared[lid] = v;

        it.barrier(sycl::access::fence_space::local_space);

        // Tree reduction in local memory
        for (std::size_t s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
            if (lid < s) {
                shared[lid] += shared[lid + s];
            }
            it.barrier(sycl::access::fence_space::local_space);
        }

        // First thread in block writes to global result atomically
        if (lid == 0) {
            sycl::atomic_ref<int,
                             sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>
                afr(result[0]);
            afr.fetch_add(shared[0]);
        }
    }    
    void runkernel() const {
    

    std::size_t local_size  = 256;
    std::size_t global_size = ((N + local_size - 1) / local_size) * local_size;
    sycl::nd_range<1> ndr{ {global_size}, {local_size} };

    q.submit([&](sycl::handler& h) {
      h.parallel_for<Reduce>(
        ndr,
        *this             
      );
    }).wait();
  }
    ~Reduce() {
    sycl::free(data, q);
    //delete[] reduced;
  }
};


