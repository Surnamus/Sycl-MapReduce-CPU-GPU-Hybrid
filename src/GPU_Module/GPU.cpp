
//cpu mapreduce - 2 options
//parralelize by threads
//do 1 thread thingy
//no worries abt memory bcs ram
//it should make sense
//do std sort :]
//u ovom fajlu ce biti i map i reduce i combiner namenjeni za cpu, tj k mer algoritam
//dodati sort fazu, ali to moze i kod cpu-a , tj radix i std sort odvojeno benchmarking, kakogod, to kad se bude implementirao cpu deo
//add the references from main so such that it doesnt disrupt anything ( offswt and flattened) or just rewrite the whole thing if needed
//fix the naming a bit logic is OK
#include <filesystem>
#include <CL/sycl.hpp>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <execution>
#include <tuple>
#include <cstdlib> 
#include <iostream>
#include <vector>
#include <algorithm>
#include <ctime>
int constexpr MAXK=4;
namespace sycl = cl::sycl;

struct Mapped{
  char word[MAXK+1];
  int v;
   Mapped operator+(const Mapped& other) const {
       // rewrite so that its device valid

        Mapped m;
        //strcpy(m.word, word);
        for ( int i=0; i<sizeof(word) / sizeof(char*) ; i++){
          m.word[i]=word[i];
        }
        m.v = v + other.v;
        return m;
    }
};

//usm
struct Map {
    char* data;
    std::size_t N;        
    //sycl::queue& q;
    int k;

    Mapped* mappedw;      

    Map(char* _data, std::size_t _N, int _k)
        : data(_data), N(_N), k(_k) {
      //  mappedw = sycl::malloc_shared<Mapped>(N > k ? N - k + 1 : 1, q); in main
    }

    void operator()(sycl::nd_item<1> it) const  {
        size_t gid = it.get_global_id(0);
        if (gid > N - k) return;  

       
        bool valid = true;
        for (int i = 0; i < k; ++i) {
            if (data[gid + i] == '\0') {
                valid = false;
                break;
            }
        }
        if (!valid) return;

        for (int i = 0; i < k; ++i) {
            mappedw[gid].word[i] = data[gid + i];
        }
        mappedw[gid].word[k] = '\0';
        mappedw[gid].v = 1;
    }

    void runkernel(sycl::queue q) const {
        size_t local_size = 256;
        size_t global_size = ((N + local_size - 1) / local_size) * local_size;
        sycl::nd_range<1> ndr{{global_size}, {local_size}};

        q.submit([&](sycl::handler& h) {
            h.parallel_for<Map>(ndr, *this);
        }).wait();
    }

   
};

struct Reduce {
    Mapped* mappedw;  // instead of char* data and offsets
    size_t N;
   // sycl::queue& q;

    Reduce(Mapped* _mappedw, size_t _N)
        : mappedw(_mappedw), N(_N) {}

    void operator()(sycl::nd_item<1> it,
                    sycl::local_accessor<int, 1> shared,
                    int* result) const {
        constexpr int blocksize = 512;

        size_t gid = it.get_global_id(0);
        size_t lid = it.get_local_id(0);

        int v = 0;
        if (gid < N) {
            const char* kmer = mappedw[gid].word;
            int len = 0;
            while (kmer[len] != '\0') len++;
            v = len;
        }

        shared[lid] = v;
        it.barrier(sycl::access::fence_space::local_space);

        for (size_t s = blocksize / 2; s > 0; s >>= 1) {
            if (lid < s) {
                shared[lid] += shared[lid + s];
            }
            it.barrier(sycl::access::fence_space::local_space);
        }

        if (lid == 0) {
            sycl::atomic_ref<int,
                sycl::memory_order::relaxed,
                sycl::memory_scope::device,
                sycl::access::address_space::global_space>
                afr(result[0]);
            afr.fetch_add(shared[0]);
        }
    }

    void runkernel(int* result,sycl::queue q) const {
        size_t local_size = 256;
        size_t global_size = ((N + local_size - 1) / local_size) * local_size;
        sycl::nd_range<1> ndr{{global_size}, {local_size}};

        

        q.submit([&](sycl::handler& h) {
            sycl::local_accessor<int, 1> shared(sycl::range<1>(local_size), h);
            h.parallel_for<Reduce>(ndr, [=](sycl::nd_item<1> it) {
                (*this)(it, shared, result);
            });
        }).wait();
    }

    
};

//fix the structure a bit
//can use gpu code with some modifications in reduction kernel and bit in map one
//optimize the part