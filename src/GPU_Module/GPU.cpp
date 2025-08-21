//fix this gpu code ,aka memory bounds and analyse why
#include <utility> 
#include <filesystem>
#include <CL/sycl.hpp>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <execution>
#include <tuple>
#include <cstdlib> 
#include <algorithm>
#include <ctime>
#include "GPU.h"
#include <AdaptiveCpp/algorithms/algorithm.hpp>
namespace sycl = cl::sycl;

namespace GPU {

Mapped Mapped::operator+(const Mapped& other) const {
    Mapped m;
    for ( int i=0; i<MAXK+1 ; i++){
        m.word[i]=word[i];
    }
    m.v = v + other.v;
    return m;
}

Map::Map(char* _data, std::size_t _N, int _k)
    : data(_data), N(_N), k(_k), mappedw(nullptr) {}

void Map::operator()(sycl::nd_item<1> it) const  {
        size_t gid = it.get_global_id(0);
    if (gid >= (N > k ? N - k + 1 : 1)) return;  // safe mappedw indexing

    bool valid = true;
    for (int i = 0; i < k; ++i) {
        if (data[gid + i] == '\0') { valid = false; break; }
    }
    if (!valid) return;

    for (int i = 0; i < k; ++i) mappedw[gid].word[i] = data[gid + i];
    mappedw[gid].word[k] = '\0';
    mappedw[gid].v = 1;
}

void Map::runkernel(sycl::queue q,size_t lsize) const {
    size_t local_size = lsize;
    size_t global_size = ((N + local_size - 1) / local_size) * local_size;
    sycl::nd_range<1> ndr{{global_size}, {local_size}};

    q.submit([&](sycl::handler& h) {
        h.parallel_for<Map>(ndr, *this);
    }).wait();
}

Reduce::Reduce(Mapped* _mappedw, std::size_t _rN)
    : mappedw(_mappedw), rN(_rN) {}

void Reduce::operator()(sycl::nd_item<1> it,
                        sycl::local_accessor<int, 1> shared,
                        int* result) const {
    size_t gid = it.get_global_id(0);
    size_t mapped_size = rN;

    if (gid > 0) {
        bool is_last = (gid == rN - 1);
        if (!is_last) {
            for (int j = 0; j < MAXK; ++j) {
                char a = mappedw[gid].word[j];
                char b = mappedw[gid+1].word[j];
                if (a != b) { is_last = true; break; }
                if (a == '\0' || b == '\0') break;
            }
        }

        if (is_last) {
            int head = gid;
            while (head > 0) {
                bool same = true;
                for (int j = 0; j < MAXK; ++j) {
                    char a = mappedw[head].word[j];
                    char b = mappedw[head-1].word[j];
                    if (a != b) { same = false; break; }
                    if (a == '\0') break;
                }
                if (!same) break;
                head--;
            }

            int sum = 0;
            for (int i = head; i <= gid; i++) {
                sum += mappedw[i].v;
             //   if (i != head) mappedw[i].v = 0;
             //here
            }
            //here
               

            sycl::atomic_ref<int,
                sycl::memory_order::relaxed,
                sycl::memory_scope::device,
                sycl::access::address_space::global_space> afr(mappedw[head].v);
            afr.fetch_add(sum- mappedw[head].v); //- mappedw[head].v
            //here
             for (int i = head + 1; i <= gid; i++) {
                     mappedw[i].v = 0;
                    }
        }
    }

    constexpr int blocksize = 512;
    size_t lid = it.get_local_id(0);
    shared[lid] = (gid < rN && mappedw[gid].v > 0) ? mappedw[gid].v : 0;
    it.barrier(sycl::access::fence_space::local_space);

    for (size_t s = blocksize / 2; s > 0; s >>= 1) {
        if (lid < s) shared[lid] += shared[lid + s];
        it.barrier(sycl::access::fence_space::local_space);
    }

    if (lid == 0) {
        sycl::atomic_ref<int,
            sycl::memory_order::relaxed,
            sycl::memory_scope::device,
            sycl::access::address_space::global_space> afr(result[0]);
        afr.fetch_add(shared[0]);
    }
}

//here

void Reduce::runkernel(int* result, sycl::queue q,size_t lsize) const {
    size_t local_size = 512;
    size_t global_size = ((rN + local_size - 1) / local_size) * local_size;
    sycl::nd_range<1> ndr{{global_size}, {local_size}};
    
    //better cpu sorting for now
    //std::stable_sort(mappedw, mappedw + rN, [](const Mapped &a, const Mapped &b) { return std::strcmp(a.word, b.word) < 0; });
    auto cmp = [](const Mapped &a, const Mapped &b) {
    for (int i = 0; i < MAXK; ++i) {
        char ca = a.word[i];
        char cb = b.word[i];
        if (ca != cb) return ca < cb;  
        if (ca == '\0') break;        // both strings ended
    }
    return false;  // equal strings
};
//i wasted 3 days finding this because of their retarded aah documetation 
//if anyone is interested look into their github page and go to docs
//they dont have a site like normal devs do
//0/10
acpp::algorithms::sort(q,mappedw,mappedw+rN,cmp);
    auto self = *this; 
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<int, 1> shared(sycl::range<1>(local_size), h);
        h.parallel_for<Reduce>(ndr, [=](sycl::nd_item<1> it) {
            self(it, shared, result);
        });
    }).wait();
    //HERE

}

} // namespace GPU