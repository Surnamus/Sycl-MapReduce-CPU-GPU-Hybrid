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

void Map::runkernel(sycl::queue q) const {
    size_t local_size = 256;
    size_t global_size = ((N + local_size - 1) / local_size) * local_size;
    sycl::nd_range<1> ndr{{global_size}, {local_size}};

    q.submit([&](sycl::handler& h) {
        h.parallel_for<Map>(ndr, *this);
    }).wait();
}

Reduce::Reduce(Mapped* _mappedw, std::size_t _N)
    : mappedw(_mappedw), N(_N) {}

void Reduce::operator()(sycl::nd_item<1> it,
                        sycl::local_accessor<int, 1> shared,
                        int* result) const {
    size_t gid = it.get_global_id(0);
    size_t mapped_size = N > MAXK ? N - MAXK + 1 : 1;
    if (gid >= mapped_size) { shared[it.get_local_id(0)] = 0; return; }

    if (gid > 0) {
        bool same = true;
        for (int j = 0; j < MAXK; ++j) {
            if (mappedw[gid].word[j] != mappedw[gid-1].word[j]) { same = false; break; }
            if (mappedw[gid].word[j] == '\0') break;
        }
        if (same) {
            sycl::atomic_ref<int,
                sycl::memory_order::relaxed,
                sycl::memory_scope::device,
                sycl::access::address_space::global_space> afr(mappedw[gid-1].v);
            afr.fetch_add(mappedw[gid].v);
            mappedw[gid].v = 0;
        }
    }

    constexpr int blocksize = 512;
    size_t lid = it.get_local_id(0);
    shared[lid] = mappedw[gid].v;
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

void Reduce::runkernel(int* result, sycl::queue q) const {
    size_t local_size = 512;
    size_t global_size = ((N + local_size - 1) / local_size) * local_size;
    sycl::nd_range<1> ndr{{global_size}, {local_size}};
    
    radixsort(q,MAXK);

    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<int, 1> shared(sycl::range<1>(local_size), h);
        h.parallel_for<Reduce>(ndr, [=](sycl::nd_item<1> it) {
            (*this)(it, shared, result);
        });
    }).wait();
}
//i found this somewhere
void Reduce::radixsort(sycl::queue &q, size_t k) const {
    Mapped* pointer = mappedw;
    size_t n = N;

    Mapped* tmp = sycl::malloc_shared<Mapped>(n, q);
    constexpr int radix = 4;

    for (int pos = (int)k - 1; pos >= 0; --pos) {
        int* counts = sycl::malloc_shared<int>(radix, q);
        int* prefix = sycl::malloc_shared<int>(radix, q);

        for (int i = 0; i < radix; ++i) counts[i] = 0;
        q.wait();

        q.submit([&](sycl::handler &h) {
            h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> gid) {
                size_t i = gid[0];
                if (i < n) {
                    char c = pointer[i].word[pos];
                    int val = (c == 'A') ? 0 :
                              (c == 'C') ? 1 :
                              (c == 'G') ? 2 : 3;
                    sycl::atomic_ref<int,
                        sycl::memory_order::relaxed,
                        sycl::memory_scope::device,
                        sycl::access::address_space::global_space>
                        afr(counts[val]);
                    afr.fetch_add(1);
                }
            });
        }).wait();

        q.submit([&](sycl::handler &h) {
            h.single_task([=]() {
                prefix[0] = 0;
                for (int r = 1; r < radix; ++r)
                    prefix[r] = prefix[r - 1] + counts[r - 1];
            });
        }).wait();

        q.submit([&](sycl::handler &h) {
            h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> gid) {
                size_t i = gid[0];
                if (i < n) {
                    char c = pointer[i].word[pos];
                    int val = (c == 'A') ? 0 :
                              (c == 'C') ? 1 :
                              (c == 'G') ? 2 : 3;
                    sycl::atomic_ref<int,
                        sycl::memory_order::relaxed,
                        sycl::memory_scope::device,
                        sycl::access::address_space::global_space>
                        afr(prefix[val]);
                    int idx = afr.fetch_add(1);
                    tmp[idx] = pointer[i]; 
                }
            });
        }).wait();

        q.memcpy(pointer, tmp, n * sizeof(Mapped)).wait();

        sycl::free(counts, q);
        sycl::free(prefix, q);
    }

    sycl::free(tmp, q);
}

} // namespace GPU
