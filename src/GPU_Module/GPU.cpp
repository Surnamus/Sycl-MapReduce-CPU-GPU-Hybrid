//fix this gpu code ,aka memory bounds and analyse why

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
                if (i != head) mappedw[i].v = 0;
            }

            sycl::atomic_ref<int,
                sycl::memory_order::relaxed,
                sycl::memory_scope::device,
                sycl::access::address_space::global_space> afr(mappedw[head].v);
            afr.fetch_add(sum - mappedw[head].v);
        } else {
            mappedw[gid].v = 0;
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

void Reduce::runkernel(int* result, sycl::queue q) const {
    size_t local_size = 512;
    bool last_group_full = (rN % local_size == 0);
    size_t global_size = ((rN + local_size - 1) / local_size) * local_size;
    sycl::nd_range<1> ndr{{global_size}, {local_size}};
    
    //radixsort(q,MAXK);
    //better cpu sorting for now
    std::stable_sort(mappedw, mappedw + rN, [](const Mapped &a, const Mapped &b) { return std::strcmp(a.word, b.word) < 0; });

    auto self = *this; 
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<int, 1> shared(sycl::range<1>(local_size), h);
        h.parallel_for<Reduce>(ndr, [=](sycl::nd_item<1> it) {
            self(it, shared, result);
        });
    }).wait();
    //HERE

}
//i found this somewhere
void Reduce::radixsort(sycl::queue &q, size_t k) const {
    Mapped* pointer = mappedw;
    size_t n = rN;

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
                if (i >= n) return; 
                if (i < n) {
                    char c = pointer[i].word[pos];
                    int val = (c == 'A') ? 0 :
                              (c == 'C') ? 1 :
                              (c == 'G') ? 2 : 3;
                    if (val >= radix) return;        
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
                if (i >= n) return;  
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
                    if (idx >= n) return;  
                    if (val >= radix) return;
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