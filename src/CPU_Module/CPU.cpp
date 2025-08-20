
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
#include <cstring>
#include <map>

#include "CPU.h"

namespace sycl = cl::sycl;
namespace CPU {

Mapped Mapped::operator+(const Mapped& other) const {
    Mapped m;
    for (int i = 0; i < MAXK + 1; i++) {
        m.word[i] = word[i];
    }
    m.v = v + other.v;
    return m;
}

Map::Map(char* _data, std::size_t _N, int _k)
    : data(_data), N(_N), k(_k), mappedw(nullptr) {}

void Map::operator()(sycl::nd_item<1> it) const {
    size_t gid = it.get_global_id(0);
    if (gid >= (N > k ? N - k + 1 : 1)) return; // Prevent OOB

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

void Map::runkernel(sycl::queue& q) const {
    size_t local_size = 4;
    size_t global_size = ((N + local_size - 1) / local_size) * local_size;

    sycl::nd_range<1> ndr{{global_size}, {local_size}};

    q.submit([&](sycl::handler& h) {
        h.parallel_for<Map>(ndr, *this);
    }).wait();
}

Reduce::Reduce(Mapped* _mappedw, size_t _N)
    : mappedw(_mappedw), N(_N) {}

bool Reduce::lex_compare(const Mapped &a, const Mapped &b) {
    for (int i = 0; i < MAXK; ++i) {
        if (a.word[i] != b.word[i]) return a.word[i] < b.word[i];
        if (a.word[i] == '\0') break;
    }
    return false;
}

void Reduce::operator()(sycl::nd_item<1> it,
                        sycl::local_accessor<int, 1> shared,
                        int* result) const {
    size_t gid = it.get_global_id(0);
    size_t mapped_size = N;

    if (gid > 0) {
        bool is_last = (gid == N - 1);
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
               // if (i != head) mappedw[i].v = 0;
            }

            sycl::atomic_ref<int,
                sycl::memory_order::relaxed,
                sycl::memory_scope::device,
                sycl::access::address_space::global_space> afr(mappedw[head].v);
            afr.fetch_add(sum - mappedw[head].v);
            for (int i = head + 1; i <= gid; i++) {
                     mappedw[i].v = 0;
                    }
        } 
    }

    constexpr int blocksize = 64;
    size_t lid = it.get_local_id(0);
    shared[lid] = (gid < N && mappedw[gid].v > 0) ? mappedw[gid].v : 0;
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

void Reduce::runkernel(int* result, sycl::queue& q) const {
        std::stable_sort(mappedw, mappedw + N,
        [](const Mapped &a, const Mapped &b) { return std::strcmp(a.word, b.word) < 0; });

    size_t local_size = 64;
    size_t global_size = ((N + local_size - 1) / local_size) * local_size;
    sycl::nd_range<1> ndr{{global_size}, {local_size}};
            auto self =(*this);
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<int, 1> shared(sycl::range<1>(local_size), h);
        h.parallel_for<Reduce>(ndr, [=](sycl::nd_item<1> it) {
            self(it, shared, result);
        });
    }).wait();
}


void Reduce::seqRed(Mapped* mappedw, size_t* newsize, size_t s) {
    if (N == 0) {
        *newsize = 0;
        return;
    }

    std::stable_sort(mappedw, mappedw + N,
        [](const Mapped &a, const Mapped &b) { return std::strcmp(a.word, b.word) < 0; });

    int t = 0;  

    for (int i = 1; i < N; ++i) {
        if (std::strcmp(mappedw[i].word, mappedw[t].word) == 0) {
            mappedw[t].v += mappedw[i].v; 
        } else {
            t++;
            mappedw[t] = mappedw[i]; 
        }
    }

    *newsize = t + 1; 
}

 // namespace CPU
}