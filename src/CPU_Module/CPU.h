/*The cake is a lie*/
#ifndef CPU_H
#define CPU_H

#include <filesystem>
#include <CL/sycl.hpp>
#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>
#include <ctime>
#include <string>
#include <tuple>
#include <cstdlib> 

namespace sycl = cl::sycl;
namespace CPU {
    inline constexpr int MAXK = 4;

    struct Mapped {
        char word[MAXK+1];
        int v;
        Mapped operator+(const Mapped& other) const;
    };

    struct Map {
        char* data;
        std::size_t N;
        Mapped* mappedw;
        int k;

        Map(char* _data, std::size_t _N, int _k);
        void operator()(sycl::nd_item<1> it) const;
        void runkernel(sycl::queue& _q) const;
    };

    struct Reduce {
        Mapped* mappedw;
        size_t N;

        Reduce(Mapped* _mappedw, size_t _N);
        static bool lex_compare(const Mapped &a, const Mapped &b);
        void operator()(sycl::nd_item<1> it,
                        sycl::local_accessor<int, 1> shared,
                        int* result) const;
        void runkernel(int* result, sycl::queue& _q) const;
    };
}
#endif
