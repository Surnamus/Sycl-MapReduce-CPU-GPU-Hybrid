#ifndef GPU_H
#define GPU_H
#include <filesystem>
#include <CL/sycl.hpp>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <tuple>
#include <cstdlib> 

namespace sycl = cl::sycl;

namespace GPU{
  inline constexpr int MAXK=4; //
  struct Mapped{
    char word[MAXK+1];
    int v;
    Mapped operator+(const Mapped& other) const;
  }; //

  struct Map {
      char* data;
      std::size_t N;
      Mapped* mappedw;   //
      int k;

      Map(char* _data, std::size_t _N, int _k);

      void operator()(sycl::nd_item<1> it) const;
      void runkernel(sycl::queue q) const;
  };

  struct Reduce{
      Mapped* mappedw;
      std::size_t N;

      Reduce(Mapped* _mappedw, std::size_t _N);

      void operator()(sycl::nd_item<1> it,
                      sycl::local_accessor<int, 1> shared,
                      int* result) const;
      void runkernel(int* result, sycl::queue q) const;
      void radixsort(sycl::queue &q, size_t k) const;
  };
}
#endif
