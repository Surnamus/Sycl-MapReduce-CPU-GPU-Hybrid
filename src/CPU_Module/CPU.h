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
#include <iostream>
#include <string>
#include <vector>
#include <tuple>
#include <cstdlib> 
namespace sycl = cl::sycl;

struct Map {
    char* data;
    std::size_t N;
    sycl::queue& q;

    Map(char* _data, std::size_t _N, sycl::queue& _q)
      : data(_data), N(_N), q(_q) {}

    void operator()(sycl::nd_item<1> it) const;
    void runkernel() const;
    ~Map();
};
//struct Combine;
struct Reduce{
      char* data;
    std::size_t N;
    sycl::queue& q;

    Reduce(char* _data, std::size_t _N, sycl::queue& _q)
      : data(_data), N(_N), q(_q) {}

    void operator()(sycl::nd_item<1> it) const;
    void runkernel() const;
    ~Reduce();

};

#endif