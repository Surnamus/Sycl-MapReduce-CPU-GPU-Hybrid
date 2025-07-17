
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
//TODO: definiisati sve potrene funkcije i videti jel kernel moze da se napise na ikakav nacin bez lambde
struct Map {
    char** data;
    std::size_t N;
    sycl::queue& q;

    Map(char** _data, std::size_t _N, sycl::queue& _q)
      : data(_data), N(_N), q(_q) {}

    void operator()(sycl::nd_item<1> it) const;
    void runkernel() const;
    ~Map();
};
//struct Combine;
struct Reduce{
      char** data;
    std::size_t N;
    sycl::queue& q;

    Reduce(char** _data, std::size_t _N, sycl::queue& _q)
      : data(_data), N(_N), q(_q) {}

    void operator()(sycl::nd_item<1> it) const;
    void runkernel() const;
    ~Reduce();

};
#endif