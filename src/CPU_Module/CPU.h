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
namespace CPU{
  int constexpr MAXK=4; //
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
}; //
struct Map {
    char* data;
    std::size_t N;
   // sycl::queue q;
    Mapped* mappedw;   //
    int k;
    Map(char* _data, std::size_t _N, int _k)
      : data(_data), N(_N), k(_k) {}

    void operator()(sycl::nd_item<1> it) const;
    void runkernel(sycl::queue& _q) const;
    //~Map();
};
//struct Combine;
struct Reduce{
      char* data;
    std::size_t N;
    //sycl::queue& q;

    Reduce(char* _data, std::size_t _N)
      : data(_data), N(_N) {}

    void operator()(sycl::nd_item<1> it) const;
    void runkernel(sycl::queue& _q) const;
   // ~Reduce();

};


#endif
}