#ifndef HELPER_H
#define HELPER_H
#include <CL/sycl.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <tuple>
#include <string>
#include <vector>
namespace sycl = cl::sycl;
std::vector<std::string> prepare();
void init();
std::vector<std::string> dataset_selector(std::vector<std::string> data);
std::tuple<sycl::device,int> Program_device_selector();
#endif