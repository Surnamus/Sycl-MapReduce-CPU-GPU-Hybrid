//u ovom fajlu ce biti i map i reduce i combiner namenjeni za gpu, tj k mer algoritam
#include <filesystem>
#include <CL/sycl.hpp>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <tuple>
#include <cstdlib> 

namespace sycl = cl::sycl;
