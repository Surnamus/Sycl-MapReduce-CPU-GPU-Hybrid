
#include <CL/sycl.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cstring>
#include "helper.h"
#include "GPU.h"
#include "CPU.h"

namespace sycl = cl::sycl;

void run() {
    std::ofstream f("start_measure");
    f << "go\n";
    f.close();
}

std::pair<std::string, std::vector<size_t>> convert(std::vector<std::string> strings) {
    std::vector<size_t> offsets;
    std::string flattened;
    for (const auto& s : strings) {
        offsets.push_back(flattened.size());
        flattened += s;
        flattened += '\0';
    }
    return {flattened, offsets};
    // to convert to char* = std::string::c_str() or std::string::data()
}

int main() {
    //init();
    std::vector<std::string> datav = prepare();
    std::cout << "Finished preparing!" << std::endl;
    std::vector<std::string> dataset_used = dataset_selector(datav);
    std::cin.clear(); // clear fail and eof flags
std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // discard leftover input
    std::tuple<sycl::device, int> dev = Program_device_selector();
    
    // Fixed: removed hipsycl namespace, using standard sycl
    sycl::queue q{std::get<0>(dev)};
    
    std::pair<std::string, std::vector<size_t>> datadev = convert(dataset_used);

    char* flat_data = sycl::malloc_shared<char>(datadev.first.size() + 1, q);
    std::memcpy(flat_data, datadev.first.data(), datadev.first.size() + 1);
    
    int k = 3;
    std::cout << "Running on "
              << q.get_device().get_info<sycl::info::device::name>() << "\n";
    
    size_t N = datadev.first.size();
    size_t mapped_size = (N > k) ? (N - k + 1) : 1;
    GPU::Mapped* mappedwm = sycl::malloc_shared<GPU::Mapped>(mapped_size, q);
    int* result = sycl::malloc_shared<int>(1, q); // Allocate result memory
    *result = 0; // Initialize result
    
    if (std::get<1>(dev) == 1) {
        // GPU-only execution
        run();
        GPU::Map map_op(flat_data, N, k);
        map_op.runkernel(q);
        q.wait();
        GPU::Reduce reduce_op(map_op.mappedw, N);
        reduce_op.runkernel(result, q);
        q.wait();
    }
    else if (std::get<1>(dev) == 2) {
        // CPU-only execution
        run();
        CPU::Map map_op(flat_data, N, k);
        map_op.runkernel(q);
        q.wait();
        CPU::Reduce reduce_op(map_op.mappedw, N);
        reduce_op.runkernel(result,q);
        q.wait();
    }
    else {
        // Hybrid execution (GPU Map + CPU Reduce)
        run();
        sycl::queue q{sycl::gpu_selector{}};
        GPU::Map map_op(flat_data, N, k);
        map_op.runkernel(q);

// Wait for GPU map to complete before starting CPU reduce
        q.wait();

// Cast GPU::Mapped* to CPU::Mapped*
        sycl::queue cpu_q{sycl::cpu_selector{}};
        CPU::Reduce reduce_op(reinterpret_cast<CPU::Mapped*>(map_op.mappedw), N);
        reduce_op.runkernel(result, cpu_q);
        cpu_q.wait(); 
    }
    
    // Wait for all operations to complete
    
    // Output the result
    std::cout << "Final result: " << *result << std::endl;
    
    // Clean up
    sycl::free(flat_data, q);
    sycl::free(result, q);
    
    std::cout << "Program completed successfully." << std::endl;
    return 0;
}


