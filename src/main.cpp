
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
constexpr int MAXK=3;
int compute_unique_total(GPU::Mapped* mappedw, size_t setsize) {
    int total = 0;
    for (size_t i = 0; i < setsize; ++i) {
        total += mappedw[i].v; // only unique counts
    }
    return total;
}
void run() {
    std::ofstream f("start_measure");
    f << "go\n";
    f.close();
}
void print_mapped_counts(GPU::Mapped* mappedw, size_t setsize, int k) {
    for (size_t i = 0; i < setsize; ++i) {
        if (mappedw[i].v > 0 && mappedw[i].word[0] != '\0') {
            mappedw[i].word[k] = '\0'; // ensure termination
            std::cout << mappedw[i].word << " : " << mappedw[i].v << "\n";
        }
    }
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
    std::vector<std::string> datav = prepare();
    std::cout << "Finished preparing!" << std::endl;
    std::vector<std::string> dataset_used = dataset_selector(datav);

    std::cin.clear();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    std::tuple<sycl::device, int> dev = Program_device_selector();
    sycl::queue q{std::get<0>(dev)}; 

    auto datadev = convert(dataset_used);
    int k = MAXK; // you said k == MAXK
    size_t N = datadev.first.size();
    size_t setsize = (N >= static_cast<size_t>(k)) ? (N - k + 1) : 0;

    int* result = sycl::malloc_shared<int>(1, q);
    *result = 0;

    char* flat_data = nullptr;
    GPU::Mapped* mappedwm = nullptr;
    sycl::queue q_used = q; // queue used for allocations (will be updated for hybrid)

    if (std::get<1>(dev) == 1) {
        q_used = q;
        flat_data = sycl::malloc_shared<char>(datadev.first.size() + 1, q_used);
        std::memcpy(flat_data, datadev.first.data(), datadev.first.size() + 1);

        mappedwm = sycl::malloc_shared<GPU::Mapped>(setsize, q_used);
        for (size_t i = 0; i < setsize; ++i) {
            mappedwm[i].v = 0;
            std::memset(mappedwm[i].word, 0, sizeof(mappedwm[i].word));
        }

        run();
        GPU::Map mapf(flat_data, N, k);
        mapf.mappedw = mappedwm;
        mapf.runkernel(q);
        q.wait();

        GPU::Reduce reducef(mappedwm, N);
        reducef.runkernel(result, q);
        q.wait();
    }
    else if (std::get<1>(dev) == 2) {
        q_used = q;
        flat_data = sycl::malloc_shared<char>(datadev.first.size() + 1, q_used);
        std::memcpy(flat_data, datadev.first.data(), datadev.first.size() + 1);

        mappedwm = sycl::malloc_shared<GPU::Mapped>(setsize, q_used);
        for (size_t i = 0; i < setsize; ++i) {
            mappedwm[i].v = 0;
            std::memset(mappedwm[i].word, 0, sizeof(mappedwm[i].word));
        }

        run();
        CPU::Map mapf(flat_data, N, k);
        mapf.mappedw = reinterpret_cast<CPU::Mapped*>(mappedwm);
        mapf.runkernel(q);
        q.wait();

        CPU::Reduce reducef(reinterpret_cast<CPU::Mapped*>(mappedwm), N);
        reducef.runkernel(result, q);
        q.wait();
    }
    else {
        sycl::queue gpu_q{sycl::gpu_selector{}};
        sycl::queue cpu_q{sycl::cpu_selector{}};
        q_used = gpu_q; 

        flat_data = sycl::malloc_shared<char>(datadev.first.size() + 1, q_used);
        std::memcpy(flat_data, datadev.first.data(), datadev.first.size() + 1);

        mappedwm = sycl::malloc_shared<GPU::Mapped>(setsize, q_used);
        for (size_t i = 0; i < setsize; ++i) {
            mappedwm[i].v = 0;
            std::memset(mappedwm[i].word, 0, sizeof(mappedwm[i].word));
        }

        run();
        GPU::Map mapf(flat_data, N, k);
        mapf.mappedw = mappedwm;
        mapf.runkernel(gpu_q);
        gpu_q.wait();

        CPU::Reduce reducef(reinterpret_cast<CPU::Mapped*>(mappedwm), N);
        reducef.runkernel(result, cpu_q);
        cpu_q.wait();
    }

    print_mapped_counts(mappedwm, setsize, k);
    int total_unique = 0;
    for (size_t i = 0; i < setsize; ++i) if (mappedwm[i].v > 0) total_unique += mappedwm[i].v;
}



