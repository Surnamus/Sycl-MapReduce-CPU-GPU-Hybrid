//fix gpu so that it used setsize
//fix plotter
#include <CL/sycl.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cstring>
#include "helper.h"
#include "GPU.h"
#include "CPU.h"
#include <unordered_set>
#include <string>
namespace sycl = cl::sycl;
std::string safe_num_to_string(int val) {
    if (val == 0) return "null";
    return std::to_string(val);
}
void POINTSFILE(int N,int k, int lls,int llsc,int device,int metric,double value,bool warmup){
    //std::string s="";
    if (warmup) return;
    else{
    std::string s = std::to_string(N) + " " + std::to_string(k) + " " + std::to_string(lls) + " " + 
        std::to_string(llsc) + " " + std::to_string(device) + " " +std::to_string(metric) + " " + std::to_string(value);
    
    std::ofstream outFile("points.txt", std::ios::app); 
    std::ofstream outFileD("DATA.txt", std::ios::app); 
    if ( outFile.is_open() && outFileD.is_open()) {
        outFile << s << std::endl;
        outFile.close();
        outFileD << s << std::endl;
        outFileD.close();
        std::cout << "Content appended to file." << std::endl;
    } else {
        std::cerr << "Error opening file!" << std::endl;
    }
}
}
void print_mapped_countst(GPU::Mapped* mappedw, size_t setsize, int k) {
    for (size_t i = 0; i < setsize; ++i) {
        if (mappedw[i].v > 0 && mappedw[i].word[0] != '\0') {
            mappedw[i].word[k] = '\0'; // ensure termination
            std::cout << mappedw[i].word << " : " << mappedw[i].v << "\n";
        }
    }
}
int compute_unique_total(GPU::Mapped* mappedw, size_t setsize) {
    int total = 0;
    for (size_t i = 0; i < setsize; ++i) {
        total += mappedw[i].v; 
    }
    return total;
}

void print_mapped_counts(GPU::Mapped* mappedw, size_t setsize, int k) {
    std::ofstream out("/home/user/project/output.txt"); // hardcoded output file
    if (!out) {
        std::cerr << "Failed to open output file\n";
        return;
    }

    std::unordered_set<std::string> printed;

    for (size_t i = 0; i < setsize; ++i) {
        if (mappedw[i].v > 1  && mappedw[i].word[0] != '\0' ) { 
            std::string w(mappedw[i].word, k); // assuming word length k
            if (printed.find(w) == printed.end()) { // not printed yet
                out << w << " : " << mappedw[i].v << "\n";
                printed.insert(w);
            }
        }
    }
}
void print_mapped_countst(CPU::Mapped* mappedw, size_t u, int k) {
    for (size_t i = 0; i < u; ++i) {
        if (mappedw[i].v != -1 && mappedw[i].word[0] != '\0') { // 
          //  mappedw[i].word[k] = '\0'; // ensure termination
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
}





// MAIN






int main(int argc, char* argv[]) {
       if (argc <8) {
        std::cerr << "NOT ENOUGH ARGUMENTS IMAGINE LOL"<<std::endl;
        return 1;
    }


    size_t N = std::atoi(argv[1]);
    size_t k = std::atoi(argv[2]);
    size_t localsize = std::atoi(argv[3]);
    size_t lssc = std::atoi(argv[4]);
    int device = std::stoi(argv[5]);
    int metricIndex = std::stoi(argv[6]);
    bool warmup= std::stoi(argv[7]);

    std::vector<std::string> datav = prepare();
    std::cout << "Finished preparing!" << std::endl;

    sycl::device dev = Program_device_selector(device);
    sycl::queue q{dev,sycl::property::queue::enable_profiling() }; 

    auto datadev = convert(datav);
    //int k = MAXK;
     

    //device=std::get<1>(dev);

    // N = datadev.first.size(); //avoid redefintion error
    size_t setsize = (N >= static_cast<size_t>(k)) ? (N - k + 1) : 0;

    

    char* flat_data = nullptr;
    GPU::Mapped* mappedwm = nullptr;
    sycl::queue q_used = q; // queue used for allocations (will be updated for hybrid)
        std::cout << "Running on "
              << q.get_device().get_info<sycl::info::device::name>()<<"\n" ; 
SyclProfiler profiler;  
if (device == 1) {
    q_used = q;
    flat_data = sycl::malloc_device<char>(datadev.first.size() + 1, q);
    q.memcpy(flat_data, datadev.first.data(), datadev.first.size() + 1).wait();

    mappedwm = sycl::malloc_device<GPU::Mapped>(setsize, q);

    // map
    GPU::Map mapf(flat_data, N, k);
    mapf.mappedw = mappedwm;

    sycl::event e_map = mapf.runkernel(q, localsize);
    profiler.setKernelEvent(e_map);
    profiler.Metric("START", metricIndex,false);
    e_map.wait();
    profiler.Metric("STOP", metricIndex,false);

    // Reduce kernel
    GPU::Reduce reducef(mappedwm, setsize);
    sycl::event e_red = reducef.runkernel(q, localsize);
    profiler.setKernelEvent(e_red);
    profiler.Metric("START", metricIndex,false);
    e_red.wait();
    profiler.Metric("STOP", metricIndex,false);
    double result = profiler.getSum(metricIndex);

    POINTSFILE(N,k,localsize,lssc,device,metricIndex,result,warmup);

    std::vector<GPU::Mapped> host_mapped(setsize);
    q.memcpy(host_mapped.data(), mappedwm, setsize * sizeof(GPU::Mapped)).wait();
    print_mapped_counts(host_mapped.data(), setsize, k);

}
else if (device == 2) {
    q_used = q;
    flat_data = sycl::malloc_shared<char>(datadev.first.size() + 1, q);
    std::memcpy(flat_data, datadev.first.data(), datadev.first.size() + 1);

    mappedwm = sycl::malloc_shared<GPU::Mapped>(setsize, q);
    for (size_t i = 0; i < setsize; ++i) {
        mappedwm[i].v = 0;
        std::memset(mappedwm[i].word, 0, sizeof(mappedwm[i].word));
    }

    CPU::Map mapf(flat_data, N, k);
    mapf.mappedw = reinterpret_cast<CPU::Mapped*>(mappedwm);
    sycl::event e_map = mapf.runkernel(q, localsize);
    profiler.setKernelEvent(e_map);
    profiler.Metric("START", metricIndex,true);
    e_map.wait();
    profiler.Metric("STOP", metricIndex,true);

    CPU::Reduce reducef(reinterpret_cast<CPU::Mapped*>(mappedwm), setsize);
    sycl::event e_red = reducef.runkernel(q, localsize);
    profiler.setKernelEvent(e_red);
    profiler.Metric("START", metricIndex,true);
    e_red.wait();
    profiler.Metric("STOP", metricIndex,true);
    double result = profiler.getSum(metricIndex);

    POINTSFILE(N,k,localsize,lssc,device,metricIndex,result,warmup);
    print_mapped_counts(mappedwm, setsize, k);

}
else {
    // hybrid: GPU map, CPU reduce
    sycl::queue gpu_q{sycl::gpu_selector{},sycl::property::queue::enable_profiling() };
    sycl::queue cpu_q{sycl::cpu_selector{},sycl::property::queue::enable_profiling() };
    q_used = gpu_q;

    flat_data = sycl::malloc_shared<char>(datadev.first.size() + 1, gpu_q);
    std::memcpy(flat_data, datadev.first.data(), datadev.first.size() + 1);

    mappedwm = sycl::malloc_shared<GPU::Mapped>(setsize, gpu_q);
    for (size_t i = 0; i < setsize; ++i) {
        mappedwm[i].v = 0;
        std::memset(mappedwm[i].word, 0, sizeof(mappedwm[i].word));
    }

    // GPU Map kernel
    GPU::Map mapf(flat_data, N, k);
    mapf.mappedw = mappedwm;
    sycl::event e_map = mapf.runkernel(gpu_q, localsize);
    profiler.setKernelEvent(e_map);
    profiler.Metric("START", metricIndex,false);
    e_map.wait();
    profiler.Metric("STOP", metricIndex,false);

    localsize = lssc;

    // CPU Reduce kernel
    CPU::Reduce reducef(reinterpret_cast<CPU::Mapped*>(mappedwm), setsize);
    sycl::event e_red = reducef.runkernel(cpu_q, localsize);
    profiler.setKernelEvent(e_red);
    profiler.Metric("START", metricIndex,true);
    e_red.wait();
    profiler.Metric("STOP", metricIndex,true);
    double result = profiler.getSum(metricIndex);

    POINTSFILE(N,k,localsize,lssc,device,metricIndex,result,warmup);
    print_mapped_counts(mappedwm, setsize, k);

}
    sycl::free(flat_data, q);
    sycl::free(mappedwm, q);
}


