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
constexpr int MAXK=3;
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
void run() {
    std::ofstream f("start_measure");
    f << "go\n";
    f.close();
}
void stop() {
    std::ofstream f("stop");
    f << "done\n";
    f.close();
}
// Safe printer for CPU::Mapped (uses bounded length)
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
    // to convert to char* = std::string::c_str() or std::string::data()
}


int main(int argc, char* argv[]) {
       if (argc < 6) {
        std::cerr << "Usage: " << argv[0] << " N K LS BS\n";
        return 1;
    }
    std::vector<std::string> datav = prepare();
    std::cout << "Finished preparing!" << std::endl;
    //std::vector<std::string> dataset_used = dataset_selector(datav);
    //std::vector<std::string> dataset_used = datav;
    //std::cin.clear();
    //std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    std::tuple<sycl::device, int> dev = Program_device_selector();
    sycl::queue q{std::get<0>(dev)}; 

    auto datadev = convert(datav);
    //int k = MAXK;
     

    size_t N = std::atoi(argv[1]);
    size_t k = std::atoi(argv[2]);
    size_t localsize = std::atoi(argv[3]);

     N = datadev.first.size(); //avoid redefintion error
    size_t setsize = (N >= static_cast<size_t>(k)) ? (N - k + 1) : 0;

    int* result = sycl::malloc_shared<int>(1, q);
    *result = 0;

    char* flat_data = nullptr;
    GPU::Mapped* mappedwm = nullptr;
    sycl::queue q_used = q; // queue used for allocations (will be updated for hybrid)
        std::cout << "Running on "
              << q.get_device().get_info<sycl::info::device::name>()<<"\n" ; 
    if (std::get<1>(dev) == 1) {
        q_used = q;
        flat_data = sycl::malloc_device<char>(datadev.first.size() + 1, q);
        q.memcpy(flat_data, datadev.first.data(), datadev.first.size() + 1);

        mappedwm = sycl::malloc_device<GPU::Mapped>(setsize, q);
        //run();
        GPU::Map mapf(flat_data, N, k);
        mapf.mappedw = mappedwm;
        q.wait();
        
        run();
        mapf.runkernel(q,localsize);
        q.wait();
        stop();
        

        GPU::Reduce reducef(mappedwm, setsize);
        q.wait();
        run();
        reducef.runkernel(result, q,localsize);
        q.wait();
        stop();

        std::vector<GPU::Mapped> host_mapped(setsize);

        q.memcpy(host_mapped.data(), mappedwm, setsize * sizeof(GPU::Mapped)).wait();
        print_mapped_counts(host_mapped.data(), setsize, k);
        
    }
    else if (std::get<1>(dev) == 2) {
        q_used = q;
        flat_data = sycl::malloc_shared<char>(datadev.first.size() + 1, q);
        std::memcpy(flat_data, datadev.first.data(), datadev.first.size() + 1);

        mappedwm = sycl::malloc_shared<GPU::Mapped>(setsize, q);
        for (size_t i = 0; i < setsize; ++i) {
            mappedwm[i].v = 0;
            std::memset(mappedwm[i].word, 0, sizeof(mappedwm[i].word));
        }

        //run();
        CPU::Map mapf(flat_data, N, k);
        mapf.mappedw = reinterpret_cast<CPU::Mapped*>(mappedwm);
        run();
        mapf.runkernel(q,localsize);
        q.wait();
        stop();

        CPU::Reduce reducef(reinterpret_cast<CPU::Mapped*>(mappedwm), setsize);
        run(); //N
           reducef.runkernel(result,q,localsize);
            q.wait();
            stop();
        //reducef.seqRed(reinterpret_cast<CPU::Mapped*>(mappedwm),result1,N);
        print_mapped_counts(mappedwm,setsize, k);
    }
    else {
    // Select devices
        sycl::queue gpu_q{sycl::gpu_selector{}};
        sycl::queue cpu_q{sycl::cpu_selector{}};
        q_used = gpu_q; 

        flat_data = sycl::malloc_shared<char>(datadev.first.size() + 1, gpu_q);
        std::memcpy(flat_data, datadev.first.data(), datadev.first.size() + 1);

        mappedwm = sycl::malloc_shared<GPU::Mapped>(setsize, gpu_q);
        for (size_t i = 0; i < setsize; ++i) {
            mappedwm[i].v = 0;
            std::memset(mappedwm[i].word, 0, sizeof(mappedwm[i].word));
        }

        GPU::Map mapf(flat_data, N, k);
        mapf.mappedw = mappedwm;
        run();
        mapf.runkernel(gpu_q,localsize);
        gpu_q.wait();
        stop();
        size_t lssc = std::atoi(argv[4]);
        localsize=lssc;
        CPU::Reduce reducef(reinterpret_cast<CPU::Mapped*>(mappedwm), setsize); //N
        run();
        reducef.runkernel(result, cpu_q,localsize);
        cpu_q.wait();
        stop();
        print_mapped_counts(mappedwm,setsize, k);
        }

   // int total_unique = 0;
   // for (size_t i = 0; i < setsize; ++i) if (mappedwm[i].v > 0) total_unique += mappedwm[i].v;
    std::cout<<"check output.txt in /home/user/project/output.txt"<<std::endl;
    sycl::free(flat_data, q);
    sycl::free(mappedwm, q);
    sycl::free(result, q); 
}

//gpu device
//cpu doesnt matter
//hybrid shared
//better way to measure time using run command in cpp and then use time in some way

