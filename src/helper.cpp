#include <filesystem>
#include <CL/sycl.hpp>
#include <sstream>
#include <string>
#include <iostream>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <tuple>
#include <cstdlib> 
#include "helper.h"
namespace fs = std::filesystem;
namespace sycl = cl::sycl;

// bitna napomena je da program sadrzi tekst koji se stampa na konzolu radi lakseg pracenja i debagovanja
std::vector<std::string> prepare() {
    std::string modified_dir = std::string(getenv("HOME")) + "/project/dataset/modified";

    std::vector<std::string> file_contents;

    for (const auto& entry : fs::directory_iterator(modified_dir)) {
        if (entry.path().extension() == ".txt") {
            std::ifstream infile(entry.path());
            if (!infile) {
                std::cerr << "Error opening " << entry.path() << "\n";
                continue;
            }

            std::string content((std::istreambuf_iterator<char>(infile)),
                                 std::istreambuf_iterator<char>());

            file_contents.push_back(content);
        }
    }



    return file_contents;
}
//gpt carries the input issues
void init(){

    int status;

    status = std::system("~/project/scripts/decompressor.sh");
    if (status != 0) {
        std::cerr << "Error running decompressor.sh\n";
    }

    status = std::system("~/project/scripts/modifier.sh");
    if (status != 0) {
        std::cerr << "Error running modifier.sh\n";
    }

}


std::vector<std::string> dataset_selector(std::vector<std::string>& data) {
    std::cout << "Dataset size is " << data.size()
              << ". Enter indices one per line (0 to " << data.size()-1 
              << "), type -1 to finish. Press Enter to skip and use full dataset.\n";

    std::vector<std::string> selected_data;
    std::string line;

    while (true) {
        std::cout << "Enter index: ";
        if (!std::getline(std::cin, line) || line.empty()) {
            std::cout << "No input provided. Using full dataset.\n";
            return data; // safe fallback
        }

        try {
            long long idx = std::stoll(line);
            if (idx == -1) break; // finished
            if (idx >= 0 && idx < static_cast<long long>(data.size())) {
                selected_data.push_back(data[idx]);
                std::cout << "Added: \"" << data[idx] << "\"\n";
            } else {
                std::cout << "Index out of bounds, ignoring.\n";
            }
        } catch (...) {
            std::cout << "Invalid input, ignoring. Using full dataset if nothing selected.\n";
        }
    }

    if (selected_data.empty()) {
        std::cout << "No valid indices entered. Using full dataset.\n";
        return data; // safe fallback
    }

    return selected_data;
}

std::tuple<sycl::device, int> Program_device_selector() {
    std::cout << "\nSelect device:\n"
              << "1 - GPU\n2 - CPU\n3 - Hybrid (GPU map + CPU reduce)\n"
              << "Press Enter to use default (CPU).\n";

    std::string line;
    if (!std::getline(std::cin, line) || line.empty()) {
        std::cout << "No input provided. Using CPU by default.\n";
        return {sycl::device(sycl::cpu_selector_v), 2};
    }

    try {
        int choice = std::stoi(line);
        switch (choice) {
            case 1: return {sycl::device(sycl::gpu_selector_v), 1};
            case 2: return {sycl::device(sycl::cpu_selector_v), 2};
            case 3: return {sycl::device(sycl::gpu_selector_v), 3};
            default:
                std::cout << "Invalid choice. Using CPU by default.\n";
                return {sycl::device(sycl::cpu_selector_v), 1};
        }
    } catch (...) {
        std::cout << "Invalid input. Using CPU by default.\n";
        return {sycl::device(sycl::cpu_selector_v), 1};
    }
}