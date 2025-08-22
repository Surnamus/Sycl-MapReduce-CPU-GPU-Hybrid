#include <filesystem>
#include <CL/sycl.hpp>
#include <sstream>
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>
#include <cstdlib>
#include <limits>
#include <unistd.h> // for isatty, fileno (POSIX)
#include "helper.h"

namespace fs = std::filesystem;
namespace sycl = cl::sycl;

// Run external scripts safely without consuming stdin
void init() {
    int status = std::system("~/project/scripts/decompressor.sh < /dev/null");
    if (status != 0) std::cerr << "Error running decompressor.sh\n";

    status = std::system("~/project/scripts/modifier.sh < /dev/null");
    if (status != 0) std::cerr << "Error running modifier.sh\n";
}

// Load all .txt files from modified directory
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



std::vector<std::string> dataset_selector(std::vector<std::string>& data) {
    std::vector<std::string> selected_data;
    std::string line;
    std::cout << "Dataset size is " << data.size()
              << ". Enter indices one per line (0 to " 
              << (data.size() ? data.size()-1 : 0)
              << "), type -1 to finish. Press Enter to skip and use full dataset.\n";

    // Only skip if stdin is non-interactive
    if (!isatty(fileno(stdin))) {
        std::cout << "Non-interactive stdin detected. Using full dataset.\n";
        return data;
    }

    while (true) {
        std::cout << "Enter index: " << std::flush;
        if (!std::getline(std::cin, line)) { // EOF or error
            std::cout << "\nNo input provided (EOF). Using full dataset.\n";
            return data;
        }

        if (line.empty()) {
            std::cout << "No input provided. Using full dataset.\n";
            return data; // fallback to full dataset
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
        } catch (const std::exception& e) {
            std::cout << "Invalid input, ignoring.\n";
        }
    }

    if (selected_data.empty()) {
        std::cout << "No valid indices entered. Using full dataset.\n";
        return data;
    }

    return selected_data;
}

std::tuple<sycl::device, int> Program_device_selector() {
    int default_val = 2;  // CPU
    ///gdb here
    const char* val = std::getenv("device");

    if (!val) {
        return {sycl::device(sycl::cpu_selector_v), default_val};
    }

    try {
        int choice = std::stoi(val);
        switch (choice) {
            case 1: // GPU
                return {sycl::device(sycl::gpu_selector_v), 1};
            case 2: // CPU
                return {sycl::device(sycl::cpu_selector_v), 2};
            case 3: // Hybrid (GPU map + CPU reduce)
                return {sycl::device(sycl::gpu_selector_v), 3};  // or GPU as main device
            default:
                std::cerr << "Invalid 'dev' value (" << choice << "). Using CPU.\n";
                return {sycl::device(sycl::cpu_selector_v), default_val};
        }
    } catch (...) {
        std::cerr << "Failed to parse 'dev'. Using CPU.\n";
        return {sycl::device(sycl::cpu_selector_v), default_val};
    }
}