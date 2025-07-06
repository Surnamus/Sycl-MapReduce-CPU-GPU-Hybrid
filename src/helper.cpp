// uzmi podatke iz uncompresed i makni delove i predstavi ga u text
//.fan->text
//pozeljno da se sejva u modified
//main cpp poziva helper i output toga se koristi za gpu, cpu, hyb
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <cstdlib> 
#include "helper.h"
namespace fs = std::filesystem;

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
void init(){

    int status;

    // Run the unzip script
    status = std::system("~/project/compressed/decompressor.sh");
    if (status != 0) {
        std::cerr << "Error running decompressor.sh\n";
    }

    // Run the .fna → .txt processor script
    status = std::system("~/project/uncompressed/modifier.sh");
    if (status != 0) {
        std::cerr << "Error running modifier.sh\n";
    }

}