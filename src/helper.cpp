// uzmi podatke iz uncompresed i makni delove i predstavi ga u text
//.fan->text
//pozeljno da se sejva u modified
//main cpp poziva helper i output toga se koristi za gpu, cpu, hyb
#include <filesystem>
#include <CL/sycl.hpp>
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
void init(){

    int status;

    // Run the unzip script
    status = std::system("~/project/scripts/decompressor.sh");
    if (status != 0) {
        std::cerr << "Error running decompressor.sh\n";
    }

    // Run the .fna → .txt processor script
    status = std::system("~/project/scripts/modifier.sh");
    if (status != 0) {
        std::cerr << "Error running modifier.sh\n";
    }

}
std::vector<std::string> dataset_selector(std::vector<std::string> data) {
    std::cout<<"Dataset size is " << data.size()<<"." << " Type the number -1 if you wish to abort the input.";
int u;
std::vector<std::string> selected_data={};
while (true){
std::cin>>u;
if ( u < data.size() && u>=0){
selected_data.push_back(data[u]);

} else if (!(u < data.size() && u>=0) && u!=-1) {
    std::cout<<"Selected index is out of bounds. Dataset size is " << data.size()<<"\n";
} else if ( u==-1){
        std::cout<<"Aborted. Selected dataset size is now " << selected_data.size()<<"\n";
        break;
}

}
return selected_data;
}
std::tuple<sycl::device,int> Program_device_selector(){
    int n;
    std::cout<<"1-CPU, 2-GPU, 3-Hybrid ";
    //ne treba exeption jer znamo da se radi o cpu i gpu sistemu gde su oba dostupna
   while (true){
    std::cin>>n;

      if (n==1){      sycl::device dev =sycl::device(sycl::cpu_selector_v);
            return {dev,n};
        
      }
    else if ( n==2){        sycl::device dev =sycl::device(sycl::gpu_selector_v);
            return {dev,n};
    }
      else if (n==3){      sycl::device dev =sycl::device(sycl::gpu_selector_v);
        //yvati header 
            return {dev,n};  
      }
        else{ std::cout<<"Not a valid number"<<"\n";
            std::cout<<"Select a number from 1 to 3"<<"\n";

        }
        }
        sycl::device dev =sycl::device(sycl::default_selector_v);
        //yvati header 
            return {dev,n}; 
        
    }


