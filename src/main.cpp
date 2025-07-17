#include <CL/sycl.hpp>
#include <iostream>
#include <vector>
#include <string>
#include "helper.h"
#include "GPU.h"

namespace sycl = cl::sycl;
char** convert(std::vector<std::string> working_data){
        char** cstrings = (char**)malloc(working_data.size() * sizeof(char*));
    if (!cstrings) return nullptr;

    for (size_t i = 0; i < working_data.size(); ++i) {
        cstrings[i] = (char*)malloc(working_data[i].size() + 1);
        if (!cstrings[i]) {
            for (size_t j = 0; j < i; ++j) {
                free(cstrings[j]);
            }
            free(cstrings);
            return nullptr;
        }
        std::strcpy(cstrings[i], working_data[i].c_str());
    }
    return cstrings; 
}
void freeCharArray(char** cstrings, size_t count) {
    if (!cstrings) return;
    for (size_t i = 0; i < count; ++i) {
        free(cstrings[i]);
    }
    free(cstrings);
}
int main() {
   
   init();
   std::vector<std::string> datav= prepare();
   std::cout << "Finished preparing!" << std::endl;
   std::vector< std::string> dataset_used = dataset_selector(datav);
    std::tuple<sycl::device,int> dev = Program_device_selector();

    sycl::queue q{std::get<0>(dev)};

    std::cout << "Running on "
              << q.get_device().get_info<sycl::info::device::name>()<<"\n" ; 
        
     if ( std::get<1>(dev)==1){
            //cpu
     }
     else if ( std::get<2>(dev)==2){
        char ** datadev = convert(dataset_used);
        int k=3;
        Map mapk(datadev, 512,q);
        
        mapk.runkernel();

     }else{
        //hib
     }        
   //ako je gpu onda gpu header
   //analogno za cpu i hibrid
    return 0;
}
