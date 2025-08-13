#include <CL/sycl.hpp>
#include <iostream>
#include <vector>
#include <string>
#include "helper.h"
#include "GPU.h"

namespace sycl = cl::sycl;
std::pair<std::string,std::vector<size_t>> convert(std::vector<std::string> strings){
        std::vector<size_t> offsets;
std::string flattened;

for (const auto& s : strings) {
    offsets.push_back(flattened.size()); 
    flattened += s;
    flattened += '\0'; 
}
return {flattened,offsets};
        // to convert to char* = std::string:c_str(flattened)
}
int main() {
   
   init();
   std::vector<std::string> datav= prepare();
   std::cout << "Finished preparing!" << std::endl;
   std::vector< std::string> dataset_used = dataset_selector(datav);
    std::tuple<sycl::device,int> dev = Program_device_selector();

    sycl::queue q{std::get<0>(dev)};
   
    std::pair<std::string,std::vector<size_t>> datadev = convert(dataset_used);
        char* flat_data = sycl::malloc_shared<char>(datadev.first.size(), q);
        std::memcpy(flat_data, datadev.first.data(), datadev.first.size() + 1); 
        int k=3;


    std::cout << "Running on "
              << q.get_device().get_info<sycl::info::device::name>()<<"\n" ; 
        
     if ( std::get<1>(dev)==1){
      //here it can be used to change device selectors gpu
     }
     else if ( std::get<2>(dev)==2){
      
            //here it can be used to change device selectors cpu


     }else{
      //here it can be used to change device selectors hybrid
     }        
   //ako je gpu onda gpu header
   //analogno za cpu i hibrid
    return 0;
}
