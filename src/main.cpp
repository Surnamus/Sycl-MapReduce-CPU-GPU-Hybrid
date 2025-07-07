#include <CL/sycl.hpp>
#include <iostream>
#include <vector>
#include <string>
#include "helper.h"
#include "GPU.h"

namespace sycl = cl::sycl;

int main() {
   
   init();
   std::vector<std::string> datav= prepare();
   std::cout << "Finished preparing!" << std::endl;
   std::vector< std::string> dataset_used = dataset_selector(datav);
    std::tuple<sycl::device,int> dev = Program_device_selector();

    sycl::queue q{std::get<0>(dev)};

    std::cout << "Running on "
              << q.get_device().get_info<sycl::info::device::name>()<<"\n" ; 
        
             
   //ako je gpu onda gpu header
   //analogno za cpu i hibrid
//example kernel koji je gpt napisao samo da mi posluzi da istestiram kompaliranje u 2.30 ujutru kad nisam mogao da pisem, nista duboko.
 /*   int data = 0;

    {
        sycl::buffer<int, 1> buf(&data, sycl::range<1>(1));

        q.submit([&](sycl::handler& h) {
            auto acc = buf.get_access<sycl::access::mode::write>(h);
            h.single_task([=]() {
                acc[0] = 42; // do some simple work
            });
        });
    }

    std::cout << "Kernel wrote: " << data << std::endl;
*/
    return 0;
}
