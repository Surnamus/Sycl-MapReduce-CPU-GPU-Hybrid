#include <CL/sycl.hpp>
#include <iostream>
#include <vector>
#include <string>
#include "helper.h"

namespace sycl = cl::sycl;

int main() {
   
   init();
   std::vector<std::string> datav= prepare();
   std::cout << "Finished preparing!" << std::endl;

    sycl::queue q;

    std::cout << "Running on "
              << q.get_device().get_info<sycl::info::device::name>()
              << std::endl;

    int data = 0;

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

    return 0;
}
