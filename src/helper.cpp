#include <filesystem>
#include <CL/sycl.hpp>
#include <sstream>
#include <string>
#include <fstream>
#include <iostream>
#include <thread>
#include <chrono>
#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>
#include <cstdlib>
#include <limits>
#include <nvml.h> //gpu measure
#include <unistd.h> // for isatty, fileno (POSIX) and cpu measure
#include "helper.h"
namespace fs = std::filesystem;
namespace sycl = cl::sycl;

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


sycl::device Program_device_selector(int dev) {
    int default_val = 2;  // CPU
    ///gdb here


    try {
        switch (dev) {
            case 1: // GPU
                return sycl::device(sycl::gpu_selector_v);
            case 2: // CPU
                return sycl::device(sycl::cpu_selector_v);
            case 3: // Hybrid (GPU map + CPU reduce)
                return sycl::device(sycl::gpu_selector_v);  // or GPU as main device
            default:
                std::cerr << "Invalid 'dev' value (" << dev << "). Using CPU.\n";
                return sycl::device(sycl::cpu_selector_v);
        }
    } catch (...) {
        std::cerr << "Failed to parse 'dev'. Using CPU.\n";
        return sycl::device(sycl::cpu_selector_v);
    }







    //CPU monitor
class CpuMonitor {
public:
    CpuMonitor() : prevIdle(0), prevTotal(0), snapIdle(0), snapTotal(0) {}

    // rolling instant percent (keeps prev samples)
    double getUtil() { return getCpuUsage(); }

    // snapshot/delta for START/STOP windows
    void snapshotStart() { std::tie(snapTotal, snapIdle) = readCpuTimes(); }
    double getUtilSinceSnapshot() {
        auto [nowTotal, nowIdle] = readCpuTimes();
        long totald = nowTotal - snapTotal;
        long idled  = nowIdle - snapIdle;
        // update snapshot to avoid double-counting if called multiple times
        snapTotal = nowTotal;
        snapIdle  = nowIdle;
        if (totald <= 0) return 0.0;
        return (double)(totald - idled) * 100.0 / (double)totald;
    }

    int getMemMB()  { return getMemUsedMB(); }
    int getTemp()   { return getTempC(); }

private:
    long prevIdle, prevTotal;
    long snapIdle, snapTotal;

    std::pair<long,long> readCpuTimes() {
        std::ifstream f("/proc/stat");
        if (!f) return {0,0};
        std::string cpu;
        long user=0, nice=0, system=0, idle=0, iowait=0, irq=0, softirq=0, steal=0;
        if (!(f >> cpu >> user >> nice >> system >> idle >> iowait >> irq >> softirq >> steal))
            return {0,0};
        long idleAll = idle + iowait;
        long total = user + nice + system + idle + iowait + irq + softirq + steal;
        return { total, idleAll };
    }

    double getCpuUsage() {
        auto [total, idleAll] = readCpuTimes();
        long totald = total - prevTotal;
        long idled  = idleAll - prevIdle;
        prevTotal = total;
        prevIdle  = idleAll;
        if (totald == 0) return 0.0;
        return (double)(totald - idled) * 100.0 / (double)totald;
    }

    int getMemUsedMB() {
        std::ifstream f("/proc/meminfo");
        if (!f) return 0;
        std::string key; int val;
        int memTotal=0, memAvail=0;
        while (f >> key >> val) {
            if (key == "MemTotal:") memTotal = val;
            else if (key == "MemAvailable:") memAvail = val;
            if (memTotal && memAvail) break;
        }
        if (memTotal == 0) return 0;
        return (memTotal - memAvail) / 1024;
    }

    int getTempC() {
        std::ifstream f("/sys/class/hwmon/hwmon0/temp1_input");
        if (!f) return 0;
        int t=0;
        if (!(f >> t)) return 0;
        return t / 1000;
    }
};

class GpuMonitor {
public:
    GpuMonitor() {
        nvmlReturn_t r = nvmlInit();
        if (r != NVML_SUCCESS) throw std::runtime_error("NVML init failed");
        if (nvmlDeviceGetHandleByIndex(0, &device) != NVML_SUCCESS)
            throw std::runtime_error("NVML get handle failed");
    }
    ~GpuMonitor() { nvmlShutdown(); }

    unsigned int getUtil() {
        nvmlUtilization_t util{};
        if (nvmlDeviceGetUtilizationRates(device, &util) == NVML_SUCCESS) return util.gpu;
        return 0;
    }
    unsigned int getMemMB() {
        nvmlMemory_t mem{};
        if (nvmlDeviceGetMemoryInfo(device, &mem) == NVML_SUCCESS) return static_cast<unsigned int>(mem.used / (1024*1024));
        return 0;
    }
    unsigned int getTempC() {
        unsigned int t=0;
        if (nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &t) == NVML_SUCCESS) return t;
        return 0;
    }

private:
    nvmlDevice_t device;
};

class SyclProfiler {
public:
    SyclProfiler() = default;

    void setKernelEvent(const sycl::event& e) { kernelEvent = e; }

    void Metric(const std::string& signal, int metricIndex, bool isCPU = false) {
        if (signal == "START") {
            running = true;
            if (isCPU) {
                cpuStart = std::chrono::high_resolution_clock::now();
            } else {
                // GPU: rely on SYCL event, but keep CPU snapshot for utilization
                cpu.snapshotStart();
            }
        } else if (signal == "STOP") {
            running = false;
            double value = sampleMetric(metricIndex, isCPU);
            sums[metricIndex] += value;
        }
    }

    double getSum(int metricIndex) const {
        auto it = sums.find(metricIndex);
        return (it == sums.end()) ? 0.0 : it->second;
    }

    void resetSums() { sums.clear(); }

private:
    bool running = false;
    sycl::event kernelEvent;

    CpuMonitor cpu;
    GpuMonitor gpu;

    std::chrono::high_resolution_clock::time_point cpuStart;
    std::unordered_map<int,double> sums;

    double sampleMetric(int metricIndex, bool isCPU) {
        switch (metricIndex) {
            case 0: // kernel time
                return isCPU ? getKernelTimeMsCPU() : getKernelTimeMsGPU(kernelEvent);
            case 1: // GPU Util (%)
                return static_cast<double>(gpu.getUtil());
            case 2: // CPU Util (%) over last START->STOP
                return cpu.getUtilSinceSnapshot();
            case 3: // GPU Mem (MB)
                return static_cast<double>(gpu.getMemMB());
            case 4: // CPU Mem (MB)
                return static_cast<double>(cpu.getMemMB());
            case 5: // GPU Temp (C)
                return static_cast<double>(gpu.getTempC());
            case 6: // CPU Temp (C)
                return static_cast<double>(cpu.getTemp());
            default:
                return 0.0;
        }
    }

    // GPU kernel timing from SYCL event
    double getKernelTimeMsGPU(const sycl::event &e) {
        auto start = e.get_profiling_info<sycl::info::event_profiling::command_start>();
        auto end   = e.get_profiling_info<sycl::info::event_profiling::command_end>();
        return (end - start) * 1e-6; // ns → ms
    }

    // CPU kernel timing using chrono
    double getKernelTimeMsCPU() {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - cpuStart).count();
    }
};}