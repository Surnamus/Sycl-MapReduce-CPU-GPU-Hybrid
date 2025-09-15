#ifndef HELPER_H
#define HELPER_H

#include <CL/sycl.hpp>
#include <nvml.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <tuple>
#include <unordered_map>

namespace sycl = cl::sycl;
namespace fs = std::filesystem;

std::vector<std::string> prepare();
void init();
std::vector<std::string> dataset_selector(std::vector<std::string>& data);
sycl::device Program_device_selector(int dev);

class CpuMonitor {
public:
    CpuMonitor() : prevIdle(0), prevTotal(0), snapIdle(0), snapTotal(0) {}

    double getUtil() { return getCpuUsage(); }
    int getMemMB() { return getMemUsedMB(); }
    int getTemp() { return getTempC(); }

    void snapshotStart() { std::tie(snapTotal, snapIdle) = readCpuTimes(); }
    double getUtilSinceSnapshot() {
        auto [nowTotal, nowIdle] = readCpuTimes();
        long totald = nowTotal - snapTotal;
        long idled  = nowIdle - snapIdle;
        snapTotal = nowTotal;
        snapIdle  = nowIdle;
        if (totald <= 0) return 0.0;
        return (double)(totald - idled) * 100.0 / (double)totald;
    }

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
        std::ifstream f("/sys/class/thermal/thermal_zone0/temp");
        if (!f) return 0;
        int t=0;
        if (!(f >> t)) return 0;
        return t / 1000;
    }
};

class GpuMonitor {
public:
    GpuMonitor() {
        if (nvmlInit() != NVML_SUCCESS) throw std::runtime_error("NVML init failed");
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
        if (nvmlDeviceGetMemoryInfo(device, &mem) == NVML_SUCCESS)
            return static_cast<unsigned int>(mem.used / (1024*1024));
        return 0;
    }

    unsigned int getTempC() {
        unsigned int t=0;
        if (nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &t) == NVML_SUCCESS)
            return t;
        return 0;
    }

private:
    nvmlDevice_t device;
};

class SyclProfiler {
public:
    SyclProfiler() = default;

    void setKernelEvent(const sycl::event& e) { kernelEvent = e; }

    void Metric(const std::string& signal, int metricIndex) {
        if (signal == "START") {
            running = true;
            cpu.snapshotStart();
            return;
        }
        if (signal == "STOP") {
            running = false;
            double value = sampleMetric(metricIndex);
            sums[metricIndex] += value; 
            std::cout << value << std::endl;
            std::cout.flush();
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
    std::unordered_map<int,double> sums;

    double sampleMetric(int metricIndex) {
        switch(metricIndex){
            case 0: return getKernelTimeMs(kernelEvent);
            case 1: return static_cast<double>(gpu.getUtil());
            case 2: return cpu.getUtilSinceSnapshot();
            case 3: return static_cast<double>(gpu.getMemMB());
            case 4: return static_cast<double>(cpu.getMemMB());
            case 5: return static_cast<double>(gpu.getTempC());
            case 6: return static_cast<double>(cpu.getTemp());
            default: return 0.0;
        }
    }

    double getKernelTimeMs(const sycl::event& e) {
        auto start = e.get_profiling_info<sycl::info::event_profiling::command_start>();
        auto end   = e.get_profiling_info<sycl::info::event_profiling::command_end>();
        return (end - start) * 1e-6; // ns -> ms
    }
};

#endif // HELPER_H
