#include "gpu_memory.h"

#include <cuda_runtime.h>
#include <dlfcn.h>
#include <nvml.h>
#include <unistd.h>

#include <vector>

namespace {

constexpr double mib = 1024.0 * 1024.0;

struct process_reading {
    std::string unavailable; // empty when self_mib is valid
    double self_mib = -1;
    int other_processes = -1;
    double other_mib = -1;
};

// NVML ships with the driver, not the toolkit, so it is loaded at run time: a
// CUDA build must still start, and run CPU cells, on a node without the driver.
// The nvml.h macros name the versioned symbols (nvmlInit -> nvmlInit_v2).
#define APXCHOL_NVML_SYMBOL_NAME(fn) #fn
#define APXCHOL_NVML_SYMBOL(fn) APXCHOL_NVML_SYMBOL_NAME(fn)

struct nvml_api {
    decltype(&nvmlInit) init = nullptr;
    decltype(&nvmlErrorString) error_string = nullptr;
    decltype(&nvmlDeviceGetHandleByPciBusId) handle_by_bus_id = nullptr;
    decltype(&nvmlDeviceGetPciInfo) pci_info = nullptr;
    decltype(&nvmlDeviceGetMigMode) mig_mode = nullptr;
    decltype(&nvmlDeviceGetComputeRunningProcesses) compute_processes = nullptr;
};
nvml_api g_nvml;

template <class F>
bool load_symbol(void* library, F& fn, const char* name) {
    fn = reinterpret_cast<F>(dlsym(library, name));
    return fn != nullptr;
}

bool load_nvml() {
    void* library = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!library) return false;
#define APXCHOL_LOAD_NVML(field, fn) load_symbol(library, g_nvml.field, APXCHOL_NVML_SYMBOL(fn))
    return APXCHOL_LOAD_NVML(init, nvmlInit) &&
           APXCHOL_LOAD_NVML(error_string, nvmlErrorString) &&
           APXCHOL_LOAD_NVML(handle_by_bus_id, nvmlDeviceGetHandleByPciBusId) &&
           APXCHOL_LOAD_NVML(pci_info, nvmlDeviceGetPciInfo) &&
           APXCHOL_LOAD_NVML(mig_mode, nvmlDeviceGetMigMode) &&
           APXCHOL_LOAD_NVML(compute_processes, nvmlDeviceGetComputeRunningProcesses);
#undef APXCHOL_LOAD_NVML
}

nvmlDevice_t g_device{};
bool g_have_device = false;
gpu_memory_baseline g_baseline;

process_reading read_processes() {
    process_reading out;
    std::vector<nvmlProcessInfo_t> info;
    unsigned count = 0;
    nvmlReturn_t status = NVML_ERROR_INSUFFICIENT_SIZE;
    // The process list can grow between the size query and the read.
    for (int attempt = 0; attempt < 4 && status == NVML_ERROR_INSUFFICIENT_SIZE; ++attempt) {
        info.resize(count + 8);
        count = static_cast<unsigned>(info.size());
        status = g_nvml.compute_processes(g_device, &count, info.data());
    }
    if (status != NVML_SUCCESS) {
        out.unavailable = status == NVML_ERROR_NOT_SUPPORTED ? "not_supported"
                                                             : g_nvml.error_string(status);
        return out;
    }
    const auto self = static_cast<unsigned>(getpid());
    out.other_processes = 0;
    out.other_mib = 0;
    bool listed = false;
    for (unsigned i = 0; i < count; ++i) {
        const bool known = info[i].usedGpuMemory !=
            static_cast<unsigned long long>(NVML_VALUE_NOT_AVAILABLE);
        if (info[i].pid == self) {
            listed = true;
            if (known) out.self_mib = info[i].usedGpuMemory / mib;
        } else {
            ++out.other_processes;
            if (!known) out.other_mib = -1;
            else if (out.other_mib >= 0) out.other_mib += info[i].usedGpuMemory / mib;
        }
    }
    if (!listed) out.unavailable = "pid_not_listed";
    else if (out.self_mib < 0) out.unavailable = "not_available";
    return out;
}

std::string open_device() {
    if (!load_nvml()) return "nvml_library";
    if (g_nvml.init() != NVML_SUCCESS) return "nvml_init";
    int ordinal = 0;
    char cuda_bus[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE] = {};
    if (cudaGetDevice(&ordinal) != cudaSuccess ||
        cudaDeviceGetPCIBusId(cuda_bus, sizeof cuda_bus, ordinal) != cudaSuccess)
        return "cuda_device";
    if (g_nvml.handle_by_bus_id(cuda_bus, &g_device) != NVML_SUCCESS)
        return "nvml_device";
    nvmlPciInfo_t pci{};
    if (g_nvml.pci_info(g_device, &pci) == NVML_SUCCESS) g_baseline.bus_id = pci.busId;
    unsigned current = 0, pending = 0;
    // MIG partitions report per-instance memory that the parent device does not attribute.
    if (g_nvml.mig_mode(g_device, &current, &pending) == NVML_SUCCESS &&
        current == NVML_DEVICE_MIG_ENABLE)
        return "mig";
    g_have_device = true;
    return {};
}

} // namespace

const gpu_memory_baseline& record_gpu_memory_baseline() {
    static const bool recorded = [] {
        g_baseline.unavailable = open_device();
        if (g_have_device) {
            const process_reading now = read_processes();
            g_baseline.unavailable = now.unavailable;
            g_baseline.self_mib = now.self_mib;
            g_baseline.other_processes = now.other_processes;
            g_baseline.other_mib = now.other_mib;
        }
        return true;
    }();
    (void)recorded;
    return g_baseline;
}

double gpu_memory_above_baseline_mib() {
    if (!g_have_device || g_baseline.self_mib < 0) return -1;
    const process_reading now = read_processes();
    return now.self_mib < 0 ? -1 : now.self_mib - g_baseline.self_mib;
}
