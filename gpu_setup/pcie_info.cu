#include <cstdio>
#include <nvml.h>

int main() {
    nvmlReturn_t result;
    nvmlDevice_t device;

    result = nvmlInit_v2();
    if (result != NVML_SUCCESS) {
        printf("Failed to init NVML: %s\n", nvmlErrorString(result));
        return 1;
    }

    result = nvmlDeviceGetHandleByIndex(0, &device);
    if (result != NVML_SUCCESS) {
        printf("Cannot get handle: %s\n", nvmlErrorString(result));
        return 1;
    }

    unsigned int genCurrent, genMax, widthCurrent, widthMax;

    nvmlDeviceGetCurrPcieLinkGeneration(device, &genCurrent);
    nvmlDeviceGetMaxPcieLinkGeneration(device, &genMax);
    nvmlDeviceGetCurrPcieLinkWidth(device, &widthCurrent);
    nvmlDeviceGetMaxPcieLinkWidth(device, &widthMax);

    printf("PCIe link: Gen %u x%u (max Gen %u x%u)\n",
           genCurrent, widthCurrent, genMax, widthMax);

    nvmlShutdown();
    return 0;
}
