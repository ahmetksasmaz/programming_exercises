#include <iostream>

int main(){
    
    int numberOfCudaDevices;
    cudaGetDeviceCount(&numberOfCudaDevices);

    for(int i = 0; i < numberOfCudaDevices; i++){
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        std::cout << "Device number : " << i << std::endl;
        std::cout << "Clock frequency : " << prop.clockRate << "hz" << std::endl;
        std::cout << "Concurrent Kernels : " << prop.concurrentKernels << std::endl;
        std::cout << "Async engine count : " << prop.asyncEngineCount << std::endl;
        std::cout << "Kernel execution timeout enabled : " << prop.kernelExecTimeoutEnabled << std::endl;
        std::cout << "Managed memory : " << prop.managedMemory << std::endl;
        std::cout << "Maximum thread per block : " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "Maximum thread per multiprocessor : " << prop.maxThreadsPerMultiProcessor << std::endl;
        std::cout << "Number of multiprocessors : "  << prop.multiProcessorCount << std::endl;
        std::cout << "Registers per block : " << prop.regsPerBlock << std::endl;
        std::cout << "Registers per multiprocessors : " << prop.regsPerMultiprocessor << std::endl;
        std::cout << "Shared memory per multiprocessors : " << prop.sharedMemPerMultiprocessor << std::endl;
        std::cout << "Shared memory per block : " << prop.sharedMemPerBlock << std::endl;
        std::cout << "Warp size : " << prop.warpSize << std::endl;
    }
}