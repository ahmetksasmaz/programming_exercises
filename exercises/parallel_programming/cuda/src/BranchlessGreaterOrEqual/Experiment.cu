#include <iostream>
#include <string>
#include "../../include/CudaMacros.cuh"
#include "../../include/Logger.h"
#include "DefaultKernel.cuh"
#include "BranchlessKernel.cuh"

int main(int argc, char *argv[]){

    // Assumed that arguments are given correctly | ./<program-name> <kernel-type>
    // kernel-type : 0 -> default, 1 -> branchless

    char *p;
    bool kernel_type = strtol(argv[1], &p, 10);
    long size = 1 << 25;
    long array_size = size * sizeof(float);

    float * h_1 = nullptr;
    float * h_2 = nullptr;
    float * h_result = nullptr;
    float * h_ground_truth = nullptr;
    float * d_1 = nullptr;
    float * d_2 = nullptr;
    float * d_result = nullptr;

    h_1 = (float *)malloc(array_size);
    if(!h_1){
        LoggerPrint("Host memory 1 couldn't be allocated.", LogLevel::FATAL);
        exit(-1);
    }
    h_2 = (float *)malloc(array_size);
    if(!h_2){
        LoggerPrint("Host memory 2 couldn't be allocated.", LogLevel::FATAL);
        free(h_1);
        exit(-1);
    }
    h_result = (float *)malloc(array_size);
    if(!h_result){
        LoggerPrint("Host memory result couldn't be allocated.", LogLevel::FATAL);
        free(h_1);
        free(h_2);
        exit(-1);
    }
    h_ground_truth = (float *)malloc(array_size);
    if(!h_ground_truth){
        LoggerPrint("Host memory ground truth couldn't be allocated.", LogLevel::FATAL);
        free(h_1);
        free(h_2);
        free(h_result);
        exit(-1);
    }
    GpuErrChk(cudaMalloc(&d_1, array_size));
    if(!d_1){
        LoggerPrint("Device memory 1 couldn't be allocated.", LogLevel::FATAL);
        free(h_1);
        free(h_2);
        free(h_result);
        free(h_ground_truth);
        exit(-1);
    }
    GpuErrChk(cudaMalloc(&d_2, array_size));
    if(!d_2){
        LoggerPrint("Device memory 2 couldn't be allocated.", LogLevel::FATAL);
        free(h_1);
        free(h_2);
        free(h_result);
        free(h_ground_truth);
        cudaFree(d_1);
        exit(-1);
    }
    GpuErrChk(cudaMalloc(&d_result, array_size));
    if(!d_result){
        LoggerPrint("Device memory result couldn't be allocated.", LogLevel::FATAL);
        free(h_1);
        free(h_2);
        free(h_result);
        free(h_ground_truth);
        cudaFree(d_1);
        cudaFree(d_2);
        exit(-1);
    }

    for(int i = 0; i < size; i++){
        // Make it like random numbers, we test warp divergence
        h_1[i] = i % 13;
        h_2[i] = i % 47;
        h_ground_truth[i] = ((i % 13) >= (i % 47)) ? 1.0f : 0.0f ;
    }

    GpuErrChk(cudaMemcpy(d_1, h_1, array_size, cudaMemcpyHostToDevice));
    GpuErrChk(cudaMemcpy(d_2, h_2, array_size, cudaMemcpyHostToDevice));

    dim3 grid, block;
    block.x = 32;
    grid.x  = size / block.x + ( size % block.x ? 1 : 0 );
    if(kernel_type){
        BranchlessKernel<<<grid, block>>>(d_1, d_2, d_result, size);
    }
    else{
        DefaultKernel<<<grid, block>>>(d_1, d_2, d_result, size);
    }

    GpuErrChk(cudaGetLastError());

    GpuErrChk(cudaMemcpy(h_result, d_result, array_size, cudaMemcpyDeviceToHost));

    for(int i = 0; i < size; i++){
        if(h_result[i] != h_ground_truth[i]){
            LoggerPrint("Greater or equal check is erroneous at : [" + std::to_string(i) + "]\n", LogLevel::FATAL);
            free(h_1);
            free(h_2);
            free(h_result);
            free(h_ground_truth);
            cudaFree(d_1);
            cudaFree(d_2);
            cudaFree(d_result);
            exit(-1);
        }
    }
    LoggerPrint("Greater or equal check is successful.\n", LogLevel::INFO);
}