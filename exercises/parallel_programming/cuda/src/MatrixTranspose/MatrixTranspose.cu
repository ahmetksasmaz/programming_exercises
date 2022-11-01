#include <iostream>
#include <string>
#include "../../include/CudaMacros.cuh"
#include "../../include/Logger.h"
#include "MatrixTransposeDefaultKernel.cuh"

int main(int argc, char *argv[]){

    // Assumed that arguments are given correctly | ./<program-name> <width> <height>

    char *p;
    long matrix_width = strtol(argv[1], &p, 10);
    long matrix_height = strtol(argv[2], &p, 10);
    long matrix_memory_size = matrix_height * matrix_width * sizeof(float);

    float * h_matrix = nullptr;
    float * h_matrix_ground_truth = nullptr;
    float * h_matrix_result = nullptr;
    float * d_matrix = nullptr;
    float * d_matrix_result = nullptr;

    h_matrix = (float *)malloc(matrix_memory_size);
    if(!h_matrix){
        LoggerPrint("Host matrix memory couldn't be allocated.", LogLevel::FATAL);
        exit(-1);
    }
    h_matrix_ground_truth = (float *)malloc(matrix_memory_size);
    if(!h_matrix_ground_truth){
        LoggerPrint("Host ground truth matrix memory couldn't be allocated.", LogLevel::FATAL);
        free(h_matrix);
        exit(-1);
    }
    h_matrix_result = (float *)malloc(matrix_memory_size);
    if(!h_matrix_result){
        LoggerPrint("Host result matrix memory couldn't be allocated.", LogLevel::FATAL);
        free(h_matrix);
        free(h_matrix_ground_truth);
        exit(-1);
    }
    GpuErrChk(cudaMalloc(&d_matrix, matrix_memory_size));
    if(!d_matrix){
        LoggerPrint("Device matrix memory couldn't be allocated.", LogLevel::FATAL);
        free(h_matrix);
        free(h_matrix_ground_truth);
        free(h_matrix_result);
        exit(-1);
    }
    GpuErrChk(cudaMalloc(&d_matrix_result, matrix_memory_size));
    if(!d_matrix_result){
        LoggerPrint("Device result matrix memory couldn't be allocated.", LogLevel::FATAL);
        free(h_matrix);
        free(h_matrix_ground_truth);
        free(h_matrix_result);
        cudaFree(d_matrix);
        exit(-1);
    }

    LoggerPrint("Matrix\n", LogLevel::DEBUG);
    for(int i = 0; i < matrix_height; i++){
        for(int j = 0; j < matrix_width; j++){
            h_matrix[i*matrix_width + j] = i*matrix_width + j;
            LoggerPrint(std::to_string(i*matrix_width + j) + " ", LogLevel::DEBUG);
        }
        LoggerPrint("\n", LogLevel::DEBUG);
    }

    LoggerPrint("Ground Truth Matrix\n", LogLevel::DEBUG);
    for(int i = 0; i < matrix_width; i++){
        for(int j = 0; j < matrix_height; j++){
            h_matrix_ground_truth[i*matrix_height + j] = j*matrix_width + i;
            LoggerPrint(std::to_string(j*matrix_width + i) + " ", LogLevel::DEBUG);
        }
        LoggerPrint("\n", LogLevel::DEBUG);
    }

    GpuErrChk(cudaMemcpy(d_matrix, h_matrix, matrix_memory_size, cudaMemcpyHostToDevice));

    dim3 grid, block;
    block.x = 32;
    block.y = 32;
    grid.x  = matrix_width / block.x + ( matrix_width % block.x ? 1 : 0 );
    grid.y  = matrix_height / block.y + ( matrix_height % block.y ? 1 : 0 );
    MatrixTransposeDefault<<<grid, block>>>(d_matrix, d_matrix_result, matrix_width, matrix_height);

    GpuErrChk(cudaGetLastError());

    GpuErrChk(cudaMemcpy(h_matrix_result, d_matrix_result, matrix_memory_size, cudaMemcpyDeviceToHost));

    for(int i = 0; i < matrix_width * matrix_height; i++){
        if(h_matrix_result[i] != h_matrix_ground_truth[i]){
            LoggerPrint("Matrix transpose is erroneous at : [" + std::to_string(i/matrix_height) + "," + std::to_string(i%matrix_height) + "]\n", LogLevel::FATAL);
            free(h_matrix);
            free(h_matrix_ground_truth);
            free(h_matrix_result);
            cudaFree(d_matrix);
            cudaFree(d_matrix_result);
            exit(-1);
        }
    }
    LoggerPrint("Matrix transpose is successful.\n", LogLevel::INFO);
}