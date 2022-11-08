__global__ void BranchlessKernel(float* d_1, float* d_2, float* d_result, long size){
    long id = blockDim.x * blockIdx.x + threadIdx.x;
    if(id < size){
        float diff = d_1[id] - d_2[id];
        unsigned int ui = *((unsigned int *)&diff); // Read bits as unsigned int
        d_result[id] = (float)(1 - (ui >> (sizeof(float) * 8 - 1))); // Shift sign bit and cast to float
    }
}