__global__ void DefaultKernel(float* d_1, float* d_2, float* d_result, long size){
    long id = blockDim.x * blockIdx.x + threadIdx.x;
    if(id < size){
        float a = d_1[id];
        float b = d_2[id];
        if(a >= b){
            d_result[id] = 1.0f;
        }
        else{
            d_result[id] = 0.0f;
        }
    }
}