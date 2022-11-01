__global__ void MatrixTransposeDefault(float* d_matrix, float* d_matrix_result, int width, int height){
    int index_x = blockIdx.x*blockDim.x + threadIdx.x;
    int index_y = blockIdx.y*blockDim.y + threadIdx.y;

    if(index_x < width && index_y < height){
        d_matrix_result[index_x*height + index_y] = d_matrix[index_y*width + index_x];
    }
}