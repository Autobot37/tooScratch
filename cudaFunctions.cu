//_____________CUDA_MAT_ADD__________________
//https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#notices
__global__ void MatAdd(float A[N][N], float B[N][N], float C[N][N]){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;

    if(i<N && j<N){
        C[i][j] = A[i][j] + B[i][j];
    }
}

int main(){

    dim3 threadPerBlock(16,16);
    dim3 numBlocks(N/threadPerBlock.x, N/threadPerBlock.y);
    MatAdd<<<numBlocks,threadPerBlock>>>(A,B,C);


}