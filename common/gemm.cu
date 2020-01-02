#include "gemm.h"
#include "def.h"

#define block_size 32
#define grid_size 8

__global__ void axpy_(unsigned m_size, 
                       const float alpha, 
                       float *X, unsigned incx, 
                       float *Y, unsigned incy) {

    unsigned index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    Y[index * incy] += alpha * X[index * incx];
}
void _axpy_(unsigned m_size, 
            const float alpha, 
            float *X, unsigned incx, 
            float *Y, unsigned incy) {

    dim3 cuda_griddim = {(m_size-1) / BLOCK_SIZE + 1, 1, 1};
    axpy_<<<cuda_griddim, BLOCK_SIZE>>>(m_size, alpha, X, incx, Y, incy);
}

__global__ void scal_(unsigned m_size, 
                       const float alpha, 
                       float *X, unsigned incx){
    unsigned index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    X[index * incx] *= alpha;
}

void _scal_(unsigned m_size, const float alpha, float *X, unsigned incx){
    dim3 cuda_griddim = {(m_size - 1) / BLOCK_SIZE + 1, 1, 1};
    scal_<<<cuda_griddim, BLOCK_SIZE>>>(m_size, alpha, X, incx);
}

__global__ void _gemm_nn_(int M, int N, int K, float ALPHA,
                          float *A, int lda,
                          float *B, int ldb,
                          float BETA,
                          float *C, int ldc) {

    unsigned row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned col = blockIdx.x * blockDim.x + threadIdx.x;

    while(row < M) {
        while(col < N) {
            C[col * M + row] *= BETA;
            for(int i = 0; i < K; i++) {
                C[col * M + row] += ALPHA * A[i * M + row] * B[col * K + i];
            }
            col += block_size * grid_size;
        }
        col = blockIdx.x * blockDim.x + threadIdx.x;
        row += block_size * grid_size;
       //row += block_size * grid_size;
    }
}

__global__ void _gemm_tn_(int M, int N, int K, float ALPHA,
                          float *A, int lda,
                          float *B, int ldb,
                          float BETA,
                          float *C, int ldc) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    while(row < M){
        while (col < N){
            C[row * N + col] *= BETA;
            for(int i = 0; i < K; i++) {
                C[col * M + row] += ALPHA * A[row * K + i] * B[col * K + i];
            }
            col += block_size * grid_size;
        }
        col = blockIdx.x * blockDim.x + threadIdx.x;
        row += block_size * grid_size;
        //row += block_size * grid_size;
    }
}

__global__ void _gemm_nt_(int M, int N, int K, float ALPHA,
                          float *A, int lda,
                          float *B, int ldb,
                          float BETA,
                          float *C, int ldc) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    while(row < M) {
        while (col < N) {
            C[col * M + row] *= BETA;
            for(int i = 0; i < K; i++) {
                C[col * M + row] += ALPHA * A[i * M + row] * B[i * N + col];
            }
            col += block_size * grid_size;
        }
        col = blockIdx.x * blockDim.x + threadIdx.x;
        row += block_size * grid_size;
        //row += block_size * grid_size;
    }
}

//still working
__global__ void _gemm_tt_(int M, int N, int K, float ALPHA,
                          float *A, int lda,
                          float *B, int ldb,
                          float BETA,
                          float *C, int ldc){

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    float sum = 0.0;
    while(row < M){
        while(col < N) {
            C[row * ldc + col] *= BETA;
            for(int i = 0; i < K; i++) {
                sum += ALPHA * A[row + i * lda] * B[i + col * ldb];
            }
            C[row * ldc + col] += sum;
            col += 512/block_size;
            //col += block_size;
        }
    row += block_size;
    }
}



extern "C++" void _gemm_(int TA, int TB, int M, int N, int K, float ALPHA,
                         float *A, int lda,
                         float *B, int ldb, float BETA,
                         float *C, int ldc) 
{
    dim3 grid_dim(grid_size, grid_size);
    dim3 block_dim(block_size, block_size);
    //dim3 block_dim(block_size, block_size);

    if(!TA && !TB) {
        _gemm_nn_<<<grid_dim, block_dim>>>(M, N, K, ALPHA, A, lda, B, ldb, BETA, C, ldc);
    }
    else if(TA && !TB) {
        _gemm_tn_<<<grid_dim, block_dim>>>(M, N, K, ALPHA, A, lda, B, ldb, BETA, C, ldc);
    }
    else if(!TA && TB) {
        _gemm_nt_<<<grid_dim, block_dim>>>(M, N, K, ALPHA, A, lda, B, ldb, BETA, C, ldc);
    }
    else {
        _gemm_tt_<<<grid_dim, block_dim>>>(M, N, K, ALPHA, A, lda, B, ldb, BETA, C, ldc);
    }
}
