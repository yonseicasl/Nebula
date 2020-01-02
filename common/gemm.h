#ifndef __GEMM_H__
#define __GEMM_H__

#include <thread>
#include <vector>
#include <functional>


#ifdef GPU_ENABLED

void _axpy_(unsigned m_size, 
            const float alpha, 
            float *X, unsigned incx,
            float *Y, unsigned incy);

void _scal_(unsigned m_size, 
            const float alpha,
            float *X, unsigned incx);

void _gemm_(int TA, int TB, int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float BETA,
        float *C, int ldc);
#endif

template <typename T>
// Y = a * X 
void axmy(unsigned m_size, const float alpha, T *X, unsigned incx, T *Y, unsigned incy) {
    for(unsigned i = 0; i < m_size; i++) {
        Y[i * incy] *= alpha * X[i * incx];
    }
}

template <typename T>
void axpy(unsigned m_size, const float alpha, T *X, unsigned incx, T *Y, unsigned incy) {
    for(unsigned i = 0; i < m_size; i++) {
        Y[i*incy] += alpha * X[i*incx];
    }
}

template <typename T>
void scal(unsigned m_size, const float alpha, T *X, unsigned incx) {
    for(unsigned i = 0; i < m_size; i++) {
        X[i*incx] *= alpha;
    }
}
// GEMM NN
template <typename T>
void gemm_nn(unsigned begin, unsigned end, unsigned N, unsigned K, float alpha, 
             T *A, unsigned lda, T *B, unsigned ldb, T *C, unsigned ldc) {
    for(unsigned i = begin; i < end; i++) {
        for(unsigned k = 0; k < K; k++) {
            float a_p = alpha * A[i*lda + k];
            for(unsigned j = 0; j < N; j++) {
                C[i*ldc + j] += a_p * B[k*ldb + j];
            }
        }
    }
}

// GEMM NT
template <typename T>
void gemm_nt(unsigned begin, unsigned end, unsigned N, unsigned K, float alpha, 
             T *A, unsigned lda, T *B, unsigned ldb, T *C, unsigned ldc) {
    for(unsigned i = begin; i < end; i++) {
        for(unsigned j = 0; j < N; j++) {
            float sum = 0;
            for(unsigned k = 0; k < K; k++) {
                sum += alpha * A[i*lda + k] * B[j*ldb + k];
            }
            C[i*ldc + j] += sum;
        }
    }
}

// GEMM TN
template <typename T>
void gemm_tn(unsigned begin, unsigned end, unsigned N, unsigned K, float alpha, 
             T *A, unsigned lda, T *B, unsigned ldb, T *C, unsigned ldc) {
    for(unsigned i = begin; i < end; i++) {
        for(unsigned k = 0; k < K; k++) {
            float a_p = alpha * A[k*lda + i];
            for(unsigned j = 0; j  < N; j++) {
                C[i*ldc + j] += a_p * B[k*ldb + j];
            }
        }
    }
}

// GEMM TT
template <typename T>
void gemm_tt(unsigned begin, unsigned end, unsigned N, unsigned K, float alpha, 
             T *A, unsigned lda, T *B, unsigned ldb, T *C, unsigned ldc) {
    for(unsigned i = begin; i < end; i++) {
        for(unsigned j = 0; j < N; j++) {
            float sum = 0;
            for(unsigned k = 0; k < K; k++) {
                sum += alpha * A[k*lda + i] * B[j*ldb + k];
            }
            C[i*ldc + j] += sum;
        }
    }
}

// Matrix multiplication
template <typename T>
void gemm(bool TA, bool TB, 
          unsigned M, unsigned N, unsigned K, 
          float alpha,
          T *A, unsigned lda, 
          T *B, unsigned ldb,
          float beta, 
          T *C, unsigned ldc, 
          unsigned num_threads) {
    for(unsigned i = 0; i < M; i++) {
        for(unsigned j = 0; j < N; j++) { C[i*ldc + j] *= beta; }
    }

    // Diminish the number of threads if matrix size isn't large enough.
    if(M < num_threads) { num_threads = M; }
    // Do matrix multiplication based on their normal|transpose composition.
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    for(unsigned tid = 0; tid < num_threads; tid++) {
        threads.emplace_back(std::bind([&](unsigned begin, unsigned end,
                                      unsigned N, unsigned K, float alpha,
                                      T *A, unsigned lda,
                                      T *B, unsigned ldb,
                                      T *C, unsigned ldc) {
            if(!TA && !TB) {
                gemm_nn(begin, end, N, K, alpha, A, lda, B, ldb, C, ldc);
            }
            else if(TA && !TB) {
                gemm_tn(begin, end, N, K, alpha, A, lda, B, ldb, C, ldc);
            }
            else if(!TA && TB) {
                gemm_nt(begin, end, N, K, alpha, A, lda, B, ldb, C, ldc);
            }
            else {
                gemm_tt(begin, end, N, K, alpha, A, lda, B, ldb, C, ldc);
            }
        }, tid * M / num_threads, (tid + 1) * M / num_threads,
           N, K, alpha, A, lda, B, ldb, C, ldc));
    } for_each(threads.begin(), threads.end(), [](std::thread& t) { t. join(); });
}


#endif
