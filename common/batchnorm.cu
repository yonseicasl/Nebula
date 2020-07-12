extern "C++" {
#include "batchnorm.h"
#include "def.h"
}

namespace nebula {

// Calculate normalize mean value.
__global__ void _mean_(float *m_output, float *m_mean, 
                       unsigned m_channel, unsigned m_size, unsigned m_batch){

    __shared__ float mean_update[BLOCK_SIZE];

    unsigned tid = threadIdx.x;
    mean_update[tid] = 0;

    unsigned bid = blockIdx.x;

    for(unsigned i = 0; i < m_batch; i++) {
        for(unsigned j = 0; j < m_size; j += BLOCK_SIZE) {
            mean_update[tid] += (j + tid < m_size) ? 
                                m_output[i * m_size * m_channel + bid * m_size + j + tid] : 0;
        }
    }

    __syncthreads();


    if(tid == 0) {
        m_mean[bid] = 0;
        for(unsigned i = 0 ; i < BLOCK_SIZE; i++) {
            m_mean[bid] += mean_update[i];
        }
        m_mean[bid] /= m_size * m_batch;
    }
}

extern "C++" void _batchnorm_mean_(float *m_output, float *m_mean, unsigned m_channel, 
                                   unsigned m_size, unsigned m_batch){

    _mean_<<<m_channel, BLOCK_SIZE>>>(m_output, m_mean, m_channel, m_size, m_batch);
}

// calculate variance value.
__global__ void _variance_(float *m_output, float *m_mean, float *m_variance, 
                           unsigned m_channel, unsigned m_size, unsigned m_batch) {

    __shared__ float variance_update[BLOCK_SIZE];
    unsigned tid = threadIdx.x;
    variance_update[tid] = 0.0;

    unsigned bid = blockIdx.x;

    for(unsigned i = 0; i < m_batch; i++) {
        for(unsigned j =0; j < m_size; j += BLOCK_SIZE) {
            variance_update[tid] += (j + tid < m_size) ? 
                                     powf((m_output[i * m_size * m_channel + bid * m_size + j + tid] - m_mean[bid]), 2) : 0;
        }
    }

    __syncthreads();

    if(tid == 0) {
        m_variance[bid] = 0.0;
        for(unsigned i = 0; i < BLOCK_SIZE; i++) {
            m_variance[bid] += variance_update[i];
        }
        m_variance[bid] *= 1.0/(m_size * m_batch);
    }
}

extern "C++" void _batchnorm_variance_(float *m_output, float *m_mean, float *m_variance, 
                                       unsigned m_channel, unsigned m_size, unsigned m_batch) {
    _variance_<<<m_channel, BLOCK_SIZE>>>(m_output, m_mean, m_variance, m_channel, m_size, m_batch);
}

__global__ void _normalize_(float *m_output, float *m_mean, float *m_variance, unsigned m_channel, unsigned m_size, unsigned m_batch) {
    unsigned index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    unsigned filter = (index/m_size)%m_channel;
    m_output[index] = (m_output[index] - m_mean[filter]) / (sqrt(m_variance[filter] + 0.00001));
}

extern "C++" void _batchnorm_normalize_(float *m_output, float *m_mean, float *m_variance, unsigned m_channel, unsigned m_size, unsigned m_batch) {
    unsigned total_size = m_batch * m_channel * m_size;
    dim3 cuda_griddim = {(total_size -1) / BLOCK_SIZE + 1, 1, 1};
    _normalize_<<<cuda_griddim, BLOCK_SIZE>>>(m_output, m_mean, m_variance, m_channel, m_size, m_batch);

}


__global__ void _scale_(float *m_output, float *m_scale, unsigned m_channel, unsigned m_size) {
    unsigned offset = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned filter = blockIdx.y;
    unsigned batch = blockIdx.z;

    if(offset < m_size) m_output[(batch * m_channel + filter) * m_size + offset] *= m_scale[filter];

}

extern "C++" void _batchnorm_scale_down_(float *m_output, float *m_scale, unsigned m_channel, unsigned m_size, unsigned m_batch){
    dim3 dimGrid((m_size - 1) / BLOCK_SIZE + 1, m_channel, m_batch);
    dim3 dimBlock(BLOCK_SIZE, 1, 1);
    _scale_<<<dimGrid, dimBlock>>>(m_output, m_scale, m_channel, m_size);
}

// Update delta value of normalize mean.
 __global__ void _mean_delta_(float *m_delta, float *m_variance, float *m_mean_delta, 
                             unsigned m_channel, unsigned m_size, unsigned m_batch){
    __shared__ float mean_update[BLOCK_SIZE];

    unsigned tid = threadIdx.x;
    mean_update[tid] = 0;

    unsigned bid = blockIdx.x;

    for(unsigned i = 0; i < m_batch; i++) {
        for(unsigned j = 0; j < m_size; j += BLOCK_SIZE) {
            mean_update[tid] += (j + tid < m_size) ? 
                                 m_delta[i * m_size * m_channel + bid * m_size + j + tid] : 0;
        }
    }

    __syncthreads();

    if(tid == 0) {
        m_mean_delta[bid] = 0;
        for(unsigned i = 0; i < BLOCK_SIZE; i++) {
            m_mean_delta[bid] += mean_update[i];
        }
        m_mean_delta[bid] *= (-1.0/sqrt(m_variance[bid] + 0.00001));
    }
}

void _batchnorm_mean_delta_(float *m_delta, float *m_variance, float *m_mean_delta, 
                            unsigned m_channel, unsigned m_size, unsigned m_batch){
    _mean_delta_<<<m_channel, BLOCK_SIZE>>>(m_delta, m_variance, m_mean_delta, m_channel, m_size, m_batch);
}

// Update delta value of normalize variance.
__global__ void _variance_delta_(float *m_x, float *m_delta, float *m_mean, float *m_variance, float *m_variance_delta, 
                                 unsigned m_channel, unsigned m_size, unsigned m_batch){

    __shared__ float variance_update[BLOCK_SIZE];

    unsigned tid = threadIdx.x;
    variance_update[tid] = 0.0;

    unsigned bid = blockIdx.x;

    for(unsigned i = 0; i < m_batch; i++) {
        for(unsigned j = 0; j < m_size; j += BLOCK_SIZE) {
            unsigned index = i * m_size * m_channel + bid * m_size + j + tid;
            variance_update[tid] += (j + tid < m_size) ?
                                     m_delta[index] * (m_x[index] - m_mean[bid]) : 0;
        }
    }

    __syncthreads();

    if(tid == 0) {
        m_variance_delta[bid] = 0.0;
        for(unsigned i = 0; i < BLOCK_SIZE; i++) {
            m_variance_delta[bid] += variance_update[i];
        }
        m_variance_delta[bid] *= -0.5 * pow(m_variance[bid] + 0.00001, (float)(-3.0/2.0));
    }
}
void _batchnorm_variance_delta_(float *m_x, float *m_delta, float *m_mean, float *m_variance, float *m_variance_delta, 
                                unsigned m_channel, unsigned m_size, unsigned m_batch){
    _variance_delta_<<<m_channel, BLOCK_SIZE>>>(m_x, m_delta, m_mean, m_variance, m_variance_delta, m_channel, m_size, m_batch);
}

// Normalize delta value.
__global__ void _normalize_delta_(float *m_x, float *m_normalize_mean, float *m_normalize_variance, 
                                  float *m_mean_delta, float *m_variance_delta, float *m_delta, 
                                  unsigned m_channel, unsigned m_size, unsigned m_batch){

    unsigned index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    unsigned filter_index = (index / m_size) % m_channel;

    m_delta[index] = m_delta[index] * 1.0/(sqrt(m_normalize_variance[filter_index] + 0.00001)) 
                     + m_variance_delta[filter_index] * 2.0 * (m_x[index] - m_normalize_mean[filter_index]) / (m_size * m_batch) 
                     + m_mean_delta[filter_index] / (m_size * m_batch);
}

void _batchnorm_normalize_delta_(float *m_x, float *m_normalize_mean, float *m_normalize_variance, 
                                 float *m_mean_delta, float *m_variance_delta, float *m_delta, 
                                 unsigned m_channel, unsigned m_size, unsigned m_batch){
    unsigned N = m_batch * m_channel * m_size;
    dim3 cuda_griddim = { (N-1) / BLOCK_SIZE + 1, 1, 1};
    _normalize_delta_<<<cuda_griddim, BLOCK_SIZE>>>(m_x, m_normalize_mean, m_normalize_variance, m_mean_delta, m_variance_delta, m_delta, m_channel, m_size, m_batch); 
}

}
// End of namespace nebula.
