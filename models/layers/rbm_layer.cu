extern "C++" {
#include "rbm_layer.h"
#include "utils.h"
#include "gemm.h"
#include "connected_layer.h"
}
#include <cuda.h>
#include <cuda_runtime.h>
#include "activations.cu"

namespace nebula {

__global__ void _forward_bias_rbm_(float *m_output_data, float *m_bias, unsigned m_batch_size,
                               unsigned m_output_size) {
    unsigned i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if(i >= m_output_size * m_batch_size) { return; }
    unsigned j = i % m_output_size;
    i /= m_output_size;
    m_output_data[i * m_output_size + j] += m_bias[j];
}

__global__ void _backward_bias_rbm_(float *m_bias_update, float *m_delta, unsigned m_batch_size,
                                unsigned m_output_size) {
    size_t i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if(i >= m_output_size) { return; }
    for(unsigned  j = 0; j < m_batch_size; j++) {
        m_bias_update[i] += m_delta[j * m_output_size + i];
    }
}

__global__ void _sampling_(float *m_sample, float *m_probability, unsigned m_total_size) {
    size_t i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if(i >= m_total_size) { return; }
    m_sample[i] = m_sample[i] < m_probability[i] ? 1.0 : 0.0;
}

__global__ void _calc_hidden_bias_update_(float *m_hidden_bias_update_dev, float *m_hidden_mean_zero_step_dev, float *m_hidden_mean_k_step_dev, unsigned m_batch_size, unsigned m_output_size) {
    size_t j = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if(j >=m_output_size) { return; }
    for(unsigned i = 0; i < m_batch_size; i++)
    {
        m_hidden_bias_update_dev[j] += m_hidden_mean_zero_step_dev[i * m_output_size + j] - m_hidden_mean_k_step_dev[i * m_output_size + j];
    }
}

__global__ void _calc_visible_bias_update_(float *m_visible_bias_update_dev, float *m_visible_units_zero_step_dev, float *m_visible_units_k_step_dev, unsigned m_batch_size, unsigned m_input_size) {
    size_t k = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if(k >=m_input_size) { return; }
    
    for(unsigned i = 0; i < m_batch_size; i++)
    {
        m_visible_bias_update_dev[k] += m_visible_units_zero_step_dev[i * m_input_size + k] - m_visible_units_k_step_dev[i * m_input_size + k];
    }
}

extern "C++" void rbm_layer_t::_sample_hidden_units_(unsigned m_step) {
    cudaMemset(hidden_units_dev, 0.0, output_size * network->batch_size * sizeof(float));
    
    float *t_visible_units_dev;
    float *t_hidden_mean_dev;

    if(m_step==0) {
        t_visible_units_dev = visible_units_zero_step_dev;
        t_hidden_mean_dev   = hidden_mean_zero_step_dev;
    }
    else {
        t_visible_units_dev = visible_units_k_step_dev;
        t_hidden_mean_dev   = hidden_mean_k_step_dev; 
    }
    
    const float alpha = 1.0;
    const float beta  = 1.0;
    
#ifdef CUSTOM_BLAS  
    _gemm_(CUBLAS_OP_T, CUBLAS_OP_N,
           output_size, network->batch_size, input_size, 
           alpha, 
           weight_dev, input_size, 
           t_visible_units_dev, input_size, 
           beta, 
           t_hidden_mean_dev, output_size);

#else
    cublasSgemm(network->cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, 
                output_size, network->batch_size, input_size, 
                &alpha, 
                weight_dev, input_size, 
                t_visible_units_dev, input_size, 
                &beta,
                t_hidden_mean_dev, output_size);
    
#endif

    dim3 cuda_griddim = {(output_size * network->batch_size -1)/BLOCK_SIZE +1, 1, 1};
    _forward_bias_rbm_<<< cuda_griddim, BLOCK_SIZE >>> (t_hidden_mean_dev, hidden_bias_dev, 
                                                    network->batch_size, output_size);
    
    _logistic_activate_<<<cuda_griddim, BLOCK_SIZE>>>(t_hidden_mean_dev, network->batch_size * output_size);

    curandGenerateUniform(network->generator, hidden_units_dev, network->batch_size * output_size);
    _sampling_<<<cuda_griddim, BLOCK_SIZE>>>(hidden_units_dev, t_hidden_mean_dev, network->batch_size * output_size);
}

extern "C++" void rbm_layer_t::_sample_visible_units_() {
    cudaMemset(visible_units_k_step_dev, 0.0, input_size * network->batch_size * sizeof(float));

    const float alpha = 1.0;
    const float beta  = 1.0;

#ifdef CUSTOM_BLAS
  
    _gemm_(CUBLAS_OP_N, CUBLAS_OP_N,
           input_size, network->batch_size, output_size, 
           alpha, 
           weight_dev, input_size, 
           hidden_units_dev, output_size, 
           beta, 
           visible_mean_dev, input_size);
   
#else
    cublasSgemm(network->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                input_size, network->batch_size, output_size, 
                &alpha, 
                weight_dev, input_size, 
                hidden_units_dev, output_size, 
                &beta,
                visible_mean_dev, input_size);    
#endif
    
    dim3 cuda_griddim = {(input_size * network->batch_size -1)/BLOCK_SIZE +1, 1, 1};
    _forward_bias_rbm_<<< cuda_griddim, BLOCK_SIZE >>> (visible_mean_dev, visible_bias_dev, 
                                                    network->batch_size, input_size);
    
    _logistic_activate_<<<cuda_griddim, BLOCK_SIZE>>>(visible_mean_dev, network->batch_size * input_size);
    
    curandGenerateUniform(network->generator, visible_units_k_step_dev, network->batch_size * input_size);
    _sampling_<<<cuda_griddim, BLOCK_SIZE>>>(visible_units_k_step_dev, visible_mean_dev, network->batch_size * input_size);
}

extern "C++" void rbm_layer_t::_pretrain_() {
	cudaMemset(output_data_dev, 0.0, output_size * network->batch_size * sizeof(float));
    cudaMemset(delta_dev, 0.0, output_size * network->batch_size * sizeof(float));

    float *input_data_dev = prev_layer ? prev_layer->output_data_dev : network->input_data_dev;
     
    cudaMemcpy(visible_units_zero_step_dev, input_data_dev, input_size * network->batch_size * sizeof(float), cudaMemcpyDeviceToDevice);

    // K-step contrastive divergence_gradient approximation for weight update and bias update
    for(unsigned t = 0; t < k_step; t++)
    {
        if(!t)
        {
            _sample_hidden_units_(t);
            _sample_visible_units_();
        }
        else
        {
            _sample_hidden_units_(t);
            _sample_visible_units_();
        }
        _sample_hidden_units_(1);
    }
    
    const float alpha  = 1.0;
    const float alpha2 = -1.0;
    const float beta   = 1.0;
    
#ifdef CUSTOM_BLAS  
    _gemm_(CUBLAS_OP_N, CUBLAS_OP_T, 
           input_size, output_size, network->batch_size, 
           alpha, 
           visible_units_zero_step_dev, input_size, 
           hidden_mean_zero_step_dev, output_size,
           beta,
           weight_update_dev, input_size);
    _gemm_(CUBLAS_OP_N, CUBLAS_OP_T,
           input_size, output_size, network->batch_size,
           alpha2,
           visible_units_k_step_dev, input_size,
           hidden_mean_k_step_dev, output_size,
           beta, 
           weight_update_dev, input_size);
#else
    // Matrix multiplication for weight update.
    cublasSgemm(network->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, 
                input_size, output_size, network->batch_size, 
                &alpha, 
                visible_units_zero_step_dev, input_size, 
                hidden_mean_zero_step_dev, output_size, 
                &beta, 
                weight_update_dev, input_size);
    cublasSgemm(network->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, 
                input_size, output_size, network->batch_size, 
                &alpha2, 
                visible_units_k_step_dev, input_size,
                hidden_mean_k_step_dev, output_size, 
                &beta, 
                weight_update_dev, input_size);
#endif

    dim3 cuda_griddim_a = {(output_size - 1)/BLOCK_SIZE + 1, 1, 1};
    _calc_hidden_bias_update_<<< cuda_griddim_a, BLOCK_SIZE >>>(hidden_bias_update_dev, hidden_mean_zero_step_dev, hidden_mean_k_step_dev, network->batch_size, output_size);
    
    dim3 cuda_griddim_b = {(input_size - 1)/BLOCK_SIZE + 1, 1, 1};
    _calc_visible_bias_update_<<< cuda_griddim_b, BLOCK_SIZE >>> (visible_bias_update_dev, visible_units_zero_step_dev, visible_units_k_step_dev, network->batch_size, input_size);
   
    float learning_rate = network->learning_rate/network->batch_size;
    float decay = (0.0 - network->decay) * network->batch_size;
    float momentum = network->momentum;

#ifdef CUSTOM_BLAS
    _axpy_(input_size, learning_rate, visible_bias_update_dev, 1, visible_bias_dev, 1);
    _scal_(input_size, momentum, visible_bias_update_dev, 1);

    _axpy_(output_size, learning_rate, hidden_bias_update_dev, 1, hidden_bias_dev, 1);
    _scal_(output_size, momentum, hidden_bias_update_dev, 1);

    _axpy_(weight_size, decay, weight_dev, 1, weight_update_dev, 1);
    _axpy_(weight_size, learning_rate, weight_update_dev, 1, weight_dev, 1);
    _scal_(weight_size, momentum, weight_update_dev, 1);
#else
    // Update bias of visible units.
    cublasSaxpy(network->cublas_handle, input_size, &learning_rate, 
                visible_bias_update_dev, 1, visible_bias_dev, 1);
    cublasSscal(network->cublas_handle, input_size, &momentum, visible_bias_update_dev, 1);

    // Update bias of hidden units.
    cublasSaxpy(network->cublas_handle, output_size, &learning_rate, 
                hidden_bias_update_dev, 1, hidden_bias_dev, 1);
    cublasSscal(network->cublas_handle, output_size, &momentum, hidden_bias_update_dev, 1);

    // Update weight.
    cublasSaxpy(network->cublas_handle, weight_size, &decay,
                weight_dev, 1, weight_update_dev, 1);
    cublasSaxpy(network->cublas_handle, weight_size, &learning_rate, 
                weight_update_dev, 1, weight_dev, 1);
    cublasSscal(network->cublas_handle, weight_size, &momentum, weight_update_dev, 1);
#endif

    cudaMemcpy(output_data_dev, hidden_units_dev, output_size * network->batch_size * sizeof(float), cudaMemcpyDeviceToDevice);
}

// Forward propagation
extern "C++" void rbm_layer_t::_forward_() {
    cudaMemset(output_data_dev, 0, output_size*network->batch_size*sizeof(float));
    cudaMemset(delta_dev, 0, output_size*network->batch_size*sizeof(float));

    const float alpha = 1.0;
    const float beta  = 1.0;
    float *input_data_dev = prev_layer ? prev_layer->output_data_dev : network->input_data_dev;

#ifdef CUSTOM_BLAS
  
    _gemm_(CUBLAS_OP_T, CUBLAS_OP_N,
           output_size, network->batch_size, input_size, 
           alpha, 
           weight_dev, input_size, 
           input_data_dev, input_size, 
           beta, 
           output_data_dev, output_size);
   
#else
    cublasSgemm(network->cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, 
                output_size, network->batch_size, input_size, 
                &alpha, 
                weight_dev, input_size, 
                input_data_dev, input_size, 
                &beta,
                output_data_dev, output_size);
#endif
    // Forward bias
    dim3 cuda_griddim = {(output_size * network->batch_size -1)/BLOCK_SIZE +1, 1, 1};
    _forward_bias_rbm_<<< cuda_griddim, BLOCK_SIZE >>> (output_data_dev, hidden_bias_dev, 
                                                    network->batch_size, output_size);
    
    // Activate function
    _activate_();
}

// Backward propagation
extern "C++" void rbm_layer_t::_backward_() {
    // Gradient function 
    _gradient_();
     
    // backward bias.
    dim3 cuda_griddim = { (output_size - 1)/ BLOCK_SIZE + 1, 1, 1};
    _backward_bias_rbm_<<<cuda_griddim, BLOCK_SIZE>>>(hidden_bias_update_dev, delta_dev,
                                         network->batch_size, output_size);
    const float alpha = 1.0;
    const float beta  = 1.0;
    float *input_data_dev = prev_layer ? prev_layer->output_data_dev : network->input_data_dev;
    float *prev_delta_dev = prev_layer ? prev_layer->delta_dev : NULL;
    
    // Weight update
#ifdef CUSTOM_BLAS
    _gemm_(CUBLAS_OP_N, CUBLAS_OP_T, 
           input_size, output_size, network->batch_size, 
           alpha, 
           input_data_dev, input_size, 
           delta_dev, output_size, 
           beta, 
           weight_update_dev, input_size);
#else
    cublasSgemm(network->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, 
                input_size, output_size, network->batch_size, 
                &alpha, 
                input_data_dev, input_size, 
                delta_dev, output_size, 
                &beta, 
                weight_update_dev, input_size);
#endif
    // Delta update
    if(prev_delta_dev) {
#ifdef CUSTOM_BLAS
        _gemm_(CUBLAS_OP_N, CUBLAS_OP_N, 
               input_size, network->batch_size, output_size, 
               alpha, 
               weight_dev, input_size,
               delta_dev, output_size, 
               beta, 
               prev_delta_dev, input_size);
#else
        cublasSgemm(network->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                    input_size, network->batch_size, output_size, 
                    &alpha, 
                    weight_dev, input_size,
                    delta_dev, output_size, 
                    &beta, 
                    prev_delta_dev, input_size);
#endif
    }
}

extern "C++" void rbm_layer_t::_update_() {
    float learning_rate = network->learning_rate/network->batch_size;
    float decay = -network->decay*network->batch_size;
    float momentum = network->momentum;

#ifdef CUSTOM_BLAS
    _axpy_(weight_size, decay, weight_dev, 1, weight_update_dev, 1);
    _axpy_(weight_size, learning_rate, weight_update_dev, 1, weight_dev, 1);
    _scal_(weight_size, momentum, weight_update_dev, 1);

    _axpy_(output_size, learning_rate, hidden_bias_update_dev, 1, hidden_bias_dev, 1);
    _scal_(output_size, momentum, hidden_bias_update_dev, 1);
#else
    // Weight update
    cublasSaxpy(network->cublas_handle, weight_size, &decay,
                weight_dev, 1, weight_update_dev, 1);
    cublasSaxpy(network->cublas_handle, weight_size, &learning_rate, 
                weight_update_dev, 1, weight_dev, 1);
    cublasSscal(network->cublas_handle, weight_size, &momentum, weight_update_dev, 1);

    // Bias update
    cublasSaxpy(network->cublas_handle, output_size, &learning_rate, 
                hidden_bias_update_dev, 1, hidden_bias_dev, 1);
    cublasSscal(network->cublas_handle, output_size, &momentum, hidden_bias_update_dev, 1);
#endif

}

}
// End of namespace nebula.
