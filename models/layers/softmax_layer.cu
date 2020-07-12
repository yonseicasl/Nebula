extern "C++" {
#include "softmax_layer.h"
}

namespace nebula {

__global__ void _softmax_(float *m_input_data, unsigned m_input_size, unsigned m_batch_size,
                          float *m_output_data) {
    size_t tid = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (tid >= m_batch_size) { return; }
    
    float *input = m_input_data + tid * m_input_size;
    float *output = m_output_data + tid * m_input_size;
    
    float sum = 0.0;
    float max = 0.0 - INFINITY;
    for(unsigned i = 0; i < m_input_size; i++) {
        if(input[i] > max) { max = input[i]; }
    }
    for(unsigned i = 0; i < m_input_size; i++) {
        float e = exp(input[i] - max);
        sum += e;
        output[i] = e;
    }
    for(unsigned i = 0; i < m_input_size; i++) {
        output[i] /= sum;
    } 
}

// Forward propagation
void softmax_layer_t::_forward_() {
    cudaMemset(delta_dev, 0.0, output_size * network->batch_size * sizeof(float)); 

    float *input_data_dev = prev_layer ? prev_layer->output_data_dev : network->input_data_dev;
   
    dim3 cuda_griddim = {(network->batch_size - 1) / BLOCK_SIZE + 1, 1, 1};
    _softmax_<<<cuda_griddim, BLOCK_SIZE>>>(input_data_dev, input_size, network->batch_size, output_data_dev);
    cudaMemcpy(output_data, output_data_dev,
               output_size * network->batch_size * sizeof(float), cudaMemcpyDeviceToHost);
}

// Backward propagation
void softmax_layer_t::_backward_() {
    float *prev_delta_dev = prev_layer ? prev_layer->delta_dev : NULL;
    
    const float alpha = 1.0;
#ifdef CUSTOM_BLAS
    _axpy_(input_size * network->batch_size, alpha, delta_dev, 1, prev_delta_dev, 1);

#else
    cublasSaxpy(network->cublas_handle, input_size * network->batch_size, &alpha, delta_dev, 1, prev_delta_dev, 1);
#endif
}

// Layer update
void softmax_layer_t::_update_() {
    // Nothing to do
}

}
// End of namespace nebula.
