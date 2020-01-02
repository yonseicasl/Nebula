extern "C++" {
#include "dropout_layer.h"
}

__global__ void _dropout_(float *m_input_data, unsigned m_total_size, float *m_rand,
                          float m_probability, float m_scale) {
    size_t tid = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if(tid >= m_total_size) { return; }
    m_input_data[tid] = (m_rand[tid] < m_probability) ? 0.0 : m_input_data[tid] * m_scale;
}

// Forward propagation
extern "C++" void dropout_layer_t::_forward_() {
    float *input_data_dev = prev_layer ? prev_layer->output_data_dev : network->input_data_dev;

    if(network->run_type != TRAIN_RUN) { return; }
    float scale = 1.0 / (1.0 - probability);
    unsigned total_size = input_size * network->batch_size;
    
    dim3 cuda_griddim = {(total_size - 1) / BLOCK_SIZE + 1, 1, 1}; 
    _dropout_<<<cuda_griddim, BLOCK_SIZE>>>(input_data_dev, total_size, rand_dev, probability, scale);    
}

// Backward propagation
extern "C++" void dropout_layer_t::_backward_() {
    float *prev_delta = prev_layer ? prev_layer->delta_dev : NULL;

    if(!prev_delta) { return; }
    float scale = 1.0 / (1.0 - probability);
    unsigned total_size = input_size * network->batch_size;
    
    dim3 cuda_griddim = {(total_size - 1) / BLOCK_SIZE + 1, 1, 1};
    _dropout_<<<cuda_griddim, BLOCK_SIZE>>>(delta_dev, total_size, rand_dev, probability, scale);
}

// Layer update
extern "C++" void dropout_layer_t::_update_() {
    // Nothing to do
}

