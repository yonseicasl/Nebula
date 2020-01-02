extern "C++" {
#include "cost_layer.h"
}

__global__ void _l2_(unsigned m_total_size, float *m_input_data, float *m_label,
                     float *m_delta, float *m_output_data) {
    size_t tid = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if(tid >= m_total_size) { return; }
    m_delta[tid] = m_label[tid] - m_input_data[tid];
    m_output_data[tid] = m_delta[tid] * m_delta[tid];
}

__global__ void _l1_(unsigned m_total_size, float *m_input_data, float *m_label,
                     float *m_delta, float *m_output_data) {
    unsigned tid = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    m_delta[tid] = m_label[tid] - m_input_data[tid];
    m_output_data[tid] = m_label[tid] ? -log(m_input_data[tid]) : 0;
    //m_output_data[tid] = abs(m_delta[tid]);
}


// Forward propagation
extern "C++" void cost_layer_t::_forward_() {
    cudaMemset(delta_dev, 0.0, output_size * network->batch_size * sizeof(float));
    float *input_data_dev = prev_layer ? prev_layer->output_data_dev : network->input_data_dev;

    switch(cost_type) {
        case SMOOTH_COST:
        case SSE_COST:
        case L1_COST: {
            dim3 dim = {(input_size * network->batch_size -1) / BLOCK_SIZE + 1, 1, 1};
            _l1_<<<dim, BLOCK_SIZE>>>(input_size * network->batch_size, input_data_dev, network->input_label_dev, delta_dev, output_data_dev);
            break;
        }
        case L2_COST: {
            dim3 dim = {(input_size * network->batch_size - 1) / BLOCK_SIZE + 1, 1, 1};
            _l2_<<<dim, BLOCK_SIZE>>>(input_size * network->batch_size, input_data_dev,
                                      network->input_label_dev, delta_dev, output_data_dev);
            break;
        }
        default: {
            std::cerr << "Error: undefined cost " << cost_type_str[cost_type] << std::endl;
            exit(1);
        }
    }
    cudaMemcpy(output_data, output_data_dev, output_size * network->batch_size*sizeof(float), cudaMemcpyDeviceToHost);
    network->cost = 0; 
    for(unsigned i = 0; i < network->batch_size; i++){
        for(unsigned j =0; j < output_size; j++){
            network->cost += output_data[i * output_size + j];
        }
    }
}

// Backward propagation
extern "C++" void cost_layer_t::_backward_() {
    float *prev_delta_dev = prev_layer ? prev_layer->delta_dev : NULL;
    const float alpha = 1.0;
#ifdef CUSTOM_BLAS
    _axpy_(input_size * network->batch_size, alpha, delta_dev, 1, prev_delta_dev, 1);
#else
    cublasSaxpy(network->cublas_handle, input_size*network->batch_size, &alpha, delta_dev, 1, prev_delta_dev, 1);
#endif
}

// Layer update
extern "C++" void cost_layer_t::_update_() {
    // Nothing to do
}

