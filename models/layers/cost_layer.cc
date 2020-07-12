#ifndef CUSTOM_BLAS
    #include <cblas.h>
#endif
#include <cstring>
#ifdef GPU_ENABLED
#include <cuda_runtime.h>
#endif
#include "cost_layer.h"

namespace nebula {

cost_layer_t::cost_layer_t(network_t *m_network, layer_t *m_prev_layer, layer_type_t m_layer_type) :
    layer_t(m_network, m_prev_layer, m_layer_type),
    cost_type(UNDEFINED_COST) {
}

cost_layer_t::~cost_layer_t() {
    delete [] output_data;
    delete [] delta;
#ifdef GPU_ENABLED
    cudaFree(output_data_dev);
    cudaFree(delta_dev);
#endif
}

// Initialize layer.
void cost_layer_t::init(section_config_t m_section_config) {
    // Layer settings.
    std::string cost_str;
    if(m_section_config.get_setting("type", &cost_str)) {
        cost_type = (cost_type_t)get_type(cost_type_str, cost_str);
    }
    
    // Initialize layer parameters.
    input_size = prev_layer ? prev_layer->output_size : network->input_size;
    output_size = input_size;
	
    output_data = new float[output_size * network->batch_size]();
    delta = new float[output_size * network->batch_size]();
#ifdef GPU_ENABLED
    cudaMalloc((void**)&output_data_dev, output_size * network->batch_size * sizeof(float));
    cudaMalloc((void**)&delta_dev, output_size * network->batch_size * sizeof(float));
    cudaMemset(output_data_dev, 0.0, output_size * network->batch_size * sizeof(float));
    cudaMemset(delta_dev, 0.0, output_size * network->batch_size * sizeof(float));
#endif
}

// Initialize weight from file.
void cost_layer_t::init_weight(std::fstream &m_weight_file) {
    // Nothing to do
}

// Initialize weight from scratch.
void cost_layer_t::init_weight() {
    // Nothing to do
}

// Forward propagation
void cost_layer_t::forward() {
    memset(delta, 0.0, output_size * network->batch_size * sizeof(float));

    float *input_data = prev_layer ? prev_layer->output_data : network->input_data;

    switch(cost_type) {
        case SMOOTH_COST:
        case SSE_COST:
        case L1_COST:
        case L2_COST: {
            for(unsigned i = 0; i < output_size * network->batch_size; i++) {
                delta[i] = network->input_label[i] - input_data[i];
                output_data[i] = delta[i] * delta[i];
                network->cost += output_data[i];
             }
             break;
        }
        default: {
            std::cerr << "Error: undefined cost " << cost_type_str[cost_type] << std::endl;
            exit(1);
        } 
    }
}

// Backward propagation
void cost_layer_t::backward() {
    float *prev_delta = prev_layer ? prev_layer->delta : NULL;
    if(prev_delta){ 
#ifdef CUSTOM_BLAS
        axpy(input_size * network->batch_size, 1, delta, 1, prev_delta, 1);
#else
        cblas_saxpy(input_size*network->batch_size, 1, delta, 1, prev_delta, 1);
#endif
    }
}

// Layer update
void cost_layer_t::update() {
    // Nothing to do.
}

// Store weight.
void cost_layer_t::store_weight(std::fstream &m_weight_file) {
    // Nothing to do
}

}
// End of namespace nebula.
