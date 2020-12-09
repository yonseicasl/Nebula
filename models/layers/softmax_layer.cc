#ifndef CUSTOM_BLAS
    #include <cblas.h>
#endif
#include <cfloat>
#include <cmath>
#include <cstring>
#include "softmax_layer.h"

namespace nebula {

softmax_layer_t::softmax_layer_t(network_t *m_network, layer_t *m_prev_layer, layer_type_t m_layer_type) :
    layer_t(m_network, m_prev_layer, m_layer_type) {
}

softmax_layer_t::~softmax_layer_t() {
    delete [] output_data;
    delete [] delta;
}

// Initialize layer.
void softmax_layer_t::init(section_config_t m_section_config) {
    // Initialize layer parameters.
    input_size = prev_layer ? prev_layer->output_size : network->input_size;
    output_size = input_size;  
    
    output_data = new float[output_size * network->batch_size]();
    delta = new float[output_size * network->batch_size]();
}

// Initialize weight from file.
void softmax_layer_t::init_weight(std::fstream &m_weight_file) {
    // Nothing to do
}

// Initialize weight from scratch.
void softmax_layer_t::init_weight() {
    // Nothing to do
}

// Forward propagation
void softmax_layer_t::forward() {
    memset(delta, 0, output_size*network->batch_size*sizeof(float));

    // Softmax function per batch.
    softmax();
}


// Backward propagation
void softmax_layer_t::backward() {
    //Update delta value of previous layer.
    float *prev_delta = prev_layer ? prev_layer->delta : NULL;
    if(prev_delta) { 
#ifdef CUSTOM_BLAS
        axpy(input_size * network->batch_size, 1, delta, 1, prev_delta, 1);
#else
        cblas_saxpy(input_size*network->batch_size, 1, delta, 1, prev_delta, 1);
#endif
    }
}

// Layer update
void softmax_layer_t::update() {/* Nothing to do */}

// Store weight.
void softmax_layer_t::store_weight(std::fstream &m_weight_file) {/*Nothing to do*/}

// Softmax function.
void softmax_layer_t::softmax() {
    float *input_data = prev_layer ? prev_layer->output_data : network->input_data;
 
    for(unsigned i = 0; i < network->batch_size; i++) {
        float sum = 0.0;
        float max = 0.0 - FLT_MAX;
        
        float *input  = &input_data[i * input_size];
        float *output = &output_data[i * output_size];
        for(unsigned j = 0; j < input_size; j++) {
            if(input[j] > max) { max = input[j]; }
        }
        for(unsigned j = 0; j < input_size; j++) {
            float e = exp(input[j] - max);
            sum += e;
            output[j] = e;
        }
        for(unsigned j = 0; j < input_size; j++) {
            output[j] /= sum;
        }
    }
}

}
// End of namespace nebula.
