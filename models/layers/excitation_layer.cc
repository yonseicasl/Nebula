#include <algorithm>
#ifndef CUSTOM_BLAS
    #include <cblas.h>
#endif
#include <fstream>
#include <functional>
#include <cstring>
#include <random>
#include <thread>
#include "excitation_layer.h"
#include "utils.h"
#include "batchnorm.h"
#include "gemm.h"

namespace nebula { 

excitation_layer_t::excitation_layer_t(network_t *m_network, layer_t *m_prev_layer, layer_type_t m_layer_type) :
    layer_t(m_network, m_prev_layer, m_layer_type) {
}

excitation_layer_t::~excitation_layer_t() {
    delete [] output_data;
    delete [] delta;
}

void excitation_layer_t::init(section_config_t m_section_config) {
    // Get layer settings.

    input_size = prev_layer ? prev_layer->output_size : network->input_size;

    input_height = prev_layer ? prev_layer->output_height : network->input_height;
    input_width = prev_layer ? prev_layer->output_width : network->input_width;
    input_channel = prev_layer ? prev_layer->output_channel : network->input_channel;

    m_section_config.get_setting("output_height", &output_height);
    m_section_config.get_setting("output_width", &output_width);
    m_section_config.get_setting("output_channel", &output_channel);
    output_size = output_height * output_width * output_channel;
    
    input_data = prev_layer ? prev_layer->output_data : network->input_data;
    output_data = new float[output_size * network->batch_size]();
    delta = new float[output_size * network->batch_size]();

    // Print out structure of the network.

    // Initialize parameters for batch normalization.
}

void excitation_layer_t::init_weight(std::fstream &m_input_weight) {
   
}

void excitation_layer_t::init_weight() {
}

void excitation_layer_t::store_weight(std::fstream &m_output_weight) {
}

void excitation_layer_t::forward() {
}

void excitation_layer_t::backward() {
}
void excitation_layer_t::update() {
}

// Forward batch normalization.
void excitation_layer_t::forward_batchnorm() {
}

//Backward batch normalization.
void excitation_layer_t::backward_batchnorm() {
}

}
//End of namespace nebula.
