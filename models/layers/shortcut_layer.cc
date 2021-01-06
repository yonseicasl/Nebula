#include <algorithm>
#ifndef CUSTOM_BLAS
    #include <cblas.h>
#endif
#include <fstream>
#include <functional>
#include <cstring>
#include <random>
#include <thread>
#include "shortcut_layer.h"
#include "gemm.h"

namespace nebula {

shortcut_layer_t::shortcut_layer_t(network_t *m_network, layer_t *m_prev_layer, layer_type_t m_layer_type) :
    layer_t(m_network, m_prev_layer, m_layer_type) {
}

shortcut_layer_t::~shortcut_layer_t() {
    delete [] output_data;
    delete [] delta;
}

void shortcut_layer_t::init(section_config_t m_section_config) {
    // Get layer settings.
    m_section_config.get_setting("hops", &hops);
    std::string activation_str;
    if(m_section_config.get_setting("activation", &activation_str)) {
        activation_type = (activation_type_t)get_type(activation_type_str, activation_str);
    }
    // Initialize layer parameters.
    connection = this;
    for(unsigned i = 0; i < hops; i++) { connection = connection->prev_layer ? connection->prev_layer : NULL;}

    input_height  = connection->output_height;
    input_width   = connection->output_width;
    input_channel = connection->output_channel;
    
    output_height = prev_layer ? prev_layer->output_height : network->input_height;
    output_width = prev_layer ? prev_layer->output_width : network->input_width;
    output_channel = prev_layer ? prev_layer->output_channel : network->input_channel;
    
    output_size = output_height * output_width * output_channel;
    input_size = output_size;
	
    output_data = new float[output_size * network->batch_size]();
    delta = new float[output_size * network->batch_size]();

}

void shortcut_layer_t::init_weight(std::fstream &m_weight_file){}
void shortcut_layer_t::init_weight(){}

void shortcut_layer_t::forward() {
    memset(output_data, 0.0, output_size * network->batch_size * sizeof(float));
    memset(delta , 0.0, output_size * network->batch_size * sizeof(float));

    input_data = prev_layer ? prev_layer->output_data : network->input_data;
    
    memcpy(output_data, input_data, output_size * network->batch_size * sizeof(float));
    shortcut(num_threads, input_width, input_height, input_channel, connection->output_data, 
             output_width, output_height, output_channel, output_data, network->batch_size);
    // Activate function
    activate();
}

void shortcut_layer_t::backward() {
    // Gradient function
    gradient();

#ifdef CUSTOM_BLAS
    axpy(output_size * network->batch_size, 1, delta, 1, prev_layer->delta, 1);
#else
    cblas_saxpy(output_size * network->batch_size, 1, delta, 1, prev_layer->delta,1); 
#endif
    shortcut(num_threads, output_width, output_height, output_channel, delta, 
             input_width, input_height, input_channel, connection->delta, network->batch_size);
}

void shortcut_layer_t::update(){
    // Nothing to do.    
}
void shortcut_layer_t::store_weight(std::fstream &m_weight_file){
    // Nothing to do,    
}

}
// End of namespace nebula.
