#include <cstring>
#ifndef CUSTOM_BLAS
#include <cblas.h>
#endif

#include "concat_layer.h"

namespace nebula {

concat_layer_t::concat_layer_t(network_t *m_network, layer_t *m_prev_layer, layer_type_t m_layer_type) :
    layer_t(m_network, m_prev_layer, m_layer_type),
    num_concats(1) {

}

concat_layer_t::~concat_layer_t() {
    delete [] output_data;
    delete [] delta;
#ifdef GPU_ENABLED
    cudaFree(output_data_dev);
    cudaFree(delta_dev);
#endif
}

void concat_layer_t::init(section_config_t m_section_config) {

    std::string line;
    m_section_config.get_setting("hops", &line);
    line.erase(remove(line.begin(), line.end(),' '),line.end());
    char line_array[line.length() + 1];
    strcpy(line_array, line.c_str());

    // Count the number of layers to concatenate
    for(unsigned i = 0; i < line.length() + 1; i++) {
        if(line_array[i] == ',') {
            num_concats++;
        }
    }
    hop.reserve(num_concats);
    hop.assign(num_concats, 0);
    connections.reserve(num_concats);

    m_section_config.get_vector_setting("hops", &hop);

    for(unsigned i = 0; i < num_concats; i++) {
        layer_t *connection_ = this;
        for(unsigned j = 0; j < hop[i]; j++) {
            connection_ = connection_->prev_layer ? connection_->prev_layer : NULL;
        }
        connections.push_back(connection_);
    }

    output_height = connections[0]->output_height;
    output_width  = connections[0]->output_width;
    output_channel = 0;
    for(unsigned i = 0; i < connections.size(); i++) {
        output_channel += connections[i]->output_channel;
    }

    output_size = output_height * output_width * output_channel;
    output_data = new float[output_size * network->batch_size]();
    delta = new float[output_size * network->batch_size]();

#ifdef GPU_ENABLED
    cudaMalloc((void**)&output_data_dev, output_size * network->batch_size * sizeof(float));
    cudaMalloc((void**)&delta_dev, output_size * network->batch_size * sizeof(float));

    cudaMemset(output_data_dev, 0.0, output_size * network->batch_size * sizeof(float));
    cudaMemset(delta_dev, 0.0, output_size * network->batch_size * sizeof(float));
#endif
}

void concat_layer_t::init_weight(std::fstream &m_weight_file) {
    // Nothing to do.
}

void concat_layer_t::init_weight() {
    // Nothing to do.
}

void concat_layer_t::forward() {
    memset(output_data, 0.0, output_size * network->batch_size * sizeof(float));
    memset(delta, 0.0, output_size * network->batch_size * sizeof(float));

    // std::cout << "concat layer" << std::endl;
    // std::cout << output_height << "*" << output_width << "*" << output_channel << std::endl;

    for(unsigned i = 0; i < num_concats; i++) {
        for(unsigned b = 0; b < network->batch_size; b++) {
            memcpy(output_data, connections[i]->output_data, connections[i]->output_size * sizeof(float));
            output_data += connections[i]->output_size;
        }
    }
}

void concat_layer_t::backward() {
    for(unsigned i = 0; i < num_concats; i++) {
        for(unsigned b = 0; b < network->batch_size; b++) {
#ifdef CUSTOM_BLAS
            axpy(connections[i]->output_size*network->batch_size, 1, delta, 1, connections[i]->delta);
#else
            cblas_saxpy(connections[i]->output_size, 1, delta, 1, connections[i]->delta, 1);
#endif
            delta += connections[i]->output_size;
        }
    }
    //distribute delta
}

void concat_layer_t::update() {
    // Nothing to do.
}

void concat_layer_t::store_weight(std::fstream &m_weight_file) {
    // Nothing to do.
}

// End of namespace nebula
}
