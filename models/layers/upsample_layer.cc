#include <cstring>
#include "upsample_layer.h"

namespace nebula {

upsample_layer_t::upsample_layer_t(network_t *m_network, layer_t *m_prev_layer, layer_type_t m_layer_type) :
    layer_t(m_network, m_prev_layer, m_layer_type),
    stride(1),
    scale(1.0) {

}

upsample_layer_t::~upsample_layer_t() {
    delete [] output_data;
    delete [] delta;

#ifdef GPU_ENABLED
    cudaFree(output_data_dev);
    cudaFree(delta_dev);
#endif
}

void upsample_layer_t::init(section_config_t m_section_config) {
    m_section_config.get_setting("stride", &stride);
    m_section_config.get_setting("scale", &scale);

    input_height  = prev_layer ? prev_layer->output_height : network->input_height;
    input_width   = prev_layer ? prev_layer->output_width  : network->input_width;
    input_channel = prev_layer ? prev_layer->output_channel : network->input_channel;

    input_size = input_height * input_width * input_channel;

    output_height = input_height * stride;
    output_width  = input_width  * stride;
    output_channel = input_channel;
    
    output_size = output_height * output_width * output_channel;

    output_data = new float[output_size * network->batch_size]();
    delta       = new float[output_size * network->batch_size]();

#ifdef GPU_ENABLED
    cudaMalloc((void**)&output_data_dev, output_size * network->batch_size * sizeof(float));
    cudaMalloc((void**)&delta_dev, output_size * network->batch_size * sizeof(float));
    cudaMemset(output_data_dev, 0.0, output_size * network->batch_size * sizeof(float));
    cudaMemset(delta_dev, 0.0, output_size * network->batch_size * sizeof(float));
#endif

}

void upsample_layer_t::init_weight(std::fstream &m_weight_file) {
    // Nothing to do
}

void upsample_layer_t::init_weight() {
    // Nothing to do
}

void upsample_layer_t::forward() {
    memset(output_data, 0.0, output_size * network->batch_size * sizeof(float));
    memset(delta, 0.0, output_size * network->batch_size * sizeof(float));

    float *input_data = prev_layer ? prev_layer->output_data : network->input_data;
    for(unsigned b = 0; b < network->batch_size; b++) {
        for(unsigned c = 0; c < output_channel; c++) {
            for(unsigned  h = 0; h < output_height; h++) {
                for(unsigned w = 0; w < output_width; w++) {
                    output_data[b * output_height * output_width * output_channel + c * output_height * output_width + h * output_width + w] = 
                    scale * input_data[b * input_height * input_width * input_channel + c * input_height * input_width + (h/stride)*input_width + w/stride];
                }
            }
        }
    }
}

void upsample_layer_t::backward() {
    float *prev_delta = prev_layer ? prev_layer->delta : NULL;

    for(unsigned b = 0; b < network->batch_size; b++) {
        for(unsigned c = 0; c < output_channel; c++) {
            for(unsigned h = 0; h < output_height; h++) {
                for(unsigned w = 0; w < output_width; w++) {
                    prev_delta[b * input_height * input_width * input_channel + c * input_height * input_width + (h/stride) * input_width + w / stride] +=
                    scale * output_data[b * output_height * output_width * output_channel + c * output_height * output_width + h * output_width * w];
                }
            }
        }
    }
}

void upsample_layer_t::update() {
    // Nothing to do.
}

void upsample_layer_t::store_weight(std::fstream &m_weight_file) {
    // Nothing to do.
}

}
