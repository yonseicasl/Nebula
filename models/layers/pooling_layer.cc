#include <algorithm>
#include <functional>
#include <cstring>
#include <thread>

#include "pooling_layer.h"

namespace nebula {

pooling_layer_t::pooling_layer_t(network_t *m_network, layer_t *m_prev_layer, layer_type_t m_layer_type) :
    layer_t(m_network, m_prev_layer, m_layer_type),
    index(NULL) {
}

pooling_layer_t::~pooling_layer_t() {
    delete [] output_data;
    delete [] delta;
    
    if(layer_type == MAXPOOL_LAYER) {
        delete [] index;
    }

}

void pooling_layer_t::init(section_config_t m_section_config) {
    // Get layer setting.
    m_section_config.get_setting("stride", &stride);
    // If the filters size is not set in the configure file,
    // Filter size is equal to stride.
    filter_size = stride;
    m_section_config.get_setting("size", &filter_size);

    padding = (filter_size - 1) / 2;
    m_section_config.get_setting("padding", &padding);
    m_section_config.get_setting("hops", &hops);

    // Set input parameters.
    layer_t *connection = this;
    if(hops > 1) {
        for(unsigned i = 0; i < hops; i++) { connection = connection->prev_layer ? connection->prev_layer : NULL; }
        input_height = connection ? connection->output_height : network->input_height;
        input_width = connection ? connection->output_width : network->input_width;
        input_channel = connection ? connection->output_channel : network->input_channel;
        input_size = connection ? connection->output_size : network->input_size;
        input_data = connection ? connection->output_data : network->input_data;
    }
    else {
        input_height  = prev_layer ? prev_layer->output_height : network->input_height;
        input_width   = prev_layer ? prev_layer->output_width : network->input_width;
        input_channel = prev_layer ? prev_layer->output_channel : network->input_channel;
        input_size = prev_layer ? prev_layer->output_size : network->input_size;
        input_data = prev_layer ? prev_layer->output_data : network->input_data;
    }
    
    // Set output parameters.
    output_height = (input_height + 2 * padding - filter_size) / stride + 1;
    output_width = (input_width + 2 * padding - filter_size) / stride + 1;
    output_channel = input_channel;
    output_size = output_height * output_width * output_channel;
    
    //std::cout << input_height << " * " << input_width << " * " << input_channel << std::endl;
    // Allocate memory for pooling layer.
    output_data = new float[output_size * network->batch_size]();
    npu_mmu::npu_malloc((uint64_t)output_data);
    delta = new float[output_size * network->batch_size]();
    
    if(layer_type == MAXPOOL_LAYER) {
        index = new unsigned[output_size * network->batch_size]();
    }
}

void pooling_layer_t::init_weight(std::fstream &m_weight_file) {
    //Nothing to do
}

void pooling_layer_t::init_weight() {
    //Nothing to do
}

void pooling_layer_t::forward() {
    memset(output_data, 0.0, output_size * network->batch_size * sizeof(float));
    memset(delta, 0.0, output_size * network->batch_size * sizeof(float));
    
    std::cout << "pooling layer" << std::endl;
    std::cout << output_height << "*" << output_width << "*" << output_channel << std::endl;
    input_data = prev_layer ? prev_layer->output_data : network->input_data;
    // Case 1: Max pooling
    // Select the biggest element in the filter.
    if(layer_type == MAXPOOL_LAYER) {
        std::vector<std::thread> threads;
        threads.reserve(num_threads);
        for(unsigned tid = 0; tid < num_threads; tid++) {
            threads.emplace_back(std::bind([&](const unsigned begin, const unsigned end, const unsigned tid) {
                for(unsigned b = begin; b < end; b++) {
                    for(unsigned c = 0; c < output_channel; c++) {
                        for(unsigned h = 0; h < output_height; h++) {
                            for(unsigned w = 0; w < output_width; w++) {
                                unsigned output_index = w + output_width * (h + output_height * (c + output_channel * b));
                                float max_value = -1e8;
                                int max_index = -1;
                                for(unsigned i = 0; i < filter_size; i++) {
                                    for(unsigned j = 0; j < filter_size; j++) {
                                        unsigned current_height = h * stride + i - padding;
                                        unsigned current_width = w * stride + j - padding;
                                        unsigned input_index = current_width + input_width * (current_height + input_height * (c + b * input_channel));
                                        bool valid = ((current_height >= 0) && (current_height < input_height) && (current_width >= 0) && (current_width < input_width));
                                        float val = valid ? input_data[input_index] : -1e8;
                                        max_index = (val > max_value) ? input_index : max_index;
                                        max_value = (val > max_value) ? val : max_value;
                                    }
                                }
                                output_data[output_index] = max_value;
                                index[output_index] = max_index;
                            }
                        }
                    }
                }
            }, tid * network->batch_size / num_threads, (tid + 1) * network->batch_size / num_threads, tid));
        } std::for_each(threads.begin(), threads.end(), [](std::thread& t) { t.join();});
    }
    // Case 2: average pooling
    // Calculate average value of elements in the filter..
    else if(layer_type == AVGPOOL_LAYER) {
        std::vector<std::thread> threads;
        threads.reserve(num_threads);
        for(unsigned tid = 0; tid < num_threads; tid++) {
            threads.emplace_back(std::bind([&](const unsigned begin, const unsigned end, const unsigned tid) {
                for(unsigned b = begin; b < end; b++) {
                    for(unsigned c = 0; c < output_channel; c++) {
                        for(unsigned h = 0; h < output_height; h++) {
                            for(unsigned w = 0; w < output_width; w++) {
                                unsigned output_index = w + output_width * (h + output_height * (c + output_channel * b));
                                float val = 0.0;
                                for(unsigned i = 0; i < filter_size; i++) {
                                    for(unsigned j = 0; j < filter_size; j++) {
                                        unsigned current_height = h * stride + i - padding;
                                        unsigned current_width = w * stride + j - padding;
                                        unsigned input_index = current_width + input_width * (current_height + input_height * (c + b * input_channel));
                                        bool valid = ((current_height >= 0) && (current_height < input_height) && (current_width >= 0) && (current_width < input_width));
                                        val += valid ? input_data[input_index] : 0.0;
                                    }
                                }
                                output_data[output_index] = val / (filter_size * filter_size);
                            }
                        }
                    }
                }
            }, tid * network->batch_size / num_threads, (tid + 1) * network->batch_size / num_threads, tid));
        } std::for_each(threads.begin(), threads.end(), [](std::thread& t) { t.join();});
    }
    else {
        std::cerr << "undefined pooling layer type " << std::endl;
        exit(1);
    }
}

void pooling_layer_t::backward() {
    float *prev_delta = prev_layer ? prev_layer->delta : NULL;
    if(layer_type == MAXPOOL_LAYER) {
        std::vector<std::thread> threads;
        threads.reserve(num_threads);
        for(unsigned tid = 0; tid < num_threads; tid++) {
            threads.emplace_back(std::bind([&](const unsigned begin, const unsigned end, const unsigned tid) {
                for(unsigned i = begin; i < end; i++) { prev_delta[index[i]] += delta[i];}
            }, tid * output_size * network->batch_size / num_threads, 
               (tid + 1) * output_size * network->batch_size / num_threads, tid));
        } std::for_each(threads.begin(), threads.end(), [](std::thread& t) {t.join(); });
    }
    else if(layer_type == AVGPOOL_LAYER) {
        std::vector<std::thread> threads;
        threads.reserve(num_threads);
        for(unsigned tid = 0; tid < num_threads; tid++) {
            threads.emplace_back(std::bind([&](const unsigned begin, const unsigned end, const unsigned tid) {
                for(unsigned i = begin; i < end; i++) {
                        for(unsigned j = 0; j < filter_size * filter_size; j++)
                        {
                            unsigned input_index = filter_size * filter_size * i + j;
                            prev_delta[input_index] += (delta[i] / (filter_size * filter_size)); 
                        }
                    }
            }, tid * output_size * network->batch_size / num_threads, (tid + 1) * output_size * network->batch_size / num_threads, tid));
        } std::for_each(threads.begin(), threads.end(), [](std::thread& t) {t.join(); });
    }
    else {
        std::cerr << "undefined pooling layer type" << std::endl;
        exit(1);
    }
}

void pooling_layer_t::update() {
    //Nothing to do
}

void pooling_layer_t::store_weight(std::fstream &m_weight_file) {
    //Nothing to do
}

}
// End of namespace nebula.
