#include <fstream>
#include "config.h"
#include "layer.h"
#include "network.h"

namespace nebula {

network_t::network_t() :
    run_type(UNDEFINED_RUN),
    num_threads(1),
    learning_rate(0.001),
    momentum(0.9),
    decay(0.0001),
    cost(0.0),
    reference_label(NULL),
    input_height(1),
    input_width(1),
    input_channel(1),
    input_size(0),
    batch_size(1),
    time_step(1),
    num_layers(0),
    input_data(NULL),
    input_label(NULL),
    input_layer(NULL),
    output_layer(NULL),
#ifdef PRUNING
    weight_threshold(0.0),
    data_threshold(0.0),
#endif
    num_classes(0),
    num_iterations(0),
    iteration(0),
    top_k(1),
	pipe_index(0),
    cumulative_cost(0.0) {
}

network_t::~network_t() {
}

void network_t::forward() {
    //for(unsigned i = 0; i < num_layers; i++) { layers[i]->forward(); }
    for(unsigned i = 0; i < num_layers; i++) { 
        layers[i]->forward(); 
    }
}

void network_t::backward() {
    for(unsigned i = num_layers; i > 0; i--) { layers[i-1]->backward(); }
}

void network_t::update() {
    for(unsigned i = 0; i < num_layers; i++) { layers[i]->update(); }
}

// Initialize network.
void network_t::init(const std::string m_network_config) {
    std::cout << "Initializing network ..." << std::endl;

    npu_mmu::init();
    // Initialize network.
    init_network(m_network_config);

    // Initialize input data.
    //init_data(m_network_config);

    // Initialize weight.
    //init_weight(m_network_config);
}

// Initialize weight.
void network_t::init_weight(const std::string m_input_weight) {
    if(m_input_weight.size()) {
        // Initialize weight from file.
        std::fstream weight_file;
        weight_file.open(m_input_weight.c_str(), std::fstream::in|std::fstream::binary);
        if(!weight_file.is_open()) {
            std::cerr << "Error: failed to open " << m_input_weight << std::endl;
            exit(1);
        }
        
        for(unsigned i = 0; i < layers.size(); i++) { layers[i]->init_weight(weight_file); }
        weight_file.close();
    }
    else {
        // Inference must have input weight.
        if(run_type == TEST_RUN) {
            std::cerr << "Error: missing input weight for " << run_type_str[run_type] << std::endl;
            exit(1);
        }

        // Initialize weight from scratch.
        for(unsigned i = 0; i < layers.size(); i++) { layers[i]->init_weight(); }
    }
}

// Store weight.
void network_t::store_weight(const std::string m_output_weight) {
    // Skip storing weight if output weight file is not given.
    if(!m_output_weight.size()) { return; }

    std::fstream weight_file;
    weight_file.open(m_output_weight.c_str(), std::fstream::out|std::fstream::binary);
    if(!weight_file.is_open()) {
        std::cerr << "Error: failed to open " << m_output_weight << std::endl;
        exit(1);
    }

    for(unsigned i = 0; i < layers.size(); i++) {
        layers[i]->store_weight(weight_file);
    }

    weight_file.close();
}

}
// End of namespace nebula.
