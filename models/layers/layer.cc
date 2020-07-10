#ifndef CUSTOM_BLAS
    #include <cblas.h>
#endif
#include <cmath>
#include <thread>
#include <cstring>
#include <functional>
#include "layer.h"
#include "activations.h"

layer_t::layer_t(network_t *m_network, layer_t *m_prev_layer, layer_type_t m_layer_type) :
    layer_type(m_layer_type),
    activation_type(UNDEFINED_ACTIVATION),
    input_height(1),
    input_width(1),
    input_channel(1),
    input_size(1),
    output_height(1),
    output_width(1),
    output_channel(1),
    output_size(0),
    output_data(NULL),
    delta(NULL),
    padding(0),
    num_filters(1),
    filter_size(1),
    stride(1), 
	group(1),
    prev_layer(m_prev_layer),
    next_layer(NULL),
#ifdef GPU_ENABLED
    output_data_dev(NULL),
    delta_dev(NULL),
#endif
    num_threads(1),
    network(m_network) {
    num_threads = network->num_threads;
    if(m_prev_layer) { m_prev_layer->next_layer = this; }
}

// Destructor.
layer_t::~layer_t() {
    // Nothing to do.
}

// Activation function
void layer_t::activate() {
    switch(activation_type) {
        case ELU_ACTIVATION: { 
            elu_activation(output_data, output_size * network->batch_size);
            break;
        }
        case HARDTAN_ACTIVATION: {
            hardtan_activation(output_data, output_size * network->batch_size);
            break;
        }
        case LEAKY_ACTIVATION: { 
            leaky_activation(output_data, output_size * network->batch_size);
            break;
        }
        case LHTAN_ACTIVATION: {
            lhtan_activation(output_data, output_size * network->batch_size);
            break;
        }
        case LINEAR_ACTIVATION: { 
            // Nothing to do
            break;
        }
        case LOGGY_ACTIVATION: { 
            loggy_activation(output_data, output_size * network->batch_size);
            break;
        }
        case LOGISTIC_ACTIVATION: { 
            logistic_activation(output_data, output_size * network->batch_size);
            break;
        }
        case PLSE_ACTIVATION: {
            plse_activation(output_data, output_size * network->batch_size);
            break;
        }
        case RAMP_ACTIVATION: { 
            ramp_activation(output_data, output_size * network->batch_size);
            break;
        }
        case RELIE_ACTIVATION: { 
            relie_activation(output_data, output_size * network->batch_size);
            break;
        }
        case RELU_ACTIVATION: { 
            relu_activation(output_data, output_size * network->batch_size);
            break;
        }
        case STAIR_ACTIVATION: {
            stair_activation(output_data, output_size * network->batch_size);
            break;
        }
        case TANH_ACTIVATION: { 
            tanh_activation(output_data, output_size * network->batch_size);
            break;
        }
        default : {
            std::cerr << "Error: undefined activation type "
                      << activation_type_str[activation_type] << std::endl;
            exit(1);
        }
    }
}

// Gradient function
void layer_t::gradient() {
    switch(activation_type) {
        case ELU_ACTIVATION: { 
            elu_gradient(delta, output_data, output_size * network->batch_size);
            break;
        }
        case HARDTAN_ACTIVATION: { 
            hardtan_gradient(delta, output_data, output_size * network->batch_size);
            break;
        }
	    case LEAKY_ACTIVATION: { 
            leaky_gradient(delta, output_data, output_size * network->batch_size);
            break;
        }
        case LHTAN_ACTIVATION: {
            lhtan_gradient(delta, output_data, output_size * network->batch_size);
            break;
        }
        case LINEAR_ACTIVATION: {
            // Nothing to do
            break;
        }
        case LOGGY_ACTIVATION: { 
            loggy_gradient(delta, output_data, output_size * network->batch_size);
            break;
        }
        case LOGISTIC_ACTIVATION: { 
            logistic_gradient(delta, output_data, output_size * network->batch_size);
            break;
        }
        case PLSE_ACTIVATION: { 
            plse_gradient(delta, output_data, output_size * network->batch_size);
            break;
        }
        case RAMP_ACTIVATION: { 
            ramp_gradient(delta, output_data, output_size * network->batch_size);
            break;
        }
	    case RELIE_ACTIVATION: { 
            relie_gradient(delta, output_data, output_size * network->batch_size);
            break;
        }
        case RELU_ACTIVATION: { 
            relu_gradient(delta, output_data, output_size * network->batch_size);
            break;
        }
        case STAIR_ACTIVATION: { 
            stair_gradient(delta, output_data, output_size * network->batch_size);
            break;
        }
        case TANH_ACTIVATION: { 
            tanh_gradient(delta, output_data, output_size * network->batch_size);
            break;
        }
        default : {
            std::cerr << "Error: undefined activation type "
                      << activation_type_str[activation_type] << std::endl;
            exit(1);
        }
    }
}

