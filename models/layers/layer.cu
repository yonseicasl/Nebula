extern "C++" {
#include "layer.h"
#include "activations.h"
}

// Activate function
extern "C++" void layer_t::_activate_() {
    switch(activation_type) {
        case ELU_ACTIVATION: {
            _elu_activation_(output_data_dev, network->batch_size * output_size);
            break;
        }
        case HARDTAN_ACTIVATION: {
            _hardtan_activation_(output_data_dev, network->batch_size * output_size);
            break;
        }
        case LEAKY_ACTIVATION: {
            _leaky_activation_(output_data_dev, network->batch_size * output_size);
            break;
        }
        case LHTAN_ACTIVATION: {
            _lhtan_activation_(output_data_dev, network->batch_size * output_size);
            break;
        }
        case LINEAR_ACTIVATION: {
            // Nothing to do
            break;
        }
        case LOGGY_ACTIVATION: {
            _loggy_activation_(output_data_dev, network->batch_size * output_size);
            break;
        }
        case LOGISTIC_ACTIVATION: {
            _logistic_activation_(output_data_dev, network->batch_size * output_size);
            break;
        }
        case PLSE_ACTIVATION: {
            _plse_activation_(output_data_dev, network->batch_size * output_size);
            break;
        }
        case RAMP_ACTIVATION: {
            _ramp_activation_(output_data_dev, network->batch_size * output_size);
            break;
        }
        case RELIE_ACTIVATION: {
            _relie_activation_(output_data_dev, network->batch_size * output_size);
            break;
        }
        case RELU_ACTIVATION: {
            _relu_activation_(output_data_dev, network->batch_size * output_size);
            break;
        }
        case STAIR_ACTIVATION: {
            _stair_activation_(output_data_dev, network->batch_size * output_size);
            break;
        }
        case TANH_ACTIVATION: {
            _tanh_activation_(output_data_dev, network->batch_size * output_size);
            break;
        }
        default: {
            std::cerr << "Error: undefined activation type "
                      << activation_type_str[activation_type] << std::endl;
            exit(1);
        }
    }
}


// Gradient function
extern "C++" void layer_t::_gradient_() {
    switch(activation_type) {
        case ELU_ACTIVATION: {
            _elu_gradient_(output_data_dev, delta_dev, network->batch_size * output_size);
            break;
        }
        case HARDTAN_ACTIVATION: {
            _hardtan_gradient_(output_data_dev, delta_dev, network->batch_size * output_size);
            break;
        }
        case LEAKY_ACTIVATION: {
            _leaky_gradient_(output_data_dev, delta_dev, network->batch_size * output_size);
            break;
        }
        case LHTAN_ACTIVATION: {
            _lhtan_gradient_(output_data_dev, delta_dev, network->batch_size * output_size);
            break;
        }
        case LINEAR_ACTIVATION: {
            // Nothing to do
            break;
        }
        case LOGGY_ACTIVATION: {
            _loggy_gradient_(output_data_dev, delta_dev, network->batch_size * output_size);
            break;
        }
        case LOGISTIC_ACTIVATION: {
            _logistic_gradient_(output_data_dev, delta_dev, network->batch_size * output_size);
            break;
        }
        case PLSE_ACTIVATION: {
            _plse_gradient_(output_data_dev, delta_dev, network->batch_size * output_size);
            break;
        }
        case RAMP_ACTIVATION: {
            _ramp_gradient_(output_data_dev, delta_dev, network->batch_size * output_size);
            break;
        }
        case RELIE_ACTIVATION: {
            _relie_gradient_(output_data_dev, delta_dev, network->batch_size * output_size);
            break;
        }
        case RELU_ACTIVATION: {
            _relu_gradient_(output_data_dev, delta_dev, network->batch_size * output_size);
            break;
        }
        case STAIR_ACTIVATION: {
            _stair_gradient_(output_data_dev, delta_dev, network->batch_size * output_size);
            break;
        }
        case TANH_ACTIVATION: {
            _tanh_gradient_(output_data_dev, delta_dev, network->batch_size * output_size);
            break;
        }
        default: {
            std::cerr << "Error: undefined activation type "
                      << activation_type_str[activation_type] << std::endl;
            exit(1);
        }
    }
}

