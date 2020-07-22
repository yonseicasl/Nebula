#include <fstream>
#include <cstring>
#ifndef CUSTOM_BLAS
    #include <cblas.h>
#endif
#ifdef GPU_ENABLED
#include <cuda_runtime.h>
#endif
#include "rnn_layer.h"
#include "gemm.h"

namespace nebula {

rnn_layer_t::rnn_layer_t(network_t *m_network, layer_t *m_prev_layer, layer_type_t m_layer_type) :
    layer_t(m_network, m_prev_layer, m_layer_type),
#ifdef GPU_ENABLED
    state_dev(NULL),
    prev_state_dev(NULL),
#endif
    state(NULL),
    prev_state(NULL),
    batch_normalize(false) {
}

rnn_layer_t::~rnn_layer_t() {
    delete [] state;
    delete [] prev_state;
#ifdef GPU_ENABLED
    cudaFree(state_dev);
    cudaFree(prev_state_dev);
#endif
}

void rnn_layer_t::init(section_config_t m_section_config) {

    m_section_config.get_setting("output", &output_size);
    m_section_config.get_setting("batch_normalize", &batch_normalize);

    std::string activation_str;
    if(m_section_config.get_setting("activation", &activation_str)) {
        activation_type = (activation_type_t)get_type(activation_type_str, activation_str);
    }
    input_size = prev_layer ? prev_layer->output_size : network->input_size;
    
    state = new float[output_size * network->batch_size / network->time_step]();
    prev_state = new float[output_size * network->batch_size / network->time_step]();

    // Initialize all gates of rnn cell.
    input_gate  = new connected_layer_t(network, prev_layer ? prev_layer : NULL, CONNECTED_LAYER); 
    hidden_gate = new connected_layer_t(network, input_gate, CONNECTED_LAYER); 
    output_gate = new connected_layer_t(network, hidden_gate, CONNECTED_LAYER);

    input_gate->init(m_section_config);
    hidden_gate->init(m_section_config);
    output_gate->init(m_section_config);

    output_data = output_gate->output_data;
    delta = output_gate->delta;

#ifdef GPU_ENABLED
    cudaMalloc((void**)&state_dev, output_size * network->batch_size * sizeof(float) / network->time_step);
    cudaMalloc((void**)&prev_state_dev, output_size * network->batch_size * sizeof(float) / network->time_step);

    cudaMemset(state_dev, 0.0, output_size * network->batch_size * sizeof(float) / network->time_step);
    cudaMemset(prev_state_dev, 0.0, output_size * network->batch_size * sizeof(float) / network->time_step);

    output_data_dev = output_gate->output_data_dev;
    delta_dev = output_gate->delta_dev;
#endif

}

void rnn_layer_t::init_weight(std::fstream &m_input_weight) {
    input_gate->init_weight(m_input_weight);
    hidden_gate->init_weight(m_input_weight);
    output_gate->init_weight(m_input_weight);
}

void rnn_layer_t::init_weight() {
    input_gate->init_weight();
    hidden_gate->init_weight();
    output_gate->init_weight();
}

void rnn_layer_t::store_weight(std::fstream &m_weight_file) {
    input_gate->store_weight(m_weight_file);
    hidden_gate->store_weight(m_weight_file);
    output_gate->store_weight(m_weight_file);
}

void rnn_layer_t::forward() {
    network->batch_size /= network->time_step;

    if(network->run_type == TRAIN_RUN) {
        memset(delta, 0.0, output_size * network->batch_size * sizeof(float));
        memcpy(prev_state, state, output_size * network->batch_size * sizeof(float));
    }
   
    for(unsigned step = 0; step < network->time_step; step++) {
        // Forward propagation of input gate in rnn cell.
        input_gate->forward();

        // Forward propagation of hidden gate in hidden layer.
        if(step) {hidden_gate->forward(state);}
        else{hidden_gate->forward();}
       
        // Add hidden gate and input gate.
        // The result of addition becomes input of output gate in hidden layer.
        memset(state, 0.0, output_size * network->batch_size * sizeof(float));
#ifdef CUSTOM_BLAS
        axpy(output_size * network->batch_size, 1.0, input_gate->output_data, 1, state, 1);
        axpy(output_size * network->batch_size, 1.0, hidden_gate->output_data, 1, state, 1);
#else
        cblas_saxpy(output_size * network->batch_size, 1.0, input_gate->output_data, 1, state, 1);
        cblas_saxpy(output_size * network->batch_size, 1.0, hidden_gate->output_data, 1, state, 1);
#endif
        if(step) {output_gate->forward(state);}
        else {output_gate->forward();}

        if(prev_layer) {
            prev_layer->output_data += prev_layer->output_size * network->batch_size;
        }
        else {
            network->input_data += network->input_size * network->batch_size;
        }

        input_gate->increment(1);
        hidden_gate->increment(1);
        output_gate->increment(1);
    }

    // Move the pointer to the head.
    if(prev_layer) { prev_layer->output_data -= prev_layer->output_size * network->batch_size * network->time_step; }
    else { network->input_data -= network->input_size * network->batch_size * network->time_step; }

    input_gate->increment(-network->time_step);
    hidden_gate->increment(-network->time_step);
    output_gate->increment(-network->time_step);

    network->batch_size *= network->time_step; 
}


void rnn_layer_t::backward() {

    network->batch_size /= network->time_step;

    input_gate->increment(network->time_step);
    hidden_gate->increment(network->time_step);
    output_gate->increment(network->time_step);

    if(prev_layer) { prev_layer->output_data += prev_layer->output_size * network->batch_size * network->time_step; }
    else { network->input_data += network->input_size * network->batch_size * network->time_step; }

    for(int step = network->time_step - 1; step >=0; step--) {
        input_gate->increment(-1);
        hidden_gate->increment(-1);
        output_gate->increment(-1);
        
        if(prev_layer) {
            prev_layer->output_data -= prev_layer->output_size * network->batch_size;
        }
        else {
            network->input_data -= network->input_size * network->batch_size;
        }

        memset(state, 0.0, output_size * network->batch_size * sizeof(float));
        // Add output data of input gate and hidden gate in hidden layer.
        // The sum is input of output gate in hidden layer.
#ifdef CUSTOM_BLAS
        axpy(output_size * network->batch_size, 1.0, 
             input_gate->output_data, 1, state, 1);
        axpy(output_size * network->batch_size, 1.0,
             hidden_gate->output_data, 1, state, 1);
#else
        cblas_saxpy(output_size * network->batch_size, 1.0, 
                    input_gate->output_data, 1, state, 1);
        cblas_saxpy(output_size * network->batch_size, 1.0,
                    hidden_gate->output_data, 1, state, 1);
#endif
        output_gate->backward(state, 0);

        // Backward propagation of output gate of hidden layer.
        if(step == 0) { 
            memcpy(state, prev_state, output_size * network->batch_size * sizeof(float)); 
        }
        else {
            memset(state, 0.0, output_size * network->batch_size * sizeof(float));
#ifdef CUSTOM_BLAS
            axpy(output_size * network->batch_size, 1.0, 
                 input_gate->output_data - output_size * network->batch_size, 1, state, 1);
            axpy(output_size * network->batch_size, 1.0, 
                 hidden_gate->output_data - output_size * network->batch_size, 1, state, 1);
#else
            cblas_saxpy(output_size * network->batch_size, 1, 
                        input_gate->output_data - output_size * network->batch_size, 1, state, 1);
            cblas_saxpy(output_size * network->batch_size, 1, 
                        hidden_gate->output_data - output_size * network->batch_size, 1, state, 1);
#endif
        }

        memcpy(input_gate->delta, hidden_gate->delta, 
               output_size * network->batch_size * sizeof(float));

        // Backward propagation of hidden gate in hidden layer.
        hidden_gate->backward(state, (step > 0) ? 
                                      hidden_gate->delta - output_size * network->batch_size : 0);

        // Backward propagation of input gate in hidden layer.
        input_gate->backward();

    }
    memset(state, 0.0, output_size * network->batch_size * sizeof(float));
#ifdef CUSTOM_BLAS
    axpy(output_size * network->batch_size, 1.0, 
         input_gate->output_data, 1, state, 1);
    axpy(output_size * network->batch_size, 1.0, 
         hidden_gate->output_data, 1, state, 1);
#else
    cblas_saxpy(output_size * network->batch_size, 1, 
                input_gate->output_data, 1, state, 1);
    cblas_saxpy(output_size * network->batch_size, 1, 
                hidden_gate->output_data, 1, state, 1);
#endif

    network->batch_size *= network->time_step;
}

void rnn_layer_t:: update() {
    input_gate->update();
    hidden_gate->update();
    output_gate->update();
}

}
// End of namespace nebula.
