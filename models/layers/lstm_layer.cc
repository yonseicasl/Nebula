#include <cmath>
#include <fstream>
#include <cstring>
#ifndef CUSTOM_BLAS
    #include <cblas.h>
#endif
#include "gemm.h"
#include "lstm_layer.h"
#include "activations.h"

namespace nebula {

lstm_layer_t::lstm_layer_t(network_t *m_network, layer_t *m_prev_layer, layer_type_t m_layer_type) :
    layer_t(m_network, m_prev_layer, m_layer_type),
    cell_state(NULL),
    current_cell_state(NULL),
    cell_delta(NULL),
    hidden_state(NULL),

    input_gate(NULL),
    forget_gate(NULL),
    cell_gate(NULL),
    output_gate(NULL),
    batch_normalize(false){
}

lstm_layer_t::~lstm_layer_t() {
    delete [] output_data;
    delete [] delta;
    delete [] cell_state;
    delete [] current_cell_state;
    delete [] cell_delta;

    delete [] hidden_state;

    delete [] input_gate;
    delete [] forget_gate;
    delete [] cell_gate;
    delete [] output_gate; 


    delete input_gate_W;
    delete forget_gate_W;
    delete cell_gate_W;
    delete output_gate_W;
    
    delete input_gate_U;
    delete forget_gate_U;
    delete cell_gate_U;
    delete output_gate_U;
}

void lstm_layer_t::init(section_config_t m_section_config) {
    // Get layer settings.
    m_section_config.get_setting("output", &output_size);
    m_section_config.get_setting("batch_normalize", &batch_normalize);
    
    std::string activation_str;
    if(m_section_config.get_setting("activation", &activation_str)) {
        activation_type = (activation_type_t)get_type(activation_type_str, activation_str);
    }

    // Initialize layer parameters.
    input_size = prev_layer ? prev_layer->output_size : network->input_size;

    input_gate_W  = new connected_layer_t(network, this , CONNECTED_LAYER);
    forget_gate_W = new connected_layer_t(network, this, CONNECTED_LAYER);
    cell_gate_W   = new connected_layer_t(network, this, CONNECTED_LAYER);
    output_gate_W = new connected_layer_t(network, this, CONNECTED_LAYER);
    
    input_gate_U  = new connected_layer_t(network, 
                                          prev_layer ? prev_layer : NULL, CONNECTED_LAYER);
    forget_gate_U = new connected_layer_t(network, 
                                          prev_layer ? prev_layer : NULL, CONNECTED_LAYER);
    cell_gate_U   = new connected_layer_t(network, 
                                          prev_layer ? prev_layer : NULL, CONNECTED_LAYER);
    output_gate_U = new connected_layer_t(network, 
                                          prev_layer ? prev_layer : NULL, CONNECTED_LAYER);

    // Initialize each gate 
    unsigned temp_size = prev_layer ? prev_layer-> output_size : network->input_size;
    if(prev_layer) prev_layer->output_size = output_size;
    else network->input_size = output_size;

    // Initialize each gate.
    input_gate_W->init(m_section_config);
    forget_gate_W->init(m_section_config);
    cell_gate_W->init(m_section_config);
    output_gate_W->init(m_section_config);

    if(prev_layer) prev_layer->output_size = temp_size;
    else network->input_size = temp_size;
    input_gate_U->init(m_section_config);
    forget_gate_U->init(m_section_config);
    cell_gate_U->init(m_section_config);
    output_gate_U->init(m_section_config);

    output_data        = new float[output_size * network->batch_size]();
    delta              = new float[output_size * network->batch_size]();
    
    cell_state         = new float[output_size * network->batch_size]();
    current_cell_state  = new float[output_size * network->batch_size / network->time_step]();
    cell_delta         = new float[output_size * network->batch_size / network->time_step]();
    
    hidden_state       = new float[output_size * network->batch_size / network->time_step]();

    input_gate         = new float[output_size * network->batch_size / network->time_step]();
    forget_gate        = new float[output_size * network->batch_size / network->time_step]();
    next_forget_gate   = new float[output_size * network->batch_size / network->time_step]();
    cell_gate          = new float[output_size * network->batch_size / network->time_step]();
    output_gate        = new float[output_size * network->batch_size / network->time_step]();
}

void lstm_layer_t::init_weight(std::fstream &m_input_weight) {

    input_gate_W->init_weight(m_input_weight);
    forget_gate_W->init_weight(m_input_weight);
    output_gate_W->init_weight(m_input_weight);
    cell_gate_W->init_weight(m_input_weight);

    input_gate_U->init_weight(m_input_weight);
    forget_gate_U->init_weight(m_input_weight);
    output_gate_U->init_weight(m_input_weight);
    cell_gate_U->init_weight(m_input_weight);

}

void lstm_layer_t::init_weight() {
    input_gate_W->init_weight();
    forget_gate_W->init_weight();
    output_gate_W->init_weight();
    cell_gate_W->init_weight();

    input_gate_U->init_weight();
    forget_gate_U->init_weight();
    output_gate_U->init_weight();
    cell_gate_U->init_weight();
}

void lstm_layer_t::store_weight(std::fstream &m_weight_file) {

    input_gate_W->store_weight(m_weight_file);
    forget_gate_W->store_weight(m_weight_file);
    output_gate_W->store_weight(m_weight_file);
    cell_gate_W->store_weight(m_weight_file);

    input_gate_U->store_weight(m_weight_file);
    forget_gate_U->store_weight(m_weight_file);
    output_gate_U->store_weight(m_weight_file);
    cell_gate_U->store_weight(m_weight_file);
}

void lstm_layer_t::forward() {
    network->batch_size /= network->time_step;

    if(network->run_type == TRAIN_RUN) {
        memset(delta, 0.0, output_size * network->batch_size * network->time_step * sizeof(float));
    }

    connected_layer_t *t_input_gate_W = input_gate_W;
    connected_layer_t *t_forget_gate_W = forget_gate_W;
    connected_layer_t *t_cell_gate_W = cell_gate_W;
    connected_layer_t *t_output_gate_W = output_gate_W;

    connected_layer_t *t_input_gate_U = input_gate_U;
    connected_layer_t *t_forget_gate_U = forget_gate_U;
    connected_layer_t *t_cell_gate_U = cell_gate_U;
    connected_layer_t *t_output_gate_U = output_gate_U;

    for(unsigned step = 0; step < network->time_step; step++) {

        t_input_gate_W->forward(hidden_state);
        t_forget_gate_W->forward(hidden_state);
        t_cell_gate_W->forward(hidden_state);
        t_output_gate_W->forward(hidden_state);
        
        t_input_gate_U->forward(0);
        t_forget_gate_U->forward(0);
        t_cell_gate_U->forward(0);
        t_output_gate_U->forward(0);

        
        // Copy the value from input data to each gate.
        memcpy(input_gate, t_input_gate_U->output_data, 
               output_size * network->batch_size * sizeof(float));
        memcpy(forget_gate, t_forget_gate_U->output_data, 
               output_size * network->batch_size * sizeof(float));
        memcpy(cell_gate, t_cell_gate_U->output_data, 
               output_size * network->batch_size * sizeof(float));
        memcpy(output_gate, t_output_gate_U->output_data, 
               output_size * network->batch_size * sizeof(float));

#ifdef CUSTOM_BLAS
        axpy(output_size * network->batch_size, 1.0, t_input_gate_W->output_data, 1, input_gate, 1);
        axpy(output_size * network->batch_size, 1.0, t_forget_gate_W->output_data, 1, forget_gate, 1);
        axpy(output_size * network->batch_size, 1.0, t_cell_gate_W->output_data, 1, cell_gate, 1);
        axpy(output_size * network->batch_size, 1.0, t_output_gate_W->output_data, 1, output_gate, 1);
#else
        cblas_saxpy(output_size * network->batch_size, 1.0, t_input_gate_W->output_data, 1, input_gate, 1);
        cblas_saxpy(output_size * network->batch_size, 1.0, t_forget_gate_W->output_data, 1, forget_gate, 1);
        cblas_saxpy(output_size * network->batch_size, 1.0, t_cell_gate_W->output_data, 1, cell_gate, 1);
        cblas_saxpy(output_size * network->batch_size, 1.0, t_output_gate_W->output_data, 1, output_gate, 1);
#endif

        /***activate all gates.***/
        logistic_activation(input_gate, output_size * network->batch_size);
        logistic_activation(forget_gate, output_size * network->batch_size);
        logistic_activation(output_gate, output_size * network->batch_size);
        tanh_activation(cell_gate, output_size * network->batch_size);

        axmy(output_size * network->batch_size, 1.0, cell_gate, 1, input_gate, 1);

        if(step) {
            memcpy(cell_state, cell_state - output_size * network->batch_size,
                   output_size * network->batch_size * sizeof(float));
        }
        axmy(output_size * network->batch_size, 1.0, forget_gate, 1, cell_state, 1);

#ifdef CUSTOM_BLAS
        axpy(output_size * network->batch_size, 1.0, input_gate, 1, cell_state, 1);
#else
        cblas_saxpy(output_size * network->batch_size, 1.0, input_gate, 1, cell_state, 1);
#endif
        memcpy(hidden_state, cell_state,
               output_size * network->batch_size * sizeof(float));
        tanh_activation(hidden_state, output_size * network->batch_size);
        
        axmy(output_size * network->batch_size, 1.0, output_gate, 1, hidden_state, 1);

        memcpy(output_data, hidden_state,
               output_size * network->batch_size * sizeof(float));

        /*go to next time step*/
        if(prev_layer) {
            prev_layer->output_data += prev_layer->output_size * network->batch_size;
        }
        else {
            network->input_data += network->input_size * network->batch_size;
        }
        output_data += output_size * network->batch_size;
        cell_state += output_size * network->batch_size;

        t_input_gate_W->increment(1);
        t_forget_gate_W->increment(1);
        t_cell_gate_W->increment(1);
        t_output_gate_W->increment(1);

        t_input_gate_U->increment(1);
        t_forget_gate_U->increment(1);
        t_cell_gate_U->increment(1);
        t_output_gate_U->increment(1);
    }

    // Go back to time step 0.
    if(prev_layer) {
        prev_layer->output_data -= prev_layer->output_size * network->batch_size * network->time_step;
    }
    else {
        network->input_data -= network->input_size * network->batch_size * network->time_step;
    }

    output_data -= output_size * network->batch_size * network->time_step;
    cell_state -= output_size * network->batch_size * network->time_step;

    network->batch_size *= network->time_step;
}

void lstm_layer_t::backward() {
    network->batch_size /= network->time_step;

    connected_layer_t *t_input_gate_W = input_gate_W;
    connected_layer_t *t_forget_gate_W = forget_gate_W;
    connected_layer_t *t_cell_gate_W = cell_gate_W;
    connected_layer_t *t_output_gate_W = output_gate_W;

    connected_layer_t *t_input_gate_U = input_gate_U;
    connected_layer_t *t_forget_gate_U = forget_gate_U;
    connected_layer_t *t_cell_gate_U = cell_gate_U;
    connected_layer_t *t_output_gate_U = output_gate_U;

    if(prev_layer) {
        prev_layer->output_data += prev_layer->output_size * network->batch_size * network->time_step;
        prev_layer->delta += prev_layer->output_size * network->batch_size * network->time_step;
    }
    else {
        network->input_data += network->input_size * network->batch_size * network->time_step;
    }
    t_input_gate_W->increment(network->time_step);
    t_forget_gate_W->increment(network->time_step);
    t_cell_gate_W->increment(network->time_step);
    t_output_gate_W->increment(network->time_step);

    t_input_gate_U->increment(network->time_step);
    t_forget_gate_U->increment(network->time_step);
    t_cell_gate_U->increment(network->time_step);
    t_output_gate_U->increment(network->time_step);

    for(int step = network->time_step - 1; step >= 0; step--) {
        if(prev_layer) {
            prev_layer->output_data -= prev_layer->output_size * network->batch_size;
            prev_layer->delta -= prev_layer->output_size * network->batch_size;
        }
        else {
            network->input_data -= network->input_size * network->batch_size;
        }
        output_data -= output_size * network->batch_size;
        delta       -= output_size * network->batch_size;
        cell_state  -= output_size * network->batch_size;

        t_input_gate_W->increment(-1);
        t_forget_gate_W->increment(-1);
        t_cell_gate_W->increment(-1);
        t_output_gate_W->increment(-1);

        t_input_gate_U->increment(-1);
        t_forget_gate_U->increment(-1);
        t_cell_gate_U->increment(-1);
        t_output_gate_U->increment(-1);

        memcpy(current_cell_state, cell_state,
               output_size * network->batch_size * sizeof(float));

        memcpy(forget_gate, t_forget_gate_W->output_data,
               output_size * network->batch_size * sizeof(float));

        memcpy(input_gate, t_input_gate_W->output_data, 
               output_size * network->batch_size * sizeof(float));

        memcpy(cell_gate, t_cell_gate_W->output_data, 
               output_size * network->batch_size * sizeof(float));

        memcpy(output_gate, t_output_gate_W->output_data,
               output_size * network->batch_size * sizeof(float));
#ifdef CUSTOM_BLAS
        axpy(output_size * network->batch_size, 1.0, t_forget_gate_U->output_data, 1, forget_gate, 1);
        axpy(output_size * network->batch_size, 1.0, t_input_gate_U->output_data, 1, input_gate, 1);
        axpy(output_size * network->batch_size, 1.0, t_cell_gate_U->output_data, 1, cell_gate, 1);
        axpy(output_size * network->batch_size, 1.0, t_output_gate_U->output_data, 1, output_gate, 1);
#else
        cblas_saxpy(output_size * network->batch_size, 1.0, t_forget_gate_U->output_data, 1, forget_gate, 1);
        cblas_saxpy(output_size * network->batch_size, 1.0, t_input_gate_U->output_data, 1, input_gate, 1);
        cblas_saxpy(output_size * network->batch_size, 1.0, t_cell_gate_U->output_data, 1, cell_gate, 1);
        cblas_saxpy(output_size * network->batch_size, 1.0, t_output_gate_U->output_data, 1, output_gate, 1);
#endif

        logistic_activation(forget_gate, output_size * network->batch_size);
        logistic_activation(input_gate, output_size * network->batch_size);
        logistic_activation(output_gate, output_size * network->batch_size);
        tanh_activation(cell_gate, output_size * network->batch_size);

        tanh_activation(current_cell_state, output_size * network->batch_size);

        memcpy(cell_delta, delta, 
               output_size * network->batch_size * sizeof(float));
        
        axmy(output_size * network->batch_size, 1.0, output_gate, 1, cell_delta, 1);

        tanh_gradient(current_cell_state, cell_delta, output_size * network->batch_size);

#ifdef CUSTOM_BLAS
        axpy(output_size * network->batch_size, 1.0, next_forget_gate, 1, cell_delta, 1);
#else
        cblas_saxpy(output_size * network->batch_size, 1.0, next_forget_gate, 1, cell_delta, 1);
#endif

        float *prev_output_data = step ? output_data - output_size * network->batch_size : 0;
        float *prev_delta = delta;

        memcpy(t_cell_gate_W->delta, cell_delta,
               output_size * network->batch_size * sizeof(float));
       
        axmy(output_size * network->batch_size, 1.0, input_gate, 1, t_cell_gate_W->delta, 1);
        tanh_gradient(cell_gate, t_cell_gate_W->delta, output_size * network->batch_size);
        memcpy(t_cell_gate_U->delta, t_cell_gate_W->delta,
               output_size * network->batch_size * sizeof(float));
        t_cell_gate_W->backward(prev_output_data, prev_delta);
        t_cell_gate_U->backward(0, 0);

        memcpy(input_gate_W->delta, cell_delta,
               output_size * network->batch_size * sizeof(float));
        axmy(output_size * network->batch_size, 1.0, cell_gate, 1, input_gate_W->delta, 1);
        logistic_gradient(input_gate, input_gate_W->delta, output_size * network->batch_size);

        memcpy(input_gate_U->delta, input_gate_W->delta, 
               output_size * network->batch_size * sizeof(float));
        input_gate_W->backward(prev_output_data, prev_delta);
        input_gate_U->backward(0, 0);

        memcpy(forget_gate_W->delta, cell_delta,
               output_size * network->batch_size * sizeof(float));

        if(step) {
            axmy(output_size * network->batch_size, 1.0, cell_state - output_size * network->batch_size, 1, forget_gate_W->delta, 1);
        }
                          
        logistic_gradient(forget_gate, forget_gate_W->delta, output_size * network->batch_size);
        memcpy(forget_gate_U->delta, forget_gate_W->delta, 
               output_size * network->batch_size * sizeof(float));

        forget_gate_W->backward(prev_output_data, prev_delta);
        forget_gate_U->backward(0, 0);

        memcpy(output_gate_W->delta, cell_state,
               output_size * network->batch_size * sizeof(float));
        tanh_activation(output_gate_W->delta, output_size * network->batch_size);
        axmy(output_size * network->batch_size, 1.0, delta, 1, output_gate_W->delta, 1);
        logistic_gradient(output_gate, output_gate_W->delta, output_size * network->batch_size);

        memcpy(output_gate_U->delta, output_gate_W->delta,
               output_size * network->batch_size * sizeof(float));

        output_gate_W->backward(prev_output_data, prev_delta);
        output_gate_U->backward(0, 0);

        memcpy(next_forget_gate, forget_gate, 
               output_size * network->batch_size * sizeof(float));
        axmy(output_size * network->batch_size, 1.0, cell_delta, 1, next_forget_gate, 1);
    }
    network->batch_size *= network->time_step;
}

void lstm_layer_t::update() {

    input_gate_W->update();
    forget_gate_W->update();
    cell_gate_W->update();
    output_gate_W->update();

    input_gate_U->update();
    forget_gate_U->update();
    cell_gate_U->update();
    output_gate_U->update();
}

}
// End of namespace nebula.
