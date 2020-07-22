extern "C++" {
#include "rnn_layer.h"
#include "gemm.h"
}

namespace nebula {

extern "C++" void rnn_layer_t::_forward_() {
    network->batch_size /= network->time_step;

    if(network->run_type == TRAIN_RUN) {
        cudaMemset(delta_dev, 0.0, output_size * network->batch_size * sizeof(float));
        cudaMemcpy(prev_state_dev, state_dev, 
                   output_size * network->batch_size * sizeof(float), cudaMemcpyDeviceToDevice);
    }
    
    for(unsigned step = 0; step < network->time_step; step++) {

        input_gate->_forward_();

        // Forward propagation of hidden layer in rnn layer.
        if(step) {hidden_gate->_forward_(state_dev);}
        else {hidden_gate->_forward_();}

        cudaMemset(state_dev, 0.0, output_size * network->batch_size * sizeof(float));

        const float alpha = 1.0;

        //Add input gate and hidden gate.
#ifdef CUSTOM_BLAS
        _axpy_(output_size * network->batch_size, alpha, 
               input_gate->output_data_dev, 1, state_dev, 1);
        _axpy_(output_size * network->batch_size, alpha, 
               hidden_gate->output_data_dev, 1, state_dev, 1);
#else
        cublasSaxpy(network->cublas_handle, output_size * network->batch_size, &alpha, 
                    input_gate->output_data_dev, 1, state_dev, 1);
        cublasSaxpy(network->cublas_handle, output_size * network->batch_size, &alpha, 
                    hidden_gate->output_data_dev, 1, state_dev, 1); 
#endif
        output_gate->_forward_(state_dev);
        // Jump to next time step.
        if(prev_layer) {
            prev_layer->output_data_dev += prev_layer->output_size * network->batch_size;
        }
        else {
            network->input_data_dev += network->input_size * network->batch_size;
        }
    
        input_gate->_increment_(1);
        hidden_gate->_increment_(1);
        output_gate->_increment_(1);
    }
    
    if(prev_layer) { prev_layer->output_data_dev -= prev_layer->output_size * network->batch_size * network->time_step; }
    else { network->input_data_dev -= network->input_size * network->batch_size * network->time_step;}

    input_gate->_increment_(-network->time_step);
    hidden_gate->_increment_(-network->time_step);
    output_gate->_increment_(-network->time_step);

    network->batch_size *= network->time_step;

}


extern "C++" void rnn_layer_t::_backward_() {

    network->batch_size /= network->time_step;
    const float alpha = 1.0; 
    connected_layer_t *t_input_gate = input_gate;
    connected_layer_t *t_hidden_gate = hidden_gate;


    t_input_gate->_increment_(network->time_step);
    t_hidden_gate->_increment_(network->time_step);
    
    if(prev_layer) { prev_layer->output_data_dev += prev_layer->output_size * network->batch_size * network->time_step; }
    else { network->input_data_dev += network->input_size * network->batch_size * network->time_step; }

    for(int step = network->time_step -1; step >=0; step--) {

        t_input_gate->_increment_(-1);
        t_hidden_gate->_increment_(-1);

        if(prev_layer) { prev_layer->output_data_dev -= prev_layer->output_size * network->batch_size; }
        else { network->input_data_dev -= network->input_size * network->batch_size; }

        cudaMemset(state_dev, 0.0, output_size * network->batch_size * sizeof(float));
#ifdef _CUSTOM_BLAS
        _axpy_(output_size * network->batch_size, alpha, t_input_gate->output_data_dev, 1, state_dev, 1);
        _axpy_(output_size * network->batch_size, alpha, t_hidden_gate->output_data_dev, 1, state_dev, 1);
#else
        cublasSaxpy(network->cublas_handle, output_size * network->batch_size, &alpha, 
                    t_input_gate->output_data_dev, 1, state_dev, 1);
        cublasSaxpy(network->cublas_handle, output_size * network->batch_size, &alpha, 
                    t_hidden_gate->output_data_dev, 1, state_dev, 1); 
#endif

        if(step == 0) {
            cudaMemcpy(state_dev, prev_state_dev, 
                       output_size * network->batch_size * sizeof(float), cudaMemcpyDeviceToDevice);
        }
        else {
            cudaMemset(state_dev, 0.0, output_size * network->batch_size * sizeof(float));
#ifdef CUSTOM_BLAS
            _axpy_(output_size * network->batch_size, alpha, t_input_gate->output_data_dev - output_size * network->batch_size, 1, state_dev, 1);
            _axpy_(output_size * network->batch_size, alpha, t_hidden_gate->output_data_dev - output_size * network->batch_size, 1, state_dev, 1);
#else
            cublasSaxpy(network->cublas_handle, output_size * network->batch_size, &alpha, 
                        t_input_gate->output_data_dev - output_size * network->batch_size, 1, state_dev, 1);
            cublasSaxpy(network->cublas_handle, output_size * network->batch_size, &alpha, 
                        t_hidden_gate->output_data_dev - output_size * network->batch_size, 1, state_dev, 1);
#endif
        }

        cudaMemcpy(t_input_gate->delta_dev, t_hidden_gate->delta_dev, 
                   output_size * network->batch_size * sizeof(float), cudaMemcpyDeviceToDevice);

        t_hidden_gate->_backward_(state_dev, ((step > 0) ? t_hidden_gate->delta_dev - output_size * network->batch_size : 0));
         
        t_input_gate->_backward_();
    }
    cudaMemset(state_dev, 0.0, output_size * network->batch_size * sizeof(float));
#ifdef CUSTOM_BLAS
    _axpy_(output_size * network->batch_size, alpha, t_input_gate->output_data_dev, 1, state_dev, 1);
    _axpy_(output_size * network->batch_size, alpha, t_hidden_gate->output_data_dev, 1, state_dev, 1);
#else
    cublasSaxpy(network->cublas_handle, output_size * network->batch_size, 
                &alpha, t_input_gate->output_data_dev, 1, state_dev, 1);
    cublasSaxpy(network->cublas_handle, output_size * network->batch_size, 
                &alpha, t_hidden_gate->output_data_dev, 1, state_dev, 1);
#endif

    network->batch_size *= network->time_step;
}

extern "C++" void rnn_layer_t::_update_() {
    input_gate->_update_();
    hidden_gate->_update_();
}

}
// End of namespace nebula
