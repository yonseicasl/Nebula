extern "C++" {
#include "lstm_layer.h"
#include "activations.h"
}

namespace nebula {

__global__ void _multiply_(unsigned m_size, 
                           float *X, unsigned incx, 
                           float *Y, unsigned incy){
    unsigned index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    Y[index * incy] *= X[index * incx];
}

extern "C++" void lstm_layer_t::_forward_() {
    network->batch_size /= network->time_step;

    cudaMemset(delta_dev, 0.0, 
               output_size * network->batch_size * network->time_step * sizeof(float));
    for(unsigned step = 0; step < network->time_step; step++) {

        // Calculate Weight_hidden * Hidden_t-1 
        float *input_state_dev = prev_layer ? prev_layer->output_data_dev : network->input_data_dev;
        if(prev_layer) prev_layer->output_data_dev = hidden_state;
        forget_gate_W->_forward_();
        input_gate_W->_forward_();
        cell_gate_W->_forward_();
        output_gate_W->_forward_();

        // Calculate weight_input * input_t
        if(prev_layer)prev_layer->output_data_dev = input_state_dev;
        forget_gate_U->_forward_(); // W_if * X_t
        input_gate_U->_forward_();  // W_ii * X_t
        cell_gate_U->_forward_();   // W_ic * X_t
        output_gate_U->_forward_(); // W_io * X_t
        
        const float alpha = 1.0;
        // Add output data of input layer and hidden layer to input gate
        // W_hi * H_t-1 + W_ii * X_t
        cudaMemcpy(input_gate_dev, input_gate_U->output_data_dev, 
                   output_size * network->batch_size * sizeof(float), cudaMemcpyDeviceToDevice);
        cublasSaxpy(network->cublas_handle, output_size * network->batch_size, &alpha, 
                    input_gate_W->output_data_dev, 1, input_gate_dev, 1);

        // Add output data of input layer and hiddden layer to forget gate.
        // W_hf * H_t-1 + W_if * X_t
        cudaMemcpy(forget_gate_dev, forget_gate_U->output_data_dev,
                   output_size * network->batch_size * sizeof(float), cudaMemcpyDeviceToDevice);
        cublasSaxpy(network->cublas_handle, output_size * network->batch_size, &alpha,
                    forget_gate_W->output_data_dev, 1, forget_gate_dev, 1);

        // Add output data of input layer and hidden layer to cell gate.
        // W_hc * H_t-1 + W_ic * X_t
        cudaMemcpy(cell_gate_dev, cell_gate_U->output_data_dev,
                   output_size * network->batch_size * sizeof(float), cudaMemcpyDeviceToDevice);
        cublasSaxpy(network->cublas_handle, output_size * network->batch_size, &alpha,
                    cell_gate_W->output_data_dev, 1, cell_gate_dev, 1);

        // Add output data of input layer and hidden layer to output gate.
        // W_ho * H_t-1 + W_io * X_t
        cudaMemcpy(output_gate_dev, output_gate_U->output_data_dev,
                   output_size * network->batch_size * sizeof(float), cudaMemcpyDeviceToDevice);
        cublasSaxpy(network->cublas_handle, output_size * network->batch_size, &alpha,
                    output_gate_W->output_data_dev, 1, output_gate_dev, 1); 
       
        // Activate each gate
       
        dim3 cuda_griddim = {(output_size * network->batch_size - 1) / BLOCK_SIZE + 1, 1, 1};
        // Logistic or sigmoid function.
        _logistic_activation_(input_gate_dev, output_size * network->batch_size);
        _logistic_activation_(forget_gate_dev, output_size * network->batch_size);
        _logistic_activation_(output_gate_dev, output_size * network->batch_size);
        _tanh_activation_(cell_gate_dev, output_size * network->batch_size);

        _multiply_<<<cuda_griddim, BLOCK_SIZE>>>(output_size * network->batch_size,
                                                cell_gate_dev, 1, input_gate_dev, 1);

        // Cell unit of lstm layer.
        if(step) {
            cudaMemcpy(cell_state_dev, cell_state_dev - output_size * network->batch_size,
                       output_size * network->batch_size * sizeof(float), cudaMemcpyDeviceToDevice);
        }
        _multiply_<<<cuda_griddim, BLOCK_SIZE>>>(output_size * network->batch_size,
                                                 forget_gate_dev, 1, cell_state_dev, 1);
        cublasSaxpy(network->cublas_handle, output_size * network->batch_size, &alpha,
                    input_gate_dev, 1, cell_state_dev, 1); 
        // Hidden state of lstm layer.
       
        cudaMemcpy(hidden_state_dev, cell_state_dev, 
                   output_size * network->batch_size * sizeof(float), cudaMemcpyDeviceToDevice);
        _tanh_activation_(hidden_state_dev, output_size * network->batch_size);
        _multiply_<<<cuda_griddim, BLOCK_SIZE>>>(output_size * network->batch_size,
                                                 output_gate_dev, 1, hidden_state_dev, 1);
        // Output data of lstm layer.
        cudaMemcpy(output_data_dev, hidden_state_dev,
                   output_size * network->batch_size * sizeof(float), cudaMemcpyDeviceToDevice);

        // Go to next time step;
        if(prev_layer) {
            prev_layer->output_data_dev += prev_layer->output_size * network->batch_size;
        }
        else {
            network->input_data_dev += network->input_size * network->batch_size;
        }

        output_data_dev  += output_size * network->batch_size;
        cell_state_dev   += output_size * network->batch_size;

        input_gate_W->_increment_(1);
        forget_gate_W->_increment_(1);
        cell_gate_W->_increment_(1);
        output_gate_W->_increment_(1);

        input_gate_U->_increment_(1);
        forget_gate_U->_increment_(1);
        cell_gate_U->_increment_(1);
        output_gate_U->_increment_(1);
    }

    // Back to time step 0.
    if(prev_layer) {
        prev_layer->output_data_dev -= prev_layer->output_size * network->batch_size * network->time_step;
    }
    else {
        network->input_data_dev -= network->input_size * network->batch_size * network->time_step;
    }

    output_data_dev  -= output_size * network->batch_size * network->time_step;
    cell_state_dev   -= output_size * network->batch_size * network->time_step;

    input_gate_W->_increment_(-network->time_step);
    forget_gate_W->_increment_(-network->time_step);
    cell_gate_W->_increment_(-network->time_step);
    output_gate_W->_increment_(-network->time_step);

    input_gate_U->_increment_(-network->time_step);
    forget_gate_U->_increment_(-network->time_step);
    cell_gate_U->_increment_(-network->time_step);
    output_gate_U->_increment_(-network->time_step);

    network->batch_size *= network->time_step;
}


extern "C++" void lstm_layer_t::_backward_() {

    network->batch_size /= network->time_step;

    // Jump to time to network->time_step
    if(prev_layer) {
        prev_layer->output_data_dev += prev_layer->output_size * network->batch_size * network->time_step;
        prev_layer->delta_dev += prev_layer->output_size * network->batch_size * network->time_step;
    }
    else {
        network->input_data_dev += network->input_size * network->batch_size * network->time_step;
    }

    output_data_dev  += output_size * network->batch_size * network->time_step;
    delta_dev        += output_size * network->batch_size * network->time_step;
    cell_state_dev   += output_size * network->batch_size * network->time_step;

    input_gate_W->_increment_(network->time_step);
    forget_gate_W->_increment_(network->time_step);
    cell_gate_W->_increment_(network->time_step);
    output_gate_W->_increment_(network->time_step);
    
    input_gate_U->_increment_(network->time_step);
    forget_gate_U->_increment_(network->time_step);
    cell_gate_U->_increment_(network->time_step);
    output_gate_U->_increment_(network->time_step);

    for(int step = network->time_step-1; step >= 0; step--) {

        if(prev_layer) {
            prev_layer->output_data_dev -= prev_layer->output_size * network->batch_size; 
            prev_layer->delta_dev -= prev_layer->output_size * network->batch_size;
        }
        else {
            network->input_data_dev -= network->input_size * network->batch_size;
        }
        output_data_dev  -= output_size * network->batch_size;
        delta_dev        -= output_size * network->batch_size;
        cell_state_dev   -= output_size * network->batch_size;

        input_gate_W->_increment_(-1);
        forget_gate_W->_increment_(-1);
        cell_gate_W->_increment_(-1);
        output_gate_W->_increment_(-1);

        input_gate_U->_increment_(-1);
        forget_gate_U->_increment_(-1);
        cell_gate_U->_increment_(-1);
        output_gate_U->_increment_(-1);

        cudaMemcpy(current_cell_state_dev, cell_state_dev,
                   output_size * network->batch_size * sizeof(float), cudaMemcpyDeviceToDevice);

        const float alpha = 1.0;
        // Calculate forget gate
        cudaMemcpy(forget_gate_dev, forget_gate_W->output_data_dev, 
                   output_size * network->batch_size * sizeof(float), cudaMemcpyDeviceToDevice);
        cublasSaxpy(network->cublas_handle, output_size * network->batch_size, 
                    &alpha, forget_gate_U->output_data_dev, 1, forget_gate_dev, 1);

        // Calculate input gate.
        cudaMemcpy(input_gate_dev, input_gate_W->output_data_dev, 
                   output_size * network->batch_size * sizeof(float), cudaMemcpyDeviceToDevice);
        cublasSaxpy(network->cublas_handle, output_size * network->batch_size, 
                    &alpha, input_gate_U->output_data_dev, 1, input_gate_dev, 1);

        // Calculate cell gate.
        cudaMemcpy(cell_gate_dev, cell_gate_W->output_data_dev, 
                   output_size * network->batch_size * sizeof(float), cudaMemcpyDeviceToDevice);
        cublasSaxpy(network->cublas_handle, output_size * network->batch_size, 
                    &alpha, cell_gate_U->output_data_dev, 1, cell_gate_dev, 1);

        // Calculate output gate.
        cudaMemcpy(output_gate_dev, output_gate_W->output_data_dev, 
                   output_size * network->batch_size * sizeof(float), cudaMemcpyDeviceToDevice);
        cublasSaxpy(network->cublas_handle, output_size * network->batch_size, 
                    &alpha, output_gate_U->output_data_dev, 1, output_gate_dev, 1);
 
        dim3 cuda_griddim = {(output_size * network->batch_size - 1) / BLOCK_SIZE + 1, 1, 1};
        _logistic_activation_(forget_gate_dev, output_size * network->batch_size);
        _logistic_activation_(input_gate_dev, output_size * network->batch_size);
        _logistic_activation_(output_gate_dev, output_size * network->batch_size);
        _tanh_activation_(cell_gate_dev, output_size * network->batch_size);

        // Calculate delta value of cell state.
        // tanh(cell_state_t)
        _tanh_activation_(current_cell_state_dev, output_size * network->batch_size);

        // d(out_t) * O_t * (1 - tanh^2(cell_state_t))
        cudaMemcpy(cell_delta_dev, delta_dev, 
                   output_size * network->batch_size * sizeof(float), cudaMemcpyDeviceToDevice);
        _multiply_<<<cuda_griddim, BLOCK_SIZE>>>(output_size * network->batch_size, 
                                                 output_gate_dev, 1, cell_delta_dev, 1);
        _tanh_gradient_(current_cell_state_dev, cell_delta_dev, output_size * network->batch_size);
        // d(state_t) = d(out_t) * O_t * (1 - tanh^2(cell_state_t)) + d(cell_state_t+1) * f_t+1
        cublasSaxpy(network->cublas_handle, output_size * network->batch_size,
                    &alpha, next_forget_gate_dev, 1, cell_delta_dev, 1);

        float *prev_output_data_dev = step ? output_data_dev - output_size * network->batch_size : NULL;
        float *prev_delta_dev = step ? delta_dev : NULL;

        // Backpropagation of output gate.
        // d(O_t) = d(delta_t) * tanh(cell_state_t) * O_t * (1 - O_t)
        cudaMemcpy(output_gate_W->delta_dev, cell_state_dev, 
                   output_size * network->batch_size * sizeof(float), cudaMemcpyDeviceToDevice);
        _tanh_activation_(output_gate_W->delta_dev, output_size * network->batch_size);
        _multiply_<<<cuda_griddim, BLOCK_SIZE>>>(output_size * network->batch_size, 
                                                 delta_dev, 1, output_gate_W->delta_dev, 1);
        _logistic_gradient_(output_gate_dev, output_gate_W->delta_dev, output_size * network->batch_size);
        cudaMemcpy(output_gate_U->delta_dev, output_gate_W->delta_dev,
                   output_size * network->batch_size * sizeof(float), cudaMemcpyDeviceToDevice);

        // Backpropagation of cell gate.
        // d(C_t) = d(state_t) * I_t * (1 - C_t^2)
        cudaMemcpy(cell_gate_W->delta_dev, cell_delta_dev, 
                   output_size * network->batch_size * sizeof(float), cudaMemcpyDeviceToDevice);
        _multiply_<<<cuda_griddim, BLOCK_SIZE>>>(output_size * network->batch_size, 
                                                 input_gate_dev, 1, cell_gate_W->delta_dev, 1);
        _tanh_gradient_(cell_gate_dev, cell_gate_W->delta_dev, output_size * network->batch_size);
        cudaMemcpy(cell_gate_U->delta_dev, cell_gate_W->delta_dev,
                   output_size * network->batch_size * sizeof(float), cudaMemcpyDeviceToDevice);

        // Backpropagation of input gate.
        // d(I_t) = d(state_t) * C_t * I_t * (1 - I_t)
        cudaMemcpy(input_gate_W->delta_dev, cell_delta_dev, 
                   output_size * network->batch_size * sizeof(float), cudaMemcpyDeviceToDevice);
        _multiply_<<<cuda_griddim, BLOCK_SIZE>>>(output_size * network->batch_size, 
                                                 cell_gate_dev, 1, input_gate_W->delta_dev, 1);
        _logistic_gradient_(input_gate_dev, input_gate_W->delta_dev, 
                            output_size * network->batch_size);

        cudaMemcpy(input_gate_U->delta_dev, input_gate_W->delta_dev, 
                   output_size * network->batch_size * sizeof(float), cudaMemcpyDeviceToDevice);

        // Backpropagation of forget gate.
        // d(F_t) = d(state_t) * state_t-1 * F_t * (1 - F_t)
      
        if(step) {
            cudaMemcpy(forget_gate_W->delta_dev, cell_delta_dev, 
                       output_size * network->batch_size * sizeof(float), cudaMemcpyDeviceToDevice);
            _multiply_<<<cuda_griddim, BLOCK_SIZE>>>(output_size * network->batch_size, 
                                                     cell_state_dev - output_size * network->batch_size, 1, forget_gate_W->delta_dev, 1);
        }

        _logistic_gradient_(forget_gate_dev, forget_gate_W->delta_dev, output_size * network->batch_size);
        cudaMemcpy(forget_gate_U->delta_dev, forget_gate_W->delta_dev, 
                   output_size * network->batch_size * sizeof(float), cudaMemcpyDeviceToDevice);
        
        float *input_data_dev = prev_layer ? prev_layer->output_data_dev : network->input_data_dev;
        float *input_delta_dev = prev_layer ? prev_layer->delta_dev : NULL;

        if(prev_layer) {
            prev_layer->output_data_dev = prev_output_data_dev;
            prev_layer->delta_dev = prev_delta_dev;
        }
        cell_gate_W->_backward_();
        input_gate_W->_backward_();
        forget_gate_W->_backward_();
        output_gate_W->_backward_();

        if(prev_layer) {
            prev_layer->output_data_dev = input_data_dev;
            prev_layer->delta_dev = input_delta_dev;
        }
        cell_gate_W->_backward_();
        input_gate_W->_backward_();
        forget_gate_W->_backward_();
        output_gate_U->_backward_();

        cudaMemcpy(next_forget_gate_dev, forget_gate_dev,
                   output_size * network->batch_size * sizeof(float), cudaMemcpyDeviceToDevice);
        _multiply_<<<cuda_griddim, BLOCK_SIZE>>>(output_size * network->batch_size,
                                                 cell_delta_dev, 1, next_forget_gate_dev, 1);
    }
    cudaMemcpy(hidden_state_dev, output_data_dev, output_size * network->batch_size * sizeof(float), cudaMemcpyDeviceToHost);
    network->batch_size *= network->time_step;
}

extern "C++" void lstm_layer_t::_update_() {

    network->batch_size /= network->time_step;
    input_gate_W->_update_();
    forget_gate_W->_update_();
    cell_gate_W->_update_();
    output_gate_W->_update_();

    input_gate_U->_update_();
    forget_gate_U->_update_();
    cell_gate_U->_update_();
    output_gate_U->_update_();

    network->batch_size *= network->time_step;
}

}
// End of namespace nebula.
