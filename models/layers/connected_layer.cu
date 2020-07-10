extern "C++" {
#include "connected_layer.h"
#include "utils.h"
#include "batchnorm.h"
#include "gemm.h"
}

// Forward propagation
extern "C++" void connected_layer_t::_forward_() {
    cudaMemset(output_data_dev, 0.0, output_size*network->batch_size*sizeof(float));
    cudaMemset(delta_dev, 0.0, output_size*network->batch_size*sizeof(float));

    const float alpha = 1.0;
    const float beta  = 1.0;
    float *input_data_dev = prev_layer ? prev_layer->output_data_dev : network->input_data_dev;



    // Matrix multiplication
#ifdef CUSTOM_BLAS
  
    _gemm_(CUBLAS_OP_T, CUBLAS_OP_N,
           output_size, network->batch_size, input_size, 
           alpha, 
           weight_dev, input_size, 
           input_data_dev, input_size, 
           beta, 
           output_data_dev, output_size);
   
#else
    cublasSgemm(network->cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, 
                output_size, network->batch_size, input_size, 
                &alpha, 
                weight_dev, input_size, 
                input_data_dev, input_size, 
                &beta,
                output_data_dev, output_size);
#endif
    // Forward bias
    if(batch_normalize) {
        _forward_batchnorm_();
    }
    _forward_bias_(output_data_dev, bias_dev, 1, output_size, network->batch_size);
    // Activate function
    _activate_();
}

extern "C++" void connected_layer_t::_forward_(float *m_input_data_dev) {
    cudaMemset(output_data_dev, 0.0, output_size*network->batch_size*sizeof(float));
    cudaMemset(delta_dev, 0.0, output_size*network->batch_size*sizeof(float));

    float *input_data_dev = m_input_data_dev ? m_input_data_dev : 
                            prev_layer ? prev_layer->output_data_dev : network->input_data_dev;

    const float alpha = 1.0;
    const float beta  = 1.0;

    // Matrix multiplication
#ifdef CUSTOM_BLAS
  
    _gemm_(CUBLAS_OP_T, CUBLAS_OP_N,
           output_size, network->batch_size, input_size, 
           alpha, 
           weight_dev, input_size, 
           input_data_dev, input_size, 
           beta, 
           output_data_dev, output_size);
   
#else
    cublasSgemm(network->cublas_handle, 
                CUBLAS_OP_T, CUBLAS_OP_N,
                output_size, network->batch_size, input_size, 
                &alpha, 
                weight_dev, input_size, 
                input_data_dev, input_size, 
                &beta,
                output_data_dev, output_size);
#endif
    // Forward bias
    if(batch_normalize) {
        _forward_batchnorm_();
    }
    _forward_bias_(output_data_dev, bias_dev, 1, output_size, network->batch_size);
    // Activate function
    _activate_();

}

// Backward propagation
extern "C++" void connected_layer_t::_backward_() {

    const float alpha = 1.0;
    const float beta  = 1.0;
    float *input_data_dev = prev_layer ? prev_layer->output_data_dev : network->input_data_dev;
    float *prev_delta_dev = prev_layer ? prev_layer->delta_dev : NULL;

    // Gradient function
    _gradient_();
   
    // backward bias.
    _backward_bias_(bias_update_dev, delta_dev, 1, output_size, network->batch_size);
    // normalize.
    if(batch_normalize) {
        _backward_batchnorm_();
    }
    
    // Weight update
#ifdef CUSTOM_BLAS
    _gemm_(CUBLAS_OP_N, CUBLAS_OP_T, 
           input_size, output_size, network->batch_size, 
           alpha, 
           input_data_dev, input_size, 
           delta_dev, output_size, 
           beta, 
           weight_update_dev, input_size);
#else
    cublasSgemm(network->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, input_size, output_size,
                network->batch_size, &alpha, input_data_dev, input_size, delta_dev,
                output_size, &beta, weight_update_dev, input_size);
#endif

    // Delta update
    if(prev_delta_dev) {
#ifdef CUSTOM_BLAS
        _gemm_(CUBLAS_OP_N, CUBLAS_OP_N, input_size, network->batch_size, output_size, 
               alpha, 
               weight_dev, input_size,
               delta_dev, output_size, 
               beta, 
               prev_delta_dev, input_size);
#else
        cublasSgemm(network->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                    input_size, network->batch_size, output_size, 
                    &alpha, 
                    weight_dev, input_size,
                    delta_dev, output_size, 
                    &beta, 
                    prev_delta_dev, input_size);
#endif
    }
}

extern "C++" void connected_layer_t::_backward_(float *m_input_data_dev, float *m_delta_dev) {

    const float alpha = 1.0;
    const float beta  = 1.0;

    float *input_data_dev = m_input_data_dev ? m_input_data_dev :
                            prev_layer ? prev_layer->output_data_dev : network->input_data_dev;

    float *prev_delta_dev = m_delta_dev ? m_delta_dev :
                            prev_layer ? prev_layer->delta_dev : 0;

    // Gradient function
    _gradient_();
   
    // backward bias.
    _backward_bias_(bias_update_dev, delta_dev, 1, output_size, network->batch_size);
    // normalize.
    if(batch_normalize) {
        _backward_batchnorm_();
    }
    
    // Weight update
#ifdef CUSTOM_BLAS
    _gemm_(CUBLAS_OP_N, CUBLAS_OP_T, input_size, output_size, network->batch_size, 
           alpha, 
           input_data_dev, input_size, 
           delta_dev, output_size, 
           beta, 
           weight_update_dev, input_size);
#else
    cublasSgemm(network->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, 
                input_size, output_size, network->batch_size, 
                &alpha, 
                input_data_dev, input_size, 
                delta_dev, output_size, 
                &beta, 
                weight_update_dev, input_size);
#endif

    // Delta update
    if(prev_delta_dev) {
#ifdef CUSTOM_BLAS
        _gemm_(CUBLAS_OP_N, CUBLAS_OP_N, input_size, network->batch_size, output_size, 
               alpha, 
               weight_dev, input_size,
               delta_dev, output_size, 
               beta, 
               prev_delta_dev, input_size);
#else
        cublasSgemm(network->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                    input_size, network->batch_size, output_size, 
                    &alpha, 
                    weight_dev, input_size,
                    delta_dev, output_size, 
                    &beta, 
                    prev_delta_dev, input_size);
#endif
    }
}

extern "C++" void connected_layer_t::_update_() {
    float learning_rate = network->learning_rate/network->batch_size;
    float decay = -network->decay*network->batch_size;
    float momentum = network->momentum;

#ifdef CUSTOM_BLAS
    _axpy_(weight_size, decay, weight_dev, 1, weight_update_dev, 1);
    _axpy_(weight_size, learning_rate, weight_update_dev, 1, weight_dev, 1);
    _scal_(weight_size, momentum, weight_update_dev, 1);

    _axpy_(output_size, learning_rate, bias_update_dev, 1, bias_dev, 1);
    _scal_(output_size, momentum, weight_update_dev, 1);
    if(batch_normalize) {
        _axpy_(output_size, learning_rate, scale_update_dev, 1, scale_dev, 1);
        _scal_(output_size, momentum, scale_update_dev, 1);
    }
#else
    // Weight update
    cublasSaxpy(network->cublas_handle, weight_size, &decay,
                weight_dev, 1, weight_update_dev, 1);
    cublasSaxpy(network->cublas_handle, weight_size, &learning_rate, 
                weight_update_dev, 1, weight_dev, 1);
    cublasSscal(network->cublas_handle, weight_size, &momentum, weight_update_dev, 1);

    // Bias update
    cublasSaxpy(network->cublas_handle, output_size, &learning_rate, 
                bias_update_dev, 1, bias_dev, 1);
    cublasSscal(network->cublas_handle, output_size, &momentum, bias_update_dev, 1);
    if(batch_normalize) {
        cublasSaxpy(network->cublas_handle, output_size, &learning_rate,
                     scale_update_dev, 1, scale_dev, 1);
        cublasSscal(network->cublas_handle, output_size, &momentum,
                    scale_update_dev, 1);
    }
#endif

}
extern "C++" void connected_layer_t::_forward_batchnorm_() {

    cudaMemcpy(x_dev, output_data_dev, 
               output_size * network->batch_size * sizeof(float), cudaMemcpyDeviceToDevice);

    if(network->run_type == TRAIN_RUN) {
        // Mean
        _batchnorm_mean_(output_data_dev, normalize_mean_dev, 
                         output_size, 1, network->batch_size);

        // Variance
        _batchnorm_variance_(output_data_dev, normalize_mean_dev, normalize_variance_dev, 
                             output_size, 1, network->batch_size);

        float rolling = 0.99;
        float normalize = 0.01;
#ifdef CUSTOM_BLAS
        _scal_(output_channel, rolling, 
               rolling_mean_dev, 1);
        _axpy_(output_channel, normalize, 
               normalize_mean_dev, 1, rolling_mean_dev, 1);

        _scal_(output_channel, rolling, 
               rolling_variance_dev, 1);
        _axpy_(output_channel, normalize, 
               normalize_variance_dev, 1, rolling_variance_dev, 1);
#else
        cublasSscal(network->cublas_handle, output_channel, 
                    &rolling, rolling_mean_dev, 1);
        cublasSaxpy(network->cublas_handle, output_channel, 
                    &normalize, normalize_mean_dev, 1, rolling_mean_dev, 1);

        cublasSscal(network->cublas_handle, output_channel, 
                    &rolling, rolling_variance_dev, 1);
        cublasSaxpy(network->cublas_handle, output_channel, 
                    &normalize, normalize_variance_dev, 1, rolling_variance_dev, 1);
#endif

        _batchnorm_normalize_(output_data_dev, normalize_mean_dev, normalize_variance_dev, 
                              output_size, 1, network->batch_size);
        cudaMemcpy(normalize_x_dev, output_data_dev, 
                   output_size * network->batch_size * sizeof(float), cudaMemcpyDeviceToDevice);
    }
    else {
        _batchnorm_normalize_(output_data_dev, rolling_mean_dev, rolling_variance_dev, 
                              output_size, 1, network->batch_size);
    }
    // Scale down the output data.
    _batchnorm_scale_down_(output_data_dev, scale_dev, 
                           output_size, 1, network->batch_size);
}

extern "C++" void connected_layer_t::_backward_batchnorm_() {
   
    _batchnorm_scale_down_(delta_dev, scale_dev, 
                           output_size, 1, network->batch_size);

    // Calculate delta value of normalize mean.
    _batchnorm_mean_delta_(delta_dev, normalize_variance_dev, mean_delta_dev, 
                           output_size, 1, network->batch_size); 
    // Calculate delta value of normalize variance.
    _batchnorm_variance_delta_(x_dev, delta_dev, normalize_mean_dev, normalize_variance_dev, variance_delta_dev, 
                               output_size, 1, network->batch_size);
    // Normalize.
    _batchnorm_normalize_delta_(x_dev, normalize_mean_dev, normalize_variance_dev, 
                                mean_delta_dev, variance_delta_dev, delta_dev, 
                                output_size, 1, network->batch_size); 
}

// Go to next time step.
extern "C++" void connected_layer_t::_increment_(int step) {
    int num = (int)output_size * (int)network->batch_size * step;
    output_data_dev += num;
    delta_dev += num;
  
    if(batch_normalize) {
        x_dev += num;
        normalize_x_dev += num;
    }
}

