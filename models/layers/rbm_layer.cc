#include <algorithm>
#ifndef CUSTOM_BLAS
    #include <cblas.h>
#endif
#include <fstream>
#include <functional>
#include <cstring>
#include <random>
#include <thread>
#ifdef GPU_ENABLED
#include <cuda_runtime.h>
#endif
#include "rbm_layer.h"
#include "gemm.h"
#include "activations.h"

rbm_layer_t::rbm_layer_t(network_t *m_network, layer_t *m_prev_layer, layer_type_t m_layer_type) :
    layer_t(m_network, m_prev_layer, m_layer_type),
    weight(NULL),
    weight_update(NULL),
    weight_size(0),
    k_step(1) {
#ifdef GPU_ENABLED
    weight_dev = NULL;
    weight_update_dev = NULL;
#endif
}

rbm_layer_t::~rbm_layer_t() {
    delete [] output_data;
    delete [] delta;
    delete [] weight;
    delete [] weight_update;
    
    delete [] hidden_units;
    delete [] hidden_mean_zero_step;
    delete [] hidden_mean_k_step;
    delete [] visible_units_zero_step;
    delete [] visible_units_k_step;
    delete [] visible_mean;  
    
    delete [] visible_bias;
    delete [] hidden_bias;
    delete [] visible_bias_update;
    delete [] hidden_bias_update;
#ifdef GPU_ENABLED
    cudaFree(output_data_dev);
    cudaFree(delta_dev);
    cudaFree(weight_dev);
    cudaFree(weight_update_dev);
    
    cudaFree(hidden_units_dev);
    cudaFree(hidden_mean_zero_step_dev);
    cudaFree(hidden_mean_k_step_dev);
    cudaFree(visible_units_zero_step_dev);
    cudaFree(visible_units_k_step_dev);
    cudaFree(visible_mean_dev);     

    cudaFree(visible_bias_dev);
    cudaFree(hidden_bias_dev);
    cudaFree(visible_bias_update_dev);
    cudaFree(hidden_bias_update_dev);
#endif
}

void rbm_layer_t::init(section_config_t m_section_config) {
    // Get layer settings.
    m_section_config.get_setting("output", &output_size);
    m_section_config.get_setting("k_step", &k_step);
    
    std::string activation_str;
    if(m_section_config.get_setting("activation", &activation_str)) {
        activation_type = (activation_type_t)get_type(activation_type_str, activation_str);
    }

    // Initialize layer parameters.
    input_size = prev_layer ? prev_layer->output_size :
                 network->input_height * network->input_width * network->input_channel;
    
    input_height = 1;
    input_width = 1;
    input_channel = input_size;

    output_height = 1;
    output_width = 1;
    output_channel = output_size;
    
    weight_size = input_size * output_size;

    output_data = new float[output_size * network->batch_size]();
    delta = new float[input_size * output_size]();
    
    weight = new float[input_size * output_size]();
    weight_update = new float[input_size * output_size]();
    
    hidden_units           = new float[network->batch_size * output_size]();
    hidden_mean_zero_step  = new float[network->batch_size * output_size]();
    hidden_mean_k_step     = new float[network->batch_size * output_size]();
   
    visible_units_zero_step = new float[network->batch_size * input_size]();
    visible_units_k_step    = new float[network->batch_size * input_size]();
    visible_mean            = new float[network->batch_size * input_size]();

    visible_bias = new float[input_size]();
    hidden_bias = new float[output_size]();
    visible_bias_update = new float[input_size]();
    hidden_bias_update = new float[output_size]();

#ifdef GPU_ENABLED
    cudaMalloc((void**)&output_data_dev, output_size * network->batch_size * sizeof(float));
    cudaMalloc((void**)&delta_dev, input_size * output_size * sizeof(float));
    cudaMemset(output_data_dev, 0.0, output_size * network->batch_size * sizeof(float));
    cudaMemset(delta_dev, 0.0, input_size * output_size * sizeof(float));
    
    cudaMalloc((void**)&weight_dev, input_size * output_size * sizeof(float));
    cudaMalloc((void**)&weight_update_dev, input_size * output_size * sizeof(float));
    cudaMemset(weight_dev, 0.0, input_size * output_size * sizeof(float));
    cudaMemset(weight_update_dev, 0.0, input_size * output_size * sizeof(float));
    
    cudaMalloc((void**)&hidden_units_dev, output_size * network->batch_size *sizeof(float));
    cudaMalloc((void**)&hidden_mean_zero_step_dev, output_size * network->batch_size *sizeof(float));
    cudaMalloc((void**)&hidden_mean_k_step_dev, output_size * network->batch_size *sizeof(float));
    cudaMemset(hidden_units_dev, 0.0, output_size * network->batch_size * sizeof(float));
    cudaMemset(hidden_mean_zero_step_dev, 0.0, output_size * network->batch_size * sizeof(float));
    cudaMemset(hidden_mean_k_step_dev, 0.0, output_size * network->batch_size * sizeof(float));
    
    cudaMalloc((void**)&visible_units_zero_step_dev, input_size* network->batch_size *sizeof(float));
    cudaMalloc((void**)&visible_units_k_step_dev, input_size* network->batch_size *sizeof(float));
    cudaMalloc((void**)&visible_mean_dev, input_size* network->batch_size *sizeof(float));
    cudaMemset(visible_units_zero_step_dev, 0.0, input_size * network->batch_size * sizeof(float));
    cudaMemset(visible_units_k_step_dev, 0.0, input_size * network->batch_size * sizeof(float));
    cudaMemset(visible_mean_dev, 0.0, input_size * network->batch_size * sizeof(float));
   
    cudaMalloc((void**)&visible_bias_dev, input_size * sizeof(float));
    cudaMalloc((void**)&hidden_bias_dev, output_size * sizeof(float));
    cudaMalloc((void**)&visible_bias_update_dev, input_size * sizeof(float));
    cudaMalloc((void**)&hidden_bias_update_dev, output_size * sizeof(float));
   
    cudaMemset(visible_bias_dev, 0.0, input_size * sizeof(float));
    cudaMemset(hidden_bias_dev, 0.0, output_size * sizeof(float));
    cudaMemset(visible_bias_update_dev, 0.0, input_size * sizeof(float));
    cudaMemset(hidden_bias_update_dev, 0.0, output_size * sizeof(float));
    
#endif
}

// Initialize weight from weight file.
void rbm_layer_t::init_weight(std::fstream &m_input_weight) {
    m_input_weight.read((char*)hidden_bias, output_size * sizeof(float));
    m_input_weight.read((char*)weight, weight_size * sizeof(float));
#ifdef GPU_ENABLED
    cudaMemcpy(hidden_bias_dev, hidden_bias, output_size * sizeof(float), cudaMemcpyHostToDevice); 
    cudaMemcpy(weight_dev, weight, weight_size * sizeof(float), cudaMemcpyHostToDevice);
#endif
}

// Initialize weight from scratch.
void rbm_layer_t::init_weight() {
    std::minstd_rand rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(-1.0, 1.0);

    // Initialize weight 
    for(unsigned i = 0; i < weight_size; i++) {
        weight[i] = sqrt(2.0 / input_size) * dist(rng);
    }
#ifdef GPU_ENABLED 
    cudaMemcpy(weight_dev, weight, weight_size * sizeof(float), cudaMemcpyHostToDevice);    
#endif
}

// Save weight to the weight file.
void rbm_layer_t::store_weight(std::fstream &m_output_weight) {
#ifdef GPU_ENABLED
    cudaMemcpy(hidden_bias, hidden_bias_dev, output_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(weight, weight_dev, weight_size * sizeof(float), cudaMemcpyDeviceToHost);
#endif
    m_output_weight.write((char*)hidden_bias, output_size * sizeof(float));
    m_output_weight.write((char*)weight, weight_size * sizeof(float));
}

// Sampling using the conditional probability
void sampling(float *sample, float *probability, unsigned size) {
    
    srand(static_cast <unsigned> (time(0)));

    for(unsigned i = 0; i < size; i++) {
        if(probability[i] < 0.0 || probability[i] > 1.0) {
            return;
        }

        // choose random floats in the half-open interval [0.0,1.0)
        float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX+1.0);
        if ( r < probability[i]) sample[i] = 1.0;
        else sample[i] = 0.0;
    }
}

// Sample hidden units using visible units value
void rbm_layer_t::sample_hidden_units(unsigned m_step) {
    memset(hidden_units, 0.0, output_size * network->batch_size * sizeof(float));
    
    float *t_visible_units;
    float *t_hidden_mean;

    if(m_step==0) {
        t_visible_units = visible_units_zero_step;
        t_hidden_mean   = hidden_mean_zero_step;
    }
    else {
        t_visible_units = visible_units_k_step;
        t_hidden_mean   = hidden_mean_k_step; 
    }
    
#ifdef CUSTOM_BLAS  
    gemm(0, 1,
         network->batch_size, output_size, input_size,
         1.0,
         t_visible_units, input_size,
         weight, input_size, 
         1.0,
         t_hidden_mean, output_size,
         num_threads);
#else
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 
                network->batch_size, output_size, input_size, 
                1.0, 
                t_visible_units, input_size, 
                weight, input_size, 
                1.0, 
                t_hidden_mean, output_size);
#endif
    
    forward_bias(num_threads, t_hidden_mean, hidden_bias, output_size, 1, network->batch_size); 
    logistic_activation(t_hidden_mean, network->batch_size * output_size);
    sampling(hidden_units, t_hidden_mean, network->batch_size * output_size);
}

// Sample visible units using hidden units value
void rbm_layer_t::sample_visible_units() {
    memset(visible_units_k_step, 0.0, input_size * network->batch_size * sizeof(float));
    
#ifdef CUSTOM_BLAS
    gemm(0, 0, 
         network->batch_size, input_size, output_size, 
         1.0, 
         hidden_units, output_size,
         weight, input_size,
         1.0,
         visible_mean, input_size,
         num_threads);
#else
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                network->batch_size, input_size, output_size, 
                1.0, 
                hidden_units, output_size, 
                weight, input_size, 
                1.0, 
                visible_mean, input_size);
#endif
    
    forward_bias(num_threads, visible_mean, visible_bias, input_size, 1, network->batch_size);
    logistic_activation(visible_mean, network->batch_size * input_size);
    sampling(visible_units_k_step, visible_mean, network->batch_size * input_size);
}

// Reconstruct the visible units and pretrain weight.
void rbm_layer_t::pretrain() {
    memset(output_data, 0, output_size * network->batch_size * sizeof(float));
    memset(delta , 0, output_size * network->batch_size * sizeof(float));

    float *input_data = prev_layer ? prev_layer->output_data : network->input_data;
    
    memcpy(visible_units_zero_step, input_data, input_size * network->batch_size * sizeof(float));

    // K-step contrastive divergence_gradient approximation for weight update and bias update
    for(unsigned t = 0; t < k_step; t++)
    {
        if(!t)
        {
            sample_hidden_units(t);
            sample_visible_units();
        }
        else
        {
            sample_hidden_units(t);
            sample_visible_units();
        }
        sample_hidden_units(1);
    }
    
#ifdef CUSTOM_BLAS
    gemm(1, 0, 
         output_size, input_size, network->batch_size, 
         1.0,
         hidden_mean_zero_step, output_size,
         visible_units_zero_step, input_size, 
         1.0,
         weight_update, input_size,
         num_threads);
    gemm(1, 0,
         output_size, input_size, network->batch_size, 
         -1.0,
         hidden_mean_k_step, output_size,
         visible_units_k_step, input_size, 
         1.0,
         weight_update, input_size,
         num_threads);
#else
    // Matrix multiplication for weight update.
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 
                output_size, input_size, network->batch_size, 
                1.0, 
                hidden_mean_zero_step, output_size, 
                visible_units_zero_step, input_size, 
                1.0, 
                weight_update, input_size);
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 
                output_size, input_size, network->batch_size, 
                -1.0, 
                hidden_mean_k_step, output_size, 
                visible_units_k_step, input_size, 
                1.0, 
                weight_update, input_size);
#endif
     
    // Calculate bias update 
    for(unsigned i = 0; i < network->batch_size; i++)
    {
        for(unsigned j = 0; j < output_size; j++)
        {
			hidden_bias_update[j] += hidden_mean_zero_step[i*output_size +j] - hidden_mean_k_step[i*output_size +j];
        }
        for(unsigned k = 0; k < input_size; k++)
        {
			visible_bias_update[k] += visible_units_zero_step[i*input_size +k] - visible_units_k_step[i*input_size +k];
        }
    }
    
    // Update bias of visible units.
#ifdef CUSTOM_BLAS
    axpy(input_size, network->learning_rate / network->batch_size, visible_bias_update, 1, visible_bias, 1);
    scal(input_size, network->momentum, visible_bias_update, 1);

    axpy(output_size, network->learning_rate / network->batch_size, hidden_bias_update, 1, hidden_bias, 1);
    scal(output_size, network->momentum, hidden_bias_update, 1);

    axpy(weight_size, (0.0 - network->decay) * network->batch_size, weight, 1, weight_update, 1);
    axpy(weight_size, network->learning_rate / network->batch_size, weight_update, 1, weight, 1);
    scal(weight_size, network->momentum, weight_update, 1);

#else
    cblas_saxpy(input_size, network->learning_rate / network->batch_size, visible_bias_update, 1, visible_bias, 1);
    cblas_sscal(input_size, network->momentum, visible_bias_update, 1);
    
    // Update bias of hidden units.
    cblas_saxpy(output_size, network->learning_rate / network->batch_size, hidden_bias_update, 1, hidden_bias, 1);
    cblas_sscal(output_size, network->momentum, hidden_bias_update, 1);
    
    // Update weight.
    cblas_saxpy(weight_size, (0.0 - network->decay) * network->batch_size, weight, 1, weight_update, 1);
    cblas_saxpy(weight_size, network->learning_rate / network->batch_size, weight_update, 1, weight, 1);
    cblas_sscal(weight_size, network->momentum, weight_update, 1);
#endif
   
    memcpy(output_data, hidden_mean_k_step, output_size * network->batch_size);
}

void rbm_layer_t::forward() {
    memset(output_data, 0, output_size * network->batch_size * sizeof(float));
    memset(delta , 0, output_size * network->batch_size * sizeof(float));
    float *input_data = prev_layer ? prev_layer->output_data : network->input_data;
   
    // Matrix multiplication
#ifdef CUSTOM_BLAS
    gemm(0, 1, 
         network->batch_size, output_size, input_size,
         1.0,
         input_data, input_size, 
         weight, input_size, 
         1.0,
         output_data, output_size,
         num_threads);
#else
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 
                network->batch_size, output_size, input_size, 
                1.0, 
                input_data, input_size, 
                weight, input_size, 
                1.0, 
                output_data, output_size);
#endif
    // Add bias.
    forward_bias(num_threads, output_data, hidden_bias, output_size, 1, network->batch_size);
   
    // Activate function 
    activate();
}

void rbm_layer_t::backward() {
    // Gradient function
    gradient();
    
    backward_bias(num_threads, hidden_bias_update, delta, 1, output_width * output_height, network->batch_size);
    
    float *input_data = prev_layer ? prev_layer->output_data : network->input_data;
    float *prev_delta = prev_layer ? prev_layer->delta : NULL;

    // Matrix multiplication for weight update.
#ifdef CUSTOM_BLAS
    gemm(1, 0, 
         output_size, input_size, network->batch_size,
         1.0, 
         delta, output_size,
         input_data, input_size, 
         1.0,
         weight_update, input_size,
         num_threads);
#else
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 
                output_size, input_size, network->batch_size, 
                1.0, 
                delta, output_size, 
                input_data, input_size, 
                1.0, 
                weight_update, input_size);
#endif

    // Matrix multiplication for delta update.
    if(prev_delta) {
#ifdef CUSTOM_BLAS
        gemm(0, 0,
             network->batch_size, input_size, output_size,
             1.0,
             delta, output_size, 
             weight, input_size, 
             1.0,
             prev_delta, input_size,
             num_threads);
#else
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                    network->batch_size, input_size, output_size, 
                    1.0, 
                    delta, output_size, 
                    weight, input_size, 
                    1.0, 
                    prev_delta, input_size);
#endif
    }
}

void rbm_layer_t::update() {
#ifdef CUSTOM_BLAS
    axpy(output_size, network->learning_rate / network->batch_size, hidden_bias_update, 1, hidden_bias, 1);
    scal(output_size, network->momentum, hidden_bias_update, 1);
    
    axpy(weight_size, (0.0 - network->decay) * network->batch_size, weight, 1, weight_update, 1);
    axpy(weight_size, network->learning_rate / network->batch_size, weight_update, 1, weight, 1);
    scal(weight_size, network->momentum, weight_update, 1);

#else
    // Update bias.
    cblas_saxpy(output_size, network->learning_rate / network->batch_size, hidden_bias_update, 1, hidden_bias, 1);
    cblas_sscal(output_size, network->momentum, hidden_bias_update, 1);

    // Update weight.
    cblas_saxpy(weight_size, (0.0 - network->decay) * network->batch_size, weight, 1, weight_update, 1);
    cblas_saxpy(weight_size, network->learning_rate / network->batch_size, weight_update, 1, weight, 1);
    cblas_sscal(weight_size, network->momentum, weight_update, 1);
#endif
}
