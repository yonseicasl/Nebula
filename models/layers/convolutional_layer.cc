#include <algorithm>
#ifndef CUSTOM_BLAS
    #include <cblas.h>
#endif
#include <fstream>
#include <functional>
#include <cstring>
#include <random>
#include <thread>
#include "convolutional_layer.h"
#include "utils.h"
#include "batchnorm.h"
#include "gemm.h"

namespace nebula { 

convolutional_layer_t::convolutional_layer_t(network_t *m_network, layer_t *m_prev_layer, layer_type_t m_layer_type) :
    layer_t(m_network, m_prev_layer, m_layer_type),
    workspace(NULL),
    workspace_size(0),
    bias(NULL),
    bias_update(NULL),
    weight_update(NULL),
    batch_normalize(false),
    scale(NULL),
    scale_update(NULL),
    normalize_mean(NULL),
    rolling_mean(NULL),
    mean_delta(NULL),
    normalize_variance(NULL),
    rolling_variance(NULL),
    variance_delta(NULL),
    x(NULL),
    normalize_x(NULL) {    
}

convolutional_layer_t::~convolutional_layer_t() {
    delete [] workspace;
    delete [] bias;
    delete [] bias_update;
    delete [] weight;
    delete [] weight_update;
    delete [] output_data;
    //delete [] input_data;
    delete [] delta;
    if(batch_normalize) {
        delete [] scale;
        delete [] scale_update;
        delete [] normalize_mean;
        delete [] rolling_mean;
        delete [] mean_delta;
        delete [] normalize_variance;
        delete [] rolling_variance;
        delete [] variance_delta;
        delete [] x;
        delete [] normalize_x;
    }
}

void convolutional_layer_t::init(section_config_t m_section_config) {
    // Get layer settings.
    m_section_config.get_setting("filters", &num_filters);
    m_section_config.get_setting("size", &filter_size);
    m_section_config.get_setting("batch_normalize", &batch_normalize);  
    m_section_config.get_setting("padding", &padding);
    m_section_config.get_setting("stride", &stride);

    std::string activation_str;
    if(m_section_config.get_setting("activation", &activation_str)) {
        activation_type = (activation_type_t)get_type(activation_type_str, activation_str);
    }

    input_size = prev_layer ? prev_layer->output_size : network->input_size;

    input_height = prev_layer ? prev_layer->output_height : network->input_height;
    input_width = prev_layer ? prev_layer->output_width : network->input_width;
    input_channel = prev_layer ? prev_layer->output_channel : network->input_channel;
    
    output_height = (input_height + 2 * padding - filter_size) / stride + 1;
    output_width = (input_width  + 2 * padding - filter_size) / stride + 1;
    output_channel = num_filters;
    output_size = output_height * output_width * output_channel;
    
    workspace_size = output_height * output_width * filter_size * filter_size * input_channel;
    weight_size = input_channel * num_filters * filter_size * filter_size / group;	   
    
    bias = new float[num_filters]();
    bias_update = new float[num_filters]();

    weight = new float[weight_size]();
    weight_update = new float[weight_size]();

    //input_data = new float[input_size * network->batch_size]();
    output_data = new float[output_size * network->batch_size]();
    delta = new float[output_size * network->batch_size]();
    workspace = new float[workspace_size]();

    // Print out structure of the network.
    //std::cout << input_height << " " << input_width << " " << input_channel << " " << filter_size << " " << filter_size << " " << num_filters << std::endl;

    // Initialize parameters for batch normalization.
    if(batch_normalize) {
        scale        = new float[num_filters]();
        scale_update = new float[num_filters]();
        for(unsigned i = 0; i < num_filters; i++) {
            scale[i] = 1.0;
        }

        normalize_mean = new float[num_filters]();
        rolling_mean = new float[num_filters]();
        mean_delta = new float[num_filters]();

        normalize_variance = new float[num_filters]();
        rolling_variance = new float[num_filters]();
        variance_delta = new float[num_filters]();

        x = new float[output_size * network->batch_size]();
        normalize_x = new float[output_size * network->batch_size]();
    }
}

void convolutional_layer_t::init_weight(std::fstream &m_input_weight) {
    m_input_weight.read((char*)bias, num_filters * sizeof(float));
    m_input_weight.read((char*)weight, weight_size * sizeof(float));
   
    if(batch_normalize) {
        m_input_weight.read((char*)scale, num_filters * sizeof(float));
        m_input_weight.read((char*)rolling_mean, num_filters * sizeof(float));
        m_input_weight.read((char*)rolling_variance, num_filters * sizeof(float));
    }
}

void convolutional_layer_t::init_weight() {
    std::minstd_rand rng(std::random_device{}());
    std::normal_distribution<float> dist(0.0, 1.0);
    
    for(unsigned i = 0; i < weight_size; i++) {
        weight[i] = sqrt(2.0 / (filter_size * filter_size * input_channel / group)) * dist(rng);
    }
}

void convolutional_layer_t::store_weight(std::fstream &m_output_weight) {
    m_output_weight.write((char*)bias, num_filters * sizeof(float));
    m_output_weight.write((char*)weight, weight_size * sizeof(float));
    if(batch_normalize) {
        m_output_weight.write((char*)scale, num_filters * sizeof(float));
        m_output_weight.write((char*)rolling_mean, num_filters * sizeof(float)); 
        m_output_weight.write((char*)rolling_variance, num_filters * sizeof(float)); 
    }
}

void convolutional_layer_t::forward() {
    memset(output_data, 0, output_size * network->batch_size * sizeof(float));
    memset(delta, 0, output_size * network->batch_size * sizeof(float));
    
    input_data = prev_layer ? prev_layer->output_data : network->input_data;
    unsigned patch_size = filter_size * filter_size * input_channel/ group;
    unsigned num_patches = output_width * output_height;

	// Convolution
	for(unsigned i = 0; i < network->batch_size; i++){
		for(unsigned j =0 ; j < group ; j++){
			im2col(&input_data[(i * group + j) * input_channel / group * input_height * input_width], input_channel / group,
					input_height, input_width, filter_size, stride, padding, workspace,
					network->num_threads);
#ifdef CUSTOM_BLAS
			gemm(0, 0,
					num_filters / group, num_patches, patch_size,
					1.0,
					&weight[j * weight_size / group], patch_size,
					workspace, num_patches, 
					1.0,
					&output_data[(i * group +j) * num_patches * num_filters / group], num_patches,
					network->num_threads);
#else
			cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
					num_filters / group, num_patches, patch_size, 
					1.0, 
					&weight[j * weight_size / group], patch_size, 
					workspace, num_patches, 
					1.0,
					&output_data[(i * group + j) * num_patches * num_filters / group], num_patches);
#endif

		}
	}

    // Forward bias
    if(batch_normalize) {
        forward_batchnorm();
    } 
    forward_bias(num_threads, output_data, bias, num_filters, num_patches, network->batch_size);

    // Activate function
    activate();
}

void convolutional_layer_t::backward() {
    // Gradient function
    gradient();

    unsigned patch_size = filter_size*filter_size*input_channel
						 / group;
    unsigned num_patches = output_width*output_height;

    // Backward bias
    backward_bias(num_threads, bias_update, delta, num_filters, num_patches, network->batch_size);
    if(batch_normalize){
        backward_batchnorm();
    }

    input_data = prev_layer ? prev_layer->output_data : network->input_data;
    float *prev_delta = prev_layer ? prev_layer->delta : NULL;

    for(unsigned i = 0; i < network->batch_size; ++i) {
		for(unsigned j = 0; j < group; ++j){
			// Matrix multiplication for weight update
			im2col(&input_data[(i * group +j) * input_channel / group * input_height * input_width], input_channel,
					input_height, input_width, filter_size, stride, padding, workspace,
					network->num_threads);
#ifdef CUSTOM_BLAS
			gemm(0, 1, 
					num_filters / group, patch_size, num_patches,
					1.0,
					&delta[(i * group +j) * num_filters / group * num_patches], num_patches,
					workspace, num_patches,
					1.0,
					&weight_update[j* weight_size / group], patch_size,
					network->num_threads);
#else
			cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, num_filters / group, patch_size, num_patches, 
					1.0, 
					&delta[(i * group +j) * num_filters / group* num_patches], num_patches, 
					workspace, num_patches, 
					1.0, 
					&weight_update[j * weight_size / group], patch_size);
#endif

			// Matrix multiplication for delta update
			if (prev_delta) { 
#ifdef CUSTOM_BLAS
				gemm(1, 0, 
						patch_size, num_patches, num_filters / group,
						1.0,
						&weight[j * weight_size / group], patch_size,
						&delta[(i * group +j) * num_filters / group* num_patches], num_patches,
						0.0,
						workspace, num_patches,
						network->num_threads);
#else
				cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 
						patch_size, num_patches, num_filters / group, 
						1.0, 
						&weight[j * weight_size /group], patch_size,
						&delta[(i * group + j) * num_filters / group * num_patches], num_patches, 
						0.0,
						workspace, num_patches);
#endif

				col2im(workspace, input_channel / group, input_height, input_width, filter_size, stride,
						padding, &prev_delta[(i * group +j) * input_channel / group * input_height * input_width],
						network->num_threads);
			}
		}
	}
}
void convolutional_layer_t::update() {
#ifdef CUSTOM_BLAS
    axpy(num_filters, network->learning_rate / network->batch_size, bias_update, 1, bias, 1);
    scal(num_filters, network->momentum, bias_update, 1);

    axpy(weight_size, -network->decay * network->batch_size, weight, 1, weight_update, 1);
    axpy(weight_size, network->learning_rate / network->batch_size, weight_update, 1, weight, 1);
    scal(weight_size, network->momentum, weight_update, 1);
#else

    // Update bias.
    cblas_saxpy(num_filters, network->learning_rate / network->batch_size, bias_update, 1, bias, 1);
    cblas_sscal(num_filters, network->momentum, bias_update, 1);

    // Update weight.
    cblas_saxpy(weight_size, (0.0 - network->decay) * network->batch_size, weight, 1, weight_update, 1);
    cblas_saxpy(weight_size, network->learning_rate / network->batch_size, weight_update, 1, weight, 1);
    cblas_sscal(weight_size, network->momentum, weight_update, 1);
#endif
}

// Forward batch normalization.
void convolutional_layer_t::forward_batchnorm() {
    unsigned num_patches = output_height * output_width;
    memcpy(x, output_data, output_size * network->batch_size * sizeof(float));
    if(network->run_type == TRAIN_RUN) {
    
        batchnorm_mean(num_threads, output_data, normalize_mean,
                       output_channel, num_patches, network->batch_size);
        batchnorm_variance(num_threads, output_data, normalize_mean, normalize_variance, 
                           output_channel, num_patches, network->batch_size);

#ifdef CUSTOM_BLAS
        scal(output_channel, 0.99, rolling_mean, 1);
        axpy(output_channel, 0.01, normalize_mean, 1, rolling_mean, 1);
        scal(output_channel, 0.99, rolling_variance, 1);
        axpy(output_channel, 0.01, normalize_variance, 1, rolling_variance, 1);
#else
        cblas_sscal(output_channel, 0.99, rolling_mean, 1);
        cblas_saxpy(output_channel, 0.01, normalize_mean, 1, rolling_mean, 1);
        cblas_sscal(output_channel, 0.99, rolling_variance, 1);
        cblas_saxpy(output_channel, 0.01, normalize_variance, 1, rolling_variance, 1);
#endif

        // Normalize all output data.
        batchnorm_normalize(num_threads, output_data, normalize_mean, normalize_variance, 
                            output_channel, num_patches, network->batch_size);
        memcpy(normalize_x, output_data, output_size*network->batch_size*sizeof(float));
    }
    else {
        // normalize all output data.
        batchnorm_normalize(num_threads, output_data, rolling_mean, rolling_variance, 
                            output_channel, num_patches, network->batch_size);
    }
    batchnorm_scale_down(num_threads, output_data, scale, 
                         output_channel, num_patches, network->batch_size);
}

//Backward batch normalization.
void convolutional_layer_t::backward_batchnorm() {
    unsigned num_patches = output_height * output_width;

    batchnorm_scale_down(num_threads, delta, scale, 
               output_channel, num_patches, network->batch_size);

    batchnorm_mean_delta(num_threads, delta, normalize_variance, mean_delta, 
                         output_channel, num_patches, network->batch_size);

    batchnorm_variance_delta(num_threads, x, delta, 
                             normalize_mean, normalize_variance, variance_delta, 
                             output_channel, num_patches, network->batch_size);

    batchnorm_normalize_delta(num_threads, x, normalize_mean, normalize_variance, 
                    mean_delta, variance_delta, delta, 
                    output_channel, num_patches, network->batch_size); 
}

}
//End of namespace nebula.
