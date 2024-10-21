extern "C++" {
#include "convolutional_layer.h"
#include "utils.h"
#include "batchnorm.h"
#include "gemm.h"
}

namespace nebula {

// Forward propagation
extern "C++" void convolutional_layer_t::_forward_() {
    cudaMemset(output_data_dev, 0, output_size*network->batch_size*sizeof(float));
    cudaMemset(delta_dev, 0, output_size*network->batch_size*sizeof(float));
    
    const float alpha = 1.0;
    unsigned num_patches = output_width * output_height;
    float *input_data_dev = prev_layer ? prev_layer->output_data_dev : network->input_data_dev;

#ifdef CUDNN_ENABLED
    size_t workspace_size = 0; 
    cudnnSetConvolutionMathType(convolution_descriptor, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION);
    cudnnGetConvolutionForwardWorkspaceSize(network->cudnn_handle, 
                                            input_descriptor,
                                            weight_descriptor, 
                                            convolution_descriptor,
                                            output_descriptor, 
                                            forward_algorithm,
                                            &workspace_size);


    for(unsigned i = 0; i < group; i++) {
        cudnnConvolutionForward(network->cudnn_handle, 
                                &alpha,
                                input_descriptor, 
                                &input_data_dev[(input_size * network->batch_size * i / group)],
                                weight_descriptor, 
                                &weight_dev[(weight_size * i / group)],
                                convolution_descriptor, 
                                forward_algorithm,
                                workspace_dev, 
                                workspace_size, 
                                &alpha,
                                output_descriptor, 
                                &output_data_dev[(output_size * network->batch_size * i / group)]);
    }
#else // Using cuBLAS.
    // Convolution
    const float beta  = 1.0;
    unsigned patch_size = filter_size * filter_size * input_channel / group;
	for(unsigned i = 0; i < network->batch_size; i++){
		for(unsigned j=0; j<group; j++){
			_im2col_(&input_data_dev[(i * group + j) * input_channel / group * input_height * input_width],
					input_channel /group, input_height, input_width, filter_size, stride, padding,
					workspace_dev);
#ifdef CUSTOM_BLAS 
			_gemm_(CUBLAS_OP_N, CUBLAS_OP_N, num_patches, num_filters / group, patch_size, 
					alpha, 
					workspace_dev, num_patches, 
					&weight_dev[j * weight_size /group], patch_size,
					beta, 
					&output_data_dev[(i * group + j) * num_patches * num_filters], num_patches); 
#else
			cublasSgemm(network->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, 
					num_patches, num_filters / group, patch_size, 
					&alpha, 
					workspace_dev, num_patches, 
					&weight_dev[j * weight_size /group], patch_size,
					&beta, 
					&output_data_dev[(i * group + j) * num_patches * num_filters/ group], num_patches); 
#endif
		}
	}

#endif
    // Forward bias
    if(batch_normalize) {
        _forward_batchnorm_();
    } 
    _forward_bias_(output_data_dev, bias_dev, num_filters, num_patches, network->batch_size);

    // Activate function
    _activate_();
    
}

// Backward propagation
extern "C++" void convolutional_layer_t::_backward_() {
    
    float alpha = 1.0;
    float beta1 = 1.0;
    float beta2 = 0.0;
    unsigned patch_size = filter_size*filter_size*input_channel / group;
    unsigned num_patches = output_width*output_height;
    float *input_data_dev = prev_layer ? prev_layer->output_data_dev : network->input_data_dev;
    float *prev_delta = prev_layer ? prev_layer->delta_dev : NULL;

    // Gradient function
    _gradient_();

    // Backward bias 
    _backward_bias_(bias_update_dev, delta_dev, num_filters, num_patches, network->batch_size);
    if(batch_normalize) {
        _backward_batchnorm_();
    }
     
	for(unsigned i = 0; i < network->batch_size; i++) { 
		for(unsigned j=0; j<group ; ++j){
			// Weight update 
			_im2col_(&input_data_dev[(i*group +j)* input_channel /group * input_height * input_width],
					input_channel/group, input_height, input_width, filter_size, stride, padding,
					workspace_dev);
#ifdef CUSTOM_BLAS
			_gemm_(CUBLAS_OP_T, CUBLAS_OP_N, patch_size, num_filters/group, num_patches, 
					alpha, 
					workspace_dev, num_patches,
					&delta_dev[(i*group+j) * num_filters /group* num_patches], num_patches, 
					beta1,
					&weight_update_dev[j*weight_size /group], patch_size);

#else
			cublasSgemm(network->cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, patch_size, num_filters/group,
					num_patches, &alpha, workspace_dev, num_patches,
					&delta_dev[(i*group+j) * num_filters /group* num_patches], num_patches, &beta1,
					&weight_update_dev[j*weight_size/group], patch_size);
#endif

			// Delta update
			if (prev_delta) {
#ifdef CUSTOM_BLAS

				_gemm_(CUBLAS_OP_N, CUBLAS_OP_T, num_patches, patch_size, num_filters/group, 
						alpha, 
						&delta_dev[(i*group +j) * num_filters /group* num_patches], num_patches, 
						&weight_dev[j*weight_size/group], patch_size, 
						beta2, 
						workspace_dev, num_patches);
#else
				cublasSgemm(network->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, num_patches, patch_size,
						num_filters/group, &alpha, &delta_dev[(i*group +j) * num_filters /group* num_patches],
						num_patches, &weight_dev[j*weight_size/group], patch_size, &beta2, workspace_dev, num_patches);
#endif

				_col2im_(workspace_dev, input_channel/group, input_height, input_width, filter_size, stride,
						padding, &prev_delta[(i*group+j) * input_channel /group* input_height * input_width]);
			}
		}
	}
}

extern "C++" void convolutional_layer_t::_update_() {
    const float learning_rate = network->learning_rate/network->batch_size;
    const float momentum = network->momentum;
    const float decay = -network->decay*network->batch_size;

#ifdef CUSTOM_BLAS
    _axpy_(weight_size, decay, weight_dev, 1, weight_update_dev, 1);
    _axpy_(weight_size, learning_rate, weight_update_dev, 1, weight_dev, 1);
    _scal_(weight_size, momentum, weight_update_dev, 1);

    _axpy_(num_filters, learning_rate, bias_update_dev, 1, bias_dev, 1);
    _scal_(num_filters, momentum, bias_update_dev, 1);
#else
    // Weight update
    cublasSaxpy(network->cublas_handle, weight_size, 
                &decay, weight_dev, 1, weight_update_dev, 1);
    cublasSaxpy(network->cublas_handle, weight_size, 
                &learning_rate, weight_update_dev, 1, weight_dev, 1);
    cublasSscal(network->cublas_handle, weight_size, 
                &momentum, weight_update_dev, 1);

    // Bias update
    cublasSaxpy(network->cublas_handle, num_filters, 
                &learning_rate, bias_update_dev, 1, bias_dev, 1);
    cublasSscal(network->cublas_handle, num_filters, &momentum, bias_update_dev, 1);
#endif

#ifdef PRUNING
    if(pruning) {
        cudaMemcpy(weight, weight_dev, weight_size*sizeof(float), cudaMemcpyDeviceToHost);
        for(unsigned i = 0; i < weight_size; i++) {
            if(weight[i] < weight_threshold && weight[i] > -weight_threshold) {
                weight[i] = 0.0;
            }
        }
        cudaMemcpy(weight_dev, weight, weight_size*sizeof(float), cudaMemcpyHostToDevice);
    }
#endif

#ifdef CHANNEL_PRUNING
    if(pruning) {
        cudaMemcpy(weight, weight_dev, weight_size*sizeof(float), cudaMemcpyDeviceToHost);
        std::vector<unsigned> indices(input_channel);
        std::vector<unsigned> sorted(input_channel);
        std::vector<float> weight_sum(input_channel);
        for(unsigned i = 0; i < input_channel; i++) {
            sorted[i] = i;
        }

        for(unsigned i = 0; i < output_channel; i++) {
            // Initialize weight sum.
            weight_sum.assign(input_channel, 0.0);
            for(unsigned j = 0; j < input_channel; j++) {
                for(unsigned k = 0; k < filter_size*filter_size; k++) {
                    weight_sum[j] += weight[i*input_channel*filter_size*filter_size + j*filter_size*filter_size + k];
                }
            }
            indices = sorted;
            std::sort(indices.begin(), indices.end(), [&](unsigned a, unsigned b) {
                return weight_sum[a] < weight_sum[b];
            });

            // Set weight data to 0.
            for(unsigned j = 0; j < channel_pruning; j++) {
                for(unsigned k = 0; k < filter_size*filter_size; k++) {
                    weight[i*input_channel*filter_size*filter_size + indices[j]*filter_size*filter_size + k] = 0.0;
                }
            }
        }
        cudaMemcpy(weight_dev, weight, weight_size*sizeof(float), cudaMemcpyHostToDevice);
    }
#endif
}


extern "C++" void convolutional_layer_t::_forward_batchnorm_() {

    unsigned num_patches = output_height * output_width;
    cudaMemcpy(x_dev, output_data_dev, 
               output_size * network->batch_size * sizeof(float), cudaMemcpyDeviceToDevice);

    if(network->run_type == TRAIN_RUN) {
        
        // Update normalize mean value.
        _batchnorm_mean_(output_data_dev, normalize_mean_dev, 
                         output_channel, num_patches, network->batch_size);

        // Update normalize variance value.
        _batchnorm_variance_(output_data_dev, normalize_mean_dev, normalize_variance_dev, 
                             output_channel, num_patches, network->batch_size);

        float rolling = 0.99;
        float normalize = 0.01;

#ifdef CUSTOM_BLAS
        _scal_(output_channel, rolling, rolling_mean_dev, 1);
        _axpy_(output_channel, normalize, normalize_mean_dev, 1, rolling_mean_dev, 1);

        _scal_(output_channel, rolling, rolling_variance_dev, 1);
        _axpy_(output_channel, normalize, normalize_variance_dev, 1, rolling_variance_dev, 1);
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
                              output_channel, num_patches, network->batch_size);
        cudaMemcpy(normalize_x_dev, output_data_dev, 
                   output_size * network->batch_size * sizeof(float), cudaMemcpyDeviceToDevice);
    }
    else {
        _batchnorm_normalize_(output_data_dev, rolling_mean_dev, rolling_variance_dev, 
                              output_channel, num_patches, network->batch_size);
    }
    // Scale down the output data.
    _batchnorm_scale_down_(output_data_dev, scale_dev, 
                           output_channel, num_patches, network->batch_size);
}

extern "C++" void convolutional_layer_t::_backward_batchnorm_() {
    unsigned num_patches = output_height * output_width;
    _batchnorm_scale_down_(delta_dev, scale_dev, 
                           output_channel, num_patches, network->batch_size);

    // Calculate delta value of normalize mean.
    _batchnorm_mean_delta_(delta_dev, normalize_variance_dev, mean_delta_dev, 
                           output_channel, num_patches, network->batch_size); 
    // Calculate delta value of normalize variance.
    _batchnorm_variance_delta_(x_dev, delta_dev, normalize_mean_dev, 
                               normalize_variance_dev, variance_delta_dev, 
                               output_channel, num_patches, network->batch_size);

    // Normalize.
    _batchnorm_normalize_delta_(x_dev, normalize_mean_dev, normalize_variance_dev, 
                                mean_delta_dev, variance_delta_dev, delta_dev, 
                                output_channel, num_patches, network->batch_size); 
}

}
// End of namespace nebula.
