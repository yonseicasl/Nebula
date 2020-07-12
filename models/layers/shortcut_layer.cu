extern "C++" {
#include "shortcut_layer.h"
#include "gemm.h"
}

namespace nebula {

__global__ void _shortcut_(unsigned m_width, unsigned m_height, unsigned m_channel, unsigned m_stride, unsigned m_sample,
                           unsigned m_input_width, unsigned m_input_height, unsigned m_input_channel, float *m_input_data_dev, 
                           unsigned m_output_width, unsigned m_output_height, unsigned m_output_channel, float *m_output_data_dev, unsigned m_batch) {
    unsigned index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= m_batch * m_width * m_height * m_channel) return;
    unsigned i = index % m_width;
    index /= m_width;
    unsigned j = index % m_height;
    index /= m_height;
    unsigned k = index % m_channel;
    index /= m_channel;
    unsigned b = index % m_batch;

    unsigned output_index = i * m_sample + m_output_width * (j * m_sample + m_output_height * (k + m_output_channel * b));
    unsigned input_index = i * m_stride + m_input_width * (j * m_stride + m_input_height * (k + m_input_channel * b));
    m_output_data_dev[output_index] += m_input_data_dev[input_index];
}

// Forward propagation
extern "C++" void shortcut_layer_t::_forward_() {
    
    cudaMemset(output_data_dev, 0.0, output_size * network->batch_size * sizeof(float));
    cudaMemset(delta_dev, 0.0, output_size * network->batch_size * sizeof(float));

    float *input_data_dev = prev_layer ? prev_layer->output_data_dev : network->input_data_dev;
    cudaMemcpy(output_data_dev, input_data_dev, output_size * network->batch_size * sizeof(float), cudaMemcpyDeviceToDevice);
    
    unsigned stride = input_width / output_width > 1 ? input_width / output_width : 1;
    unsigned sample = input_height / output_height > 1 ? input_height / output_height : 1;
    
    unsigned width   = input_width < output_width ? 
                       input_width : output_width;
    unsigned height  = input_height < output_height ?
                       input_height : output_height;
    unsigned channel = input_channel < output_channel ?
                       input_channel : output_channel;

    dim3 cuda_griddim = {(network->batch_size * width * height * channel -1) / BLOCK_SIZE + 1, 1, 1};
    _shortcut_<<<cuda_griddim, BLOCK_SIZE>>>(width, height, channel, stride, sample,
                                             input_width, input_height, input_channel, connection->output_data_dev, 
                                             output_width, output_height, output_channel, output_data_dev, network->batch_size);
     
    // Activate function
    _activate_();
}


// Backward propagation
extern "C++" void shortcut_layer_t::_backward_() {

    _gradient_();
    const float alpha = 1.0;
#ifdef CUSTOM_BLAS
    _axpy_(output_size * network->batch_size, alpha, delta_dev, 1, prev_layer->delta_dev, 1);
#else
    cublasSaxpy(network->cublas_handle, output_size*network->batch_size, &alpha, delta_dev, 1, prev_layer->delta_dev, 1);
#endif
   
    unsigned stride = output_width / input_width > 1 ? output_width / input_width : 1;
    unsigned sample = input_width / output_width > 1 ? input_width / output_width : 1;
     
    unsigned width = output_width < input_width ? 
                     output_width : input_width;
    unsigned height = output_height < input_height ? 
                      output_height : input_height;
    unsigned channel = output_channel < input_channel ?
                       output_channel : input_channel;

    dim3 cuda_griddim = {(network->batch_size * width * height * channel - 1) / BLOCK_SIZE + 1, 1, 1};
    _shortcut_<<<cuda_griddim, BLOCK_SIZE>>>(width, height, channel, stride, sample,
                                             output_width, output_height, output_channel, delta_dev,
                                             input_width, input_height, input_channel, connection->delta_dev, network->batch_size);
                            
}

extern "C++" void shortcut_layer_t::_update_() {
    // Nothing to do.    
}


}
// End of namespace nebula.
