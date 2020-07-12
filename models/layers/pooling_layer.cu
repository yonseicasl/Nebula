extern "C++" {
#include "pooling_layer.h"
}

namespace nebula {

__global__ void _forward_maxpool_(unsigned m_total_size, unsigned m_input_height,
                                  unsigned m_input_width, unsigned m_input_channel,
                                  unsigned m_stride, unsigned m_filter_size, unsigned m_padding,
                                  float *m_input_data, float *m_output_data, unsigned *m_index) {
    size_t tid = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x; 
    if(tid >= m_total_size) { return; }
    
    unsigned output_width = (m_input_width + 2 * m_padding) / m_stride;
    unsigned output_height = (m_input_height + 2 * m_padding) / m_stride;
    unsigned output_channel = m_input_channel;

    unsigned w = tid % output_width;
    tid /= output_width; 
    unsigned h = tid % output_height;
    tid /= output_height;
    unsigned c = tid % output_channel;
    tid /= output_channel;

    int output_index = w + output_width * (h + output_height * (c + output_channel * tid));
    float max_value = 0.0 - INFINITY;
    int max_index = -1;
    for(int i = 0; i < m_filter_size; i++) {
        for(int j = 0; j < m_filter_size; j++) {
            int current_height = h * m_stride + i - m_padding;
            int current_width  = w * m_stride + j - m_padding;
            unsigned index = current_width + m_input_width *
                             (current_height + m_input_height * (c + tid * m_input_channel));
            bool valid = (current_height >= 0 && current_height < m_input_height
                         && current_width >= 0 && current_width < m_input_width);
            float val = (valid == true) ? m_input_data[index] : 0.0 - INFINITY;
            max_index = (val > max_value) ? index : max_index;
            max_value = (val > max_value) ? val   : max_value;
        }
    }
    m_output_data[output_index] = max_value;
    m_index[output_index] = max_index;
}

__global__ void _forward_avgpool_(unsigned m_total_size, unsigned m_input_height,
                                  unsigned m_input_width, unsigned m_input_channel,
                                  unsigned m_stride, unsigned m_filter_size, unsigned m_padding,
                                  float *m_input_data, float *m_output_data) {
    size_t tid = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x; 
    if(tid >= m_total_size) { return; }
    
    unsigned output_width = (m_input_width + 2 * m_padding) / m_stride;
    unsigned output_height = (m_input_height + 2 * m_padding) / m_stride;
    unsigned output_channel = m_input_channel;

    unsigned w = tid % output_width;
    tid /= output_width; 
    unsigned h = tid % output_height;
    tid /= output_height;
    unsigned c = tid % output_channel;
    tid /= output_channel;

    int output_index = w + output_width * (h + output_height * (c + output_channel * tid));
    float val = 0.0;
    for(int i = 0; i < m_filter_size; i++) {
        for(int j = 0; j < m_filter_size; j++) {
            int current_height = h * m_stride + i - m_padding;
            int current_width  = w * m_stride + j - m_padding;
            unsigned index = current_width + m_input_width *
                             (current_height + m_input_height * (c + tid * m_input_channel));
            bool valid = (current_height >= 0 && current_height < m_input_height
                         && current_width >= 0 && current_width < m_input_width);
            val += (valid == true) ? m_input_data[index] : 0.0;
        }
    }
    m_output_data[output_index] = val/(m_filter_size * m_filter_size);
}

__global__ void _backward_maxpool_(unsigned m_total_size, unsigned m_input_height,
                                   unsigned m_input_width, unsigned m_input_channel,
                                   unsigned m_stride, unsigned m_filter_size, unsigned m_padding,
                                   float *m_delta, float *m_prev_delta, unsigned *m_index) {
    size_t tid = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if(tid >= m_total_size) { return; }

    unsigned output_height = (m_input_height + 2 * m_padding) / m_stride;
    unsigned output_width = (m_input_width + 2 * m_padding) / m_stride;
    unsigned output_channel = m_input_channel;
    unsigned area = (m_filter_size - 1) / m_stride;
    unsigned t = tid;

    unsigned w = tid % m_input_width;
    tid /= m_input_width;
    unsigned h = tid % m_input_height;
    tid /= m_input_height;
    unsigned c = tid % m_input_channel;
    tid /= m_input_channel;

    float delta = 0.0;
    for(int i = 0 - area; i < area + 1; i++) {
        for(int j = 0 - area; j < area + 1; j++) {
            int current_width = (w + m_padding) / m_stride + j;
            int current_height = (h + m_padding) / m_stride + i;
            unsigned output_index = current_width + output_width * (current_height +
                                    output_height * (c + output_channel * tid));
            bool valid = (current_height >= 0 && current_height < output_height
                         && current_width >= 0 && current_width < output_width);
            delta += (valid && m_index[output_index] == t) ? m_delta[output_index] : 0;
        }
    }
    m_prev_delta[t] += delta;
}

__global__ void _backward_avgpool_(unsigned m_total_size, unsigned m_input_height,
                                   unsigned m_input_width, unsigned m_input_channel,
                                   unsigned m_stride, unsigned m_filter_size, unsigned m_padding,
                                   float *m_delta, float *m_prev_delta) {
    
    size_t tid = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if(tid >= m_total_size) {return;}

    unsigned output_height = (m_input_height + 2 * m_padding) / m_stride;
    unsigned output_width = (m_input_width + 2 * m_padding) / m_stride;
    unsigned output_channel = m_input_channel;
    unsigned area = (m_filter_size -1) / m_stride;
    unsigned t = tid;

    unsigned w = tid % m_input_width;
    tid /= m_input_width;
    unsigned h = tid % m_input_height;
    tid /= m_input_height;
    unsigned c = tid % m_input_channel;
    tid /= m_input_channel;
   
    float delta = 0.0;
    for(int i = 0 - area; i < area + 1; i++) {
        for(int j = 0 - area; j < area + 1; j++) {
            int current_width = (w + m_padding) / m_stride + j;
            int current_height = (h + m_padding) / m_stride + i;
            unsigned output_index = current_width + output_width * (current_height +
                                    output_height * (c + output_channel * tid));
            bool valid = (current_height >= 0 && current_height < output_height
                         && current_width >= 0 && current_width < output_width);
                delta += valid ? m_delta[output_index] : 0.0;
        }
    }
    m_prev_delta[t] += delta / (m_filter_size * m_filter_size); 
}

extern "C++" void pooling_layer_t::_forward_() {
    float *input_data_dev = prev_layer ? prev_layer->output_data_dev : network->input_data_dev;
    cudaMemset(output_data_dev, 0.0, output_size * network->batch_size * sizeof(float));
    cudaMemset(delta_dev, 0.0, output_size * network->batch_size * sizeof(float));                            
    dim3 cuda_griddim = {(output_size * network->batch_size - 1) / BLOCK_SIZE + 1, 1, 1};
    
    if(layer_type == MAXPOOL_LAYER) {
        _forward_maxpool_<<<cuda_griddim, BLOCK_SIZE>>>(output_size * network->batch_size, input_height, input_width, input_channel, stride, filter_size, padding, input_data_dev, output_data_dev, index_dev);
    
    }
    else if(layer_type == AVGPOOL_LAYER) {
        _forward_avgpool_<<<cuda_griddim, BLOCK_SIZE>>>(output_size * network->batch_size, input_height, input_width, input_channel, stride, filter_size, padding, input_data_dev, output_data_dev);
    }
    else {
        std::cerr << "unknown pooling layer type" << std::endl;
        exit(1);
    }

}
extern "C++" void pooling_layer_t::_backward_() {
    float *prev_delta_dev = prev_layer ? prev_layer->delta_dev : NULL;
    dim3 cuda_griddim = {(input_size * network->batch_size -1) / BLOCK_SIZE + 1, 1, 1};
    if(layer_type == MAXPOOL_LAYER) {
        _backward_maxpool_<<<cuda_griddim, BLOCK_SIZE>>>(input_size * network->batch_size, input_height, input_width, input_channel, stride, filter_size, padding, delta_dev, prev_delta_dev, index_dev);
    }
    else if(layer_type == AVGPOOL_LAYER) {
        _backward_avgpool_<<<cuda_griddim, BLOCK_SIZE>>>(input_size * network->batch_size, input_height, input_width, input_channel, stride, filter_size, padding, delta_dev, prev_layer->delta_dev);
    }
    else {
        std::cerr << "unknown pooling layer type" << std::endl;
        exit(1);
    }
}
// Layer update
extern "C++" void pooling_layer_t::_update_(){
    // Nothing to do
}

}
// End of namespace nebula.
