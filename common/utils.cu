extern "C++" {
#include "utils.h"
#include "def.h"
}

__global__ void _unfold_(unsigned m_size, float* m_im_data, unsigned m_height, unsigned m_width,
                         unsigned m_filter_size, unsigned m_padding, unsigned m_stride,
                         unsigned m_col_height, unsigned m_col_width, float *m_col_data) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    for(; index < m_size; index += blockDim.x * gridDim.x) {
        int w_out = index % m_col_width;
        int w_in = w_out * m_stride - m_padding;

        int h_index = index / m_col_width;
        int h_out = h_index % m_col_height;
        int h_in = h_out * m_stride - m_padding;

        int channel_in = h_index / m_col_height;
        int channel_out = channel_in * m_filter_size * m_filter_size;

        m_col_data += (channel_out * m_col_height + h_out) * m_col_width + w_out;
        m_im_data += (channel_in * int(m_height) + h_in) * int(m_width) + w_in;

        for (int i = 0; i < m_filter_size; ++i) {
            for (int j = 0; j < m_filter_size; ++j) {
                int h = h_in + i;
                int w = w_in + j;

                *m_col_data = (h >= 0 && w >= 0 && h < int(m_height) && w < int(m_width)) ?
                              m_im_data[i * int(m_width) + j] : 0.0;
                m_col_data += m_col_height * m_col_width;
            }
        }
    }
}

// Unfold data.
void _im2col_(float *m_im_data, unsigned m_channel, unsigned m_height, unsigned m_width,
              unsigned m_filter_size, unsigned m_stride, unsigned m_padding, float *m_col_data) {
    unsigned col_height= (m_height + 2 * m_padding - m_filter_size) / m_stride + 1;
    unsigned col_width = (m_width + 2 * m_padding - m_filter_size) / m_stride + 1;
    unsigned kernel_size = col_height * col_width * m_channel;
    _unfold_<<<(kernel_size + BLOCK_SIZE - 1 ) /BLOCK_SIZE, BLOCK_SIZE>>>
    (kernel_size, m_im_data, m_height, m_width, m_filter_size, m_padding, m_stride,
     col_height, col_width, m_col_data);
}

__global__ void _fold_(unsigned m_size, float *m_col_data, unsigned m_height, unsigned m_width,
                       unsigned m_filter_size, unsigned m_padding, unsigned m_stride,
                       unsigned m_col_height, unsigned m_col_width, float *m_im_data) {
    size_t index = blockIdx.x * blockDim.x +threadIdx.x;
    for(; index < m_size; index += blockDim.x * gridDim.x) {
        float val = 0;
        int w = index % int(m_width) + m_padding;
        int h = (index / int(m_width)) % int(m_height) + m_padding;
        int c = index / (int(m_width) * int(m_height));

        // compute the start and end of the output
        int col_w_start = (w < m_filter_size) ? 0 : (w - m_filter_size) / m_stride + 1;
        int col_w_end = min(w / m_stride + 1, m_col_width);
        int col_h_start = (h < m_filter_size) ? 0 : (h - m_filter_size) / m_stride + 1;
        int col_h_end = min(h / m_stride + 1, m_col_height);

        // equivalent implementation
        int offset = (c * m_filter_size * m_filter_size + h * m_filter_size + w) *
                     m_col_height * m_col_width;
        int col_h_coeff = (1 - m_stride * m_filter_size * m_col_height) * m_col_width;
        int col_w_coeff= (1 - m_stride * m_col_height * m_col_width);

        for (int h_col = col_h_start; h_col < col_h_end; ++h_col) {
            for (int w_col = col_w_start; w_col < col_w_end; ++w_col) {
                val += m_col_data[offset + h_col * col_h_coeff + w_col * col_w_coeff];
            }
        }
        m_im_data[index] += val;
    }
}

// Fold data.
void _col2im_(float *m_col_data, unsigned m_channel, unsigned m_height, unsigned m_width,
              unsigned m_filter_size, unsigned m_stride, unsigned m_padding, float *m_im_data) {
    unsigned col_height = (m_height + 2 * m_padding - m_filter_size) / m_stride + 1;
    unsigned col_width= (m_width + 2 * m_padding - m_filter_size) / m_stride + 1;
    unsigned kernel_size = m_height * m_width * m_channel;
    _fold_<<<(kernel_size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>
    (kernel_size, m_col_data, m_height, m_width, m_filter_size, m_padding, m_stride, 
     col_height, col_width, m_im_data);
}

__global__ void _add_bias_(float *m_output_data_dev, float *m_bias_dev, 
                           unsigned m_channel, unsigned m_size, unsigned m_batch) {
    unsigned i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if(i >= m_channel * m_size * m_batch) {return ;}
    unsigned j = i % m_size;
    i /= m_size;
    unsigned k = i % m_channel;
    i /= m_channel;
    m_output_data_dev[(i * m_channel + k) * m_size + j] += m_bias_dev[k];
}

void _forward_bias_(float *m_output_data_dev, float *m_bias_dev, 
                   unsigned m_channel, unsigned m_size, unsigned m_batch) {
    
    dim3 cuda_griddim = {(m_channel * m_size * m_batch -1) / BLOCK_SIZE + 1, 1, 1};
    _add_bias_<<<cuda_griddim, BLOCK_SIZE>>>(m_output_data_dev, m_bias_dev, m_channel, m_size, m_batch); 
}

__global__ void _add_delta_(float *m_bias_update_dev, float *m_delta_dev,
                            unsigned m_channel, unsigned m_size, unsigned m_batch) {
    
    unsigned i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if(i >= m_channel * m_size * m_batch) {return ;}
    unsigned j = i % m_size;
    i /= m_size;
    unsigned k = i % m_channel;
    i /= m_channel;
    m_bias_update_dev[k] += m_delta_dev[m_size * (k + i * m_channel) + j];
}

void _backward_bias_(float *m_bias_update_dev, float *m_delta_dev, 
                     unsigned m_channel, unsigned m_size, unsigned m_batch) {

    dim3 cuda_griddim = {(m_channel * m_size * m_batch - 1) / BLOCK_SIZE + 1, 1, 1};
    _add_delta_<<<cuda_griddim, BLOCK_SIZE>>>(m_bias_update_dev, m_delta_dev, m_channel, m_size, m_batch);
}
