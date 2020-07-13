#ifndef __UTILS_H__
#define __UTILS_H__

#include <string>
#include <vector>

namespace nebula {

// Convert string to lowercase.
std::string& lowercase(std::string &m_str);

// Convert string to uppercase.
std::string& uppercase(std::string &m_str);
   
void shortcut(unsigned num_threads, 
              unsigned m_input_width, unsigned m_input_height, unsigned m_input_channel, float *m_input_data,
              unsigned m_output_width, unsigned m_output_height, unsigned m_output_channel, float *m_output_data, unsigned m_batch);

// Sampling the selected data 
void sampling(float *m_sample, float *m_probability, unsigned m_size, unsigned num_threads);

#ifdef GPU_ENABLED

void _sampling_(float *m_sample, float *m_probability, unsigned m_size);

void _update_bias_unit_(float *m_bias_unit, float *m_zero_value, float *m_k_value, unsigned m_size, unsigned m_batch);

#endif

// Unfold data.
void im2col(float *m_im_data, unsigned m_channel, unsigned m_height, unsigned m_width,
            unsigned m_filter_size, unsigned m_stride, unsigned m_padding, float *m_col_data,
            unsigned m_num_threads);

// Fold data.
void col2im(float *m_col_data, unsigned m_channel, unsigned m_height, unsigned m_width,
            unsigned m_filter_size, unsigned m_stride, unsigned m_padding, float *m_im_data,
            unsigned m_num_threads);

void forward_bias(unsigned num_threads, float *m_output, float *m_bias,
                  unsigned m_channel, unsigned m_size, unsigned m_batch);
void backward_bias(unsigned num_threads, float *m_bias_update, float *m_delta, 
                   unsigned m_channel, unsigned m_size, unsigned m_batch);


#ifdef GPU_ENABLED
// Unfold data.
void _im2col_(float *m_im_data, unsigned m_channel, unsigned m_height, unsigned m_width,
              unsigned m_filter_size, unsigned m_stride, unsigned m_padding, float *m_col_data);

// Fold data.
void _col2im_(float *m_col_data, unsigned m_channel, unsigned m_height, unsigned m_width,
              unsigned m_filter_size, unsigned m_stride, unsigned m_padding, float *m_im_data);
// Add bias for forward propagation.
void _forward_bias_(float *m_output_data_dev, float *m_bias_dev, 
                unsigned m_channel, unsigned m_size, unsigned m_batch);

// Add bias for backward propagation.
void _backward_bias_(float *m_bias_update_dev, float *m_delta_dev, 
                     unsigned m_channel, unsigned m_size, unsigned m_batch);
#endif

}
// End of namespace nebula.
#endif

