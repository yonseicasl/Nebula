#include <algorithm>
#include <functional>
#include <thread>
#include <vector>
#include <random>
#include <iostream>
#include <cmath>
#ifndef CUSTOM_BLAS
	#include <cblas.h>
#endif
#include <cstring>
#include "utils.h"

namespace nebula {

// Convert string to lowercase.
std::string& lowercase(std::string &m_str) {
    transform(m_str.begin(), m_str.end(), m_str.begin(), ::tolower);
    return m_str;
}

// Convert string to uppercase.
std::string& uppercase(std::string &m_str) {
    transform(m_str.begin(), m_str.end(), m_str.begin(), ::toupper);
    return m_str;
}

void shortcut(unsigned num_threads, 
              unsigned m_input_width, unsigned m_input_height, unsigned m_input_channel, float *m_input_data,
              unsigned m_output_width, unsigned m_output_height, unsigned m_output_channel, float *m_output_data, unsigned m_batch){

    unsigned stride = m_input_width/m_output_width > 1 ? m_input_width/m_output_width : 1;
    unsigned sample = m_output_width/m_input_width > 1 ? m_output_width/m_input_width : 1;

    unsigned width = m_input_width < m_output_width ?
                     m_input_width : m_output_width;
    unsigned height = m_input_height < m_output_height ? 
                      m_input_height : m_output_height;
    unsigned channel = m_input_channel < m_output_channel ?
                       m_input_channel : m_output_channel;
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    for(unsigned tid = 0; tid < num_threads; tid++) {
        threads.emplace_back(std::bind([&](const unsigned begin, const unsigned end,
                                           const unsigned tid) {
            for(unsigned i = begin; i < end; i++) {
                for(unsigned c = 0; c < channel; c++) {
                    for(unsigned h = 0; h < height; h++) {
                        for(unsigned w = 0; w < width; w++) {
                            unsigned output_index = w * sample + m_output_width * ( h * sample + m_output_height * (c + m_output_channel * i));
                            unsigned input_index = w * stride + m_input_width * ( h * stride + m_input_height * ( c + m_input_channel * i));
                            m_output_data[output_index] +=  m_input_data[input_index];
                        }
                    }
                }
            }
        }, tid * m_batch / num_threads,
           (tid + 1) * m_batch / num_threads, tid));
    } std::for_each(threads.begin(), threads.end(), [](std::thread& t) {t.join(); });
}

// Sampling the selected data 
void sampling(float *m_sample, float *m_probability, unsigned m_size, unsigned num_threads) {
    std::minstd_rand generator(std::random_device{}());
    std::uniform_real_distribution<float> distribution(0.0, 1.0);

    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    for(unsigned tid = 0; tid < num_threads; tid++) {
        threads.emplace_back(std::bind([&](const unsigned begin, const unsigned end,
                                           const unsigned tid) {
            for(unsigned i = begin; i < end; i++) {
                if(distribution(generator) < m_probability[i]) {m_sample[i] = 1.0;}
                else {m_sample[i] = 0.0;}
            }
        }, tid * m_size / num_threads, (tid + 1) * m_size / num_threads, tid));
    } std::for_each(threads.begin(), threads.end(), [](std::thread& t) { t.join(); });
}

// Unfold data.
void im2col(float* m_im_data, unsigned m_channel, unsigned m_height, unsigned m_width,
            unsigned m_filter_size, unsigned m_stride, unsigned m_padding, float *m_col_data,
            unsigned m_num_threads) {
    unsigned col_height = (m_height + 2 * m_padding - m_filter_size) / m_stride + 1;
    unsigned col_width = (m_width + 2 * m_padding - m_filter_size) / m_stride + 1;
    unsigned col_channel = m_channel * m_filter_size * m_filter_size;

    std::vector<std::thread> threads;
    threads.reserve(m_num_threads);

    for(unsigned tid = 0; tid < m_num_threads; tid++) {
        threads.emplace_back(std::bind([&](const unsigned begin, const unsigned end,
                                           const unsigned tid) {
            for (unsigned i = begin; i < end; i++) {
                unsigned offset_w = i % m_filter_size;
                unsigned offset_h = (i / m_filter_size) % m_filter_size;
                unsigned im_c = i / m_filter_size / m_filter_size;

                for (unsigned h = 0; h < col_height; h++) {
                    for (unsigned w = 0; w < col_width; w++) {
                        unsigned im_row = offset_h + h * m_stride;
                        unsigned im_col = offset_w + w * m_stride;
                        unsigned col_index = (i * col_height + h) * col_width + w;
                        im_row -= m_padding;
                        im_col -= m_padding;
                
                        if (im_row < 0 || im_col < 0 || im_row >= m_height || im_col >= m_width) {
                            m_col_data[col_index] = 0;
                        }
                        else {
                            m_col_data[col_index] =
                            m_im_data[im_col + m_width * (im_row + m_height * im_c)];
                        }
                    }   
                }
            }
        }, tid * col_channel / m_num_threads, (tid + 1) * col_channel / m_num_threads, tid));
    } std::for_each(threads.begin(), threads.end(), [](std::thread& t) { t.join(); });
}

// Fold data.
void col2im(float* m_col_data, unsigned m_channel, unsigned m_height, unsigned m_width,
            unsigned m_filter_size, unsigned m_stride, unsigned m_padding, float *m_im_data,
            unsigned m_num_threads) {
    unsigned col_height = (m_height + 2 * m_padding - m_filter_size) / m_stride + 1;
    unsigned col_width = (m_width + 2 * m_padding - m_filter_size) / m_stride + 1;
    unsigned col_channel = m_channel * m_filter_size * m_filter_size;

    std::vector<std::thread> threads;
    threads.reserve(m_num_threads);

    for(unsigned tid = 0; tid < m_num_threads; tid++) {
        threads.emplace_back(std::bind([&](const unsigned begin, const unsigned end,
                                           const unsigned tid) {
            for(unsigned i = begin; i < end; i++) {
                unsigned offset_w = i % m_filter_size;
                unsigned offset_h = (i / m_filter_size) % m_filter_size;
                unsigned im_c = i / m_filter_size / m_filter_size;

                for (unsigned h = 0; h < col_height; ++h) {
                    for (unsigned w = 0; w < col_width; ++w) {
                        unsigned im_row = offset_h + h * m_stride;
                        unsigned im_col = offset_w + w * m_stride;
                        unsigned col_index = (i * col_height + h) * col_width + w;
		                im_row -= m_padding;
		                im_col -= m_padding;

		                if (im_row < 0 || im_col < 0 || im_row >= m_height || im_col >= m_width) {
                            // Nothing to do
                        }
                        else {
                            m_im_data[im_col + m_width * (im_row + m_height * im_c)] +=
                            m_col_data[col_index];
                        }
                    }
                }
            }
        }, tid * col_channel / m_num_threads, (tid + 1) * col_channel / m_num_threads, tid));
    } std::for_each(threads.begin(), threads.end(), [](std::thread& t) { t.join(); });
}

void forward_bias(unsigned num_threads, float *m_output, float *m_bias,
                  unsigned m_channel, unsigned m_size, unsigned m_batch) {
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    for(unsigned tid = 0; tid < num_threads; tid++) {
        threads.emplace_back(std::bind([&](const unsigned begin, const unsigned end,
                                           const unsigned tid) {
            for(unsigned i = begin; i < end; i++) {
                for(unsigned j = 0; j < m_channel; j++) {
                    for(unsigned k = 0; k < m_size; k++) {
                        m_output[(i * m_channel + j) * m_size + k] += m_bias[j];
                    }
                }
            }
        }, tid * m_batch / num_threads,
           (tid + 1) * m_batch / num_threads, tid));
    } std::for_each(threads.begin(), threads.end(), [](std::thread& t) { t.join(); });
}

void backward_bias(unsigned num_threads, float *m_bias_update, float *m_delta, 
                   unsigned m_channel, unsigned m_size, unsigned m_batch){
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    for(unsigned tid = 0; tid < num_threads; tid++) {
        threads.emplace_back(std::bind([&](const unsigned begin,const unsigned end, const unsigned tid) {
            for(unsigned batch = begin; batch < end; batch++) {
                for(unsigned channel = 0; channel < m_channel; channel++) {
                    for(unsigned size = 0; size < m_size; size++) {
                        m_bias_update[channel] += m_delta[m_size * (channel + batch * m_channel) + size];
                    }
                }
            }
        }, tid * m_batch / num_threads,
           (tid +1) * m_batch /num_threads, tid));
    } std::for_each(threads.begin(), threads.end(), [](std::thread& t) {t.join(); });
}

}
// End of namespace nebula.
