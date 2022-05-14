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
#include "batchnorm.h"

namespace nebula {

void batchnorm_mean(unsigned num_threads, float *m_output, float *m_mean, 
                    unsigned m_channel, unsigned m_size, unsigned m_batch){
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    for(unsigned tid = 0; tid < num_threads; tid++) {
        threads.emplace_back(std::bind([&](const unsigned begin, const unsigned end,
                                           const unsigned tid) {
            for(unsigned i = begin; i < end; i++) {
                m_mean[i] = 0.0;
                for(unsigned j = 0; j < m_batch; j++) {
                    for(unsigned k = 0; k < m_size; k++) {
                        m_mean[i] += m_output[j * m_channel * m_size + i * m_size + k];
                    }
                }
                m_mean[i] *= 1.0/(m_batch * m_size);
            }
        }, tid * m_channel / num_threads,
           (tid + 1) * m_channel / num_threads, tid));
    } std::for_each(threads.begin(), threads.end(), [](std::thread& t) {t.join(); });
}

void batchnorm_variance(unsigned num_threads, float *m_output, float *m_mean, float *m_variance, 
                        unsigned m_channel, unsigned m_size, unsigned m_batch){
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    for(unsigned tid = 0; tid < num_threads; tid++) {
        threads.emplace_back(std::bind([&](const unsigned begin, const unsigned end,
                                      const unsigned tid) {
            for(unsigned i=begin; i < end; i++) {
                m_variance[i] = 0.0;
                for(unsigned j = 0; j < m_batch; j++) {
                    for(unsigned k = 0; k < m_size; k++) {
                        m_variance[i] += pow((m_output[ j * m_channel * m_size + i * m_size + k] - m_mean[i]), 2);
                    }
                }
                m_variance[i] *= 1.0/(m_batch * m_size);
            }
        }, tid * m_channel / num_threads,
           (tid + 1) * m_channel / num_threads, tid));
    } std::for_each(threads.begin(), threads.end(), [](std::thread& t) {t.join(); });
}

void batchnorm_normalize(unsigned num_threads, float *m_output, float *m_mean, float *m_variance,
                         unsigned m_channel, unsigned m_size, unsigned m_batch){
    
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    for(unsigned tid =0; tid < num_threads; tid++) {
        threads.emplace_back(std::bind([&](const unsigned begin, const unsigned end,
                                           const unsigned tid) {
            for(unsigned i = begin; i < end; i++) {
                for(unsigned j = 0; j < m_channel; j++) {
                    for(unsigned k = 0; k < m_size; k++) {
                        unsigned index = i * m_size * m_channel + j * m_size + k;
                        m_output[index] = (m_output[index] - m_mean[j])/(sqrt(m_variance[j]) + 0.00001);
                    }
                }
            }
        }, tid * m_batch / num_threads,
           (tid + 1) * m_batch / num_threads, tid));
    } std::for_each(threads.begin(), threads.end(), [](std::thread& t) {t.join(); });
}

void batchnorm_scale_down(unsigned num_threads, float *m_output, float *m_scale,
                          unsigned m_channel, unsigned m_size, unsigned m_batch){

    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    for(unsigned tid = 0; tid < num_threads; tid++) {
        threads.emplace_back(std::bind([&](const unsigned begin, const unsigned end,
                                           const unsigned tid) {
            for(unsigned i = begin; i < end; i++) {
                for(unsigned j = 0; j < m_channel; j++) {
                    for(unsigned k = 0; k <  m_size; k++) {
                        unsigned index = i * m_size * m_channel + j * m_size + k;
                        m_output[index] *= m_scale[j];
                        // m_output[(i *  m_channel + j) * m_size + k] *= m_scale[j];
                    }
                }
            }
        }, tid * m_batch / num_threads,
           (tid + 1) * m_batch / num_threads, tid));
    } std::for_each(threads.begin(), threads.end(), [](std::thread& t) { t.join(); });
}

void batchnorm_add_beta(unsigned num_threads, float *m_output, float *m_beta,
                          unsigned m_channel, unsigned m_size, unsigned m_batch){

    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    for(unsigned tid = 0; tid < num_threads; tid++) {
        threads.emplace_back(std::bind([&](const unsigned begin, const unsigned end,
                                           const unsigned tid) {
            for(unsigned i = begin; i < end; i++) {
                for(unsigned j = 0; j < m_channel; j++) {
                    for(unsigned k = 0; k <  m_size; k++) {
                        unsigned index = i * m_size * m_channel + j * m_size + k;
                        m_output[index] += m_beta[j];
                        // m_output[(i *  m_channel + j) * m_size + k] += m_beta[j];
                    }
                }
            }
        }, tid * m_batch / num_threads,
           (tid + 1) * m_batch / num_threads, tid));
    } std::for_each(threads.begin(), threads.end(), [](std::thread& t) { t.join(); });
}

void batchnorm_mean_delta(unsigned num_threads,  float *m_delta, float *m_variance, float *m_mean_delta, 
                          unsigned m_channel, unsigned m_size, unsigned m_batch) {
    for(unsigned i = 0; i < m_channel; i++) {
        m_mean_delta[i] = 0;
        for(unsigned j = 0; j < m_batch; j++) {
            for(unsigned k = 0; k < m_size; k++) {
                m_mean_delta[i] += m_delta[j * m_channel * m_size + i * m_size + k];
            }
        }
        m_mean_delta[i] *= (-1.0/sqrt(m_variance[i] + 0.00001));
    }
}
void batchnorm_variance_delta(unsigned num_threads, float *m_x, float *m_delta, 
                              float *m_mean, float *m_variance, float *m_variance_delta, 
                              unsigned m_channel, unsigned m_size, unsigned m_batch) {
    for(unsigned i = 0; i < m_channel; i++) {
        m_variance_delta[i] = 0;
        for(unsigned j = 0; j < m_batch; j++) {
            for(unsigned k = 0; k < m_size; k++) {
                unsigned index = j * m_channel * m_size + i * m_size + k;
                m_variance_delta[i] += m_delta[index] * (m_x[index] - m_mean[i]);
            }
        }
        m_variance_delta[i] *= -0.5 * pow(m_variance[i] + 0.00001, (float)(-3.0/2.0));
    }
}

void batchnorm_normalize_delta(unsigned num_threads, float *m_x, float *m_mean, float *m_variance, 
                               float *m_mean_delta, float *m_variance_delta, float *m_delta, 
                               unsigned m_channel, unsigned m_size, unsigned m_batch) {

    for(unsigned i = 0; i < m_batch; i++) {
        for(unsigned j = 0; j < m_channel; j++) {
            for(unsigned k = 0; k < m_size; k++) {
                unsigned index = i * m_channel * m_size + j * m_size + k;
                m_delta[index] = m_delta[index] * 1.0/(sqrt(m_variance[j] + 0.00001)) + m_variance_delta[j] * 2.0 * (m_x[index] - m_mean[j]) / (m_size * m_batch) + m_mean_delta[j] / (m_size * m_batch);
            }
        }
    }
}

}
// End of namespace nebula.
