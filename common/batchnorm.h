#ifndef __BATCHNORM_H__
#define __BATCHNORM_H__

namespace nebula {

void batchnorm_mean(unsigned num_threads, float *m_output, float *m_mean, 
                    unsigned m_channel, unsigned m_size, unsigned m_batch);
void batchnorm_variance(unsigned num_threads, float *m_output, float *m_mean, float *m_variance,
                        unsigned m_channel, unsigned m_size, unsigned m_batch);
void batchnorm_normalize(unsigned num_threads, float *m_output, float *m_mean, float *m_variance, 
                         unsigned m_channel, unsigned m_size, unsigned m_batch);
void batchnorm_scale_down(unsigned num_threads, float *m_output, float *m_scale, 
                          unsigned m_channel, unsigned m_size, unsigned m_batch);
void batchnorm_mean_delta(unsigned num_threads,  float *m_delta, float *m_variance, float *m_mean_delta, 
                          unsigned m_channel, unsigned m_size, unsigned m_batch);
void batchnorm_variance_delta(unsigned num_threads, float *m_x, float *m_delta, 
                              float *m_mean, float *m_variance, float *m_variance_delta, 
                              unsigned m_channel, unsigned m_size, unsigned m_batch);
void batchnorm_normalize_delta(unsigned num_threads, float *m_x, float *m_mean, float *m_variance, 
                               float *m_mean_delta,float *m_variance_delta, float *m_delta, 
                               unsigned m_channel, unsigned m_size, unsigned m_batch);

}
// End of namespace nebula

#endif

