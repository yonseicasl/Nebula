#ifndef __ACTIVATIONS_H__
#define __ACTIVATIONS_H__

namespace nebula {

// Activation function
void elu_activation(float *m_output, unsigned m_size);
void hardtan_activation(float *m_output, unsigned m_size);
void leaky_activation(float *m_output, unsigned m_size);
void lhtan_activation(float *m_output, unsigned m_size);
void linear_activation(float *m_output, unsigned m_size);
void loggy_activation(float *m_output, unsigned m_size);
void logistic_activation(float *m_output, unsigned m_size);
void plse_activation(float *m_output, unsigned m_size);
void ramp_activation(float *m_output, unsigned m_size);
void relie_activation(float *m_output, unsigned m_size);
void relu_activation(float *m_output, unsigned m_size);
void stair_activation(float *m_output, unsigned m_size);
void tanh_activation(float *m_output, unsigned m_size);

// Gradient function
void elu_gradient(float *m_delta, float *m_output, unsigned m_size);
void hardtan_gradient(float *m_delta, float *m_output, unsigned m_size);
void leaky_gradient(float *m_delta, float *m_output, unsigned m_size);
void lhtan_gradient(float *m_delta, float *m_output, unsigned m_size);
void linear_gradient(float *m_delta, float *m_output, unsigned m_size);
void loggy_gradient(float *m_delta, float *m_output, unsigned m_size);
void logistic_gradient(float *m_delta, float *m_output, unsigned m_size);
void plse_gradient(float *m_delta, float *m_output, unsigned m_size);
void ramp_gradient(float *m_delta, float *m_output, unsigned m_size);
void relie_gradient(float *m_delta, float *m_output, unsigned m_size);
void relu_gradient(float *m_delta, float *m_output, unsigned m_size);
void stair_gradient(float *m_delta, float *m_output, unsigned m_size);
void tanh_gradient(float *m_delta, float *m_output, unsigned m_size);

#ifdef GPU_ENABLED

void _elu_activation_(float *m_output_dev, unsigned m_size);
void _hardtan_activation_(float *m_output_dev, unsigned m_size);
void _leaky_activation_(float *m_output_dev, unsigned m_size);
void _lhtan_activation_(float *m_output_dev, unsigned m_size);
void _linear_activation_(float *m_output_dev, unsigned m_size);
void _loggy_activation_(float *m_output_dev, unsigned m_size);
void _logistic_activation_(float *m_output_dev, unsigned m_size);
void _plse_activation_(float *m_output_dev, unsigned m_size);
void _ramp_activation_(float *m_output_dev, unsigned m_size);
void _relie_activation_(float *m_output_dev, unsigned m_size);
void _relu_activation_(float *m_output_dev, unsigned m_size);
void _stair_activation_(float *m_output_dev, unsigned m_size);
void _tanh_activation_(float *m_output_dev, unsigned m_size);

void _elu_gradient_(float *m_output, float *m_delta, unsigned m_size);
void _hardtan_gradient_(float *m_output, float *m_delta, unsigned m_size);
void _leaky_gradient_(float *m_output, float *m_delta, unsigned m_size);
void _lhtan_gradient_(float *m_output, float *m_delta, unsigned m_size);
void _linear_gradient_(float *m_output, float *m_delta, unsigned m_size);
void _loggy_gradient_(float *m_output, float *m_delta, unsigned m_size);
void _logistic_gradient_(float *m_output, float *m_delta, unsigned m_size);
void _plse_gradient_(float *m_output, float *m_delta, unsigned m_size);
void _ramp_gradient_(float *m_output, float *m_delta, unsigned m_size);
void _relie_gradient_(float *m_output, float *m_delta, unsigned m_size);
void _relu_gradient_(float *m_output, float *m_delta, unsigned m_size);
void _stair_gradient_(float *m_output, float *m_delta, unsigned m_size);
void _tanh_gradient_(float *m_output, float *m_delta, unsigned m_size);
#endif

}
// End of namespace nebula.


#endif

