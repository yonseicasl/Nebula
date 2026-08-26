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
void sigmoid_activation(float *m_output, unsigned m_size);
void hsigmoid_activation(float *m_output, unsigned m_size);
void hswish_activation(float *m_output, unsigned m_size);
// Transformer/LLM activations (NPUsim SFU plan Phase 6). GELU uses the tanh
// approximation (Hendrycks & Gimpel); SiLU (a.k.a. swish) is x * sigmoid(x).
void gelu_activation(float *m_output, unsigned m_size);
void silu_activation(float *m_output, unsigned m_size);

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
// Forward/inference only: the gradient() interface receives the POST-activation output
// alone, from which GELU/SiLU derivatives cannot be reconstructed. These abort with a
// clear message instead of silently corrupting a training run.
void gelu_gradient(float *m_delta, float *m_output, unsigned m_size);
void silu_gradient(float *m_delta, float *m_output, unsigned m_size);

}
// End of namespace nebula.


#endif

