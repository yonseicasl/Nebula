#include "activations.h"
#include <cmath>
void elu_activation(float *m_output, unsigned m_size){
    for(unsigned i =0; i < m_size; i++){
        m_output[i] = m_output[i] > 0.0 ?
                      m_output[i] * m_output[i] :
                      m_output[i] * (exp(m_output[i]  ) - 1.0);
    }
}

void hardtan_activation(float *m_output, unsigned m_size) {
    for(unsigned i = 0; i < m_size; i++) {
        if((m_output[i] < 1.0) && (m_output[i] > -1.0)) {m_output[i] = 1.0;}
        else {m_output[i] = 0.0;}
    }
}

void leaky_activation(float *m_output, unsigned m_size){
    for(unsigned i = 0; i < m_size; i++) {
        m_output[i] = m_output[i] > 0.0 ? m_output[i] : 0.1 * m_output[i];
    }
}
void lhtan_activation(float *m_output, unsigned m_size){
    for(unsigned i = 0; i < m_size; i++) {
        if((m_output[i] > 0.0) && (m_output[i] < 1.0)) { m_output[i] = 1.0; }
        else { m_output[i] = 0.0; }
    }
}
void linear_activation(float *m_output, unsigned m_size){
}

void loggy_activation(float *m_output, unsigned m_size){
    for(unsigned i = 0; i < m_size; i++) {
        m_output[i] = 2.0 / (1.0 + exp(0.0 - m_output[i])) - 1.0; 
    }
}
void logistic_activation(float *m_output, unsigned m_size){
    for(unsigned i = 0; i < m_size; i++) {
        m_output[i] = 1.0 / (1.0 + exp(0.0 - m_output[i])); 
    }
}
void plse_activation(float *m_output, unsigned m_size){
    for(unsigned i = 0; i < m_size; i++) {
        if(m_output[i] < -4.0) {
            m_output[i] = 0.01 * (m_output[i] + 4.0);
        }
        else if(m_output[i] > 4.0) { 
            m_output[i] = 0.01 * (m_output[i]- 4.0) + 1.0;
        }
        else { 
            m_output[i] = 0.125 * m_output[i] + 0.5;
        }
    }
}
void ramp_activation(float *m_output, unsigned m_size){
    for(unsigned i = 0; i < m_size; i++) {
        m_output[i] = m_output[i] > 0.0 ?
            m_output[i] * (m_output[i] + 0.01) :
            0.01 * m_output[i];
    }
}
void relie_activation(float *m_output, unsigned m_size){
    for(unsigned i = 0; i < m_size; i++) {
        m_output[i] = m_output[i] > 0.0 ? m_output[i] : 0.01 * m_output[i];
    }
}
void relu_activation(float *m_output, unsigned m_size){
    for(unsigned i = 0; i < m_size; i++) {
        m_output[i] = m_output[i] > 0.0 ? m_output[i] : 0.0 ; 
    }
}
void stair_activation(float *m_output, unsigned m_size){
    for(unsigned i = 0; i < m_size; i++) {
        if(int(m_output[i]) % 2) {
            m_output[i] = m_output[i] - float(int(m_output[i])) +
                float(int(m_output[i] / 2.0));
        }
        else { m_output[i] = float(int(m_output[i] / 2.0)); }
    }
}
void tanh_activation(float *m_output, unsigned m_size){
    for(unsigned i = 0; i < m_size; i++) {
        m_output[i] = (exp(2.0 * m_output[i]) - 1.0) /
            (exp(2.0 * m_output[i]) + 1.0); 
    }
}

void elu_gradient(float *m_delta, float *m_output, unsigned m_size){
    for(unsigned i = 0; i < m_size; i++) {
        m_delta[i] *= float(m_output[i] >= 0.0) +
            float(m_output[i] < 0.0) * (m_output[i] + 1.0); 
    }
}
void hardtan_gradient(float *m_delta, float *m_output, unsigned m_size){
    for(unsigned i = 0; i < m_size; i++) {
        m_delta[i] *= float((m_output[i] > - 1.0) && (m_output[i] < 1.0)); 
    }
}
void leaky_gradient(float *m_delta, float *m_output, unsigned m_size){
    for(unsigned i = 0; i < m_size; i++) {
        m_delta[i] *= m_output[i] > 0.0 ? 1.0 : 0.1; 
    }
}
void lhtan_gradient(float *m_delta, float *m_output, unsigned m_size){
    for(unsigned i = 0; i < m_size; i++) {
        if((m_output[i] <= 0.0) || (m_output[i] >= 1.0)) {
            m_delta[i] *= 0.001;
        }
    }
}
void linear_gradient(float *m_delta, float *m_output, unsigned m_size){
}
void loggy_gradient(float *m_delta, float *m_output, unsigned m_size){
    for(unsigned i = 0; i < m_size; i++) {
        m_delta[i] *= 2.0 * (1.0 - (m_output[i] + 1.0) / 2.0) *
            ((m_output[i] + 1.0) / 2.0); 
    }
}
void logistic_gradient(float *m_delta, float *m_output, unsigned m_size){
    for(unsigned i = 0; i < m_size; i++) {
        m_delta[i] *= (1.0 - m_output[i]) * m_output[i]; 
    }
}
void plse_gradient(float *m_delta, float *m_output, unsigned m_size){
    for(unsigned i = 0; i < m_size; i++) {
        m_delta[i] *= (m_output[i] < 0.0) || (m_output[i] > 1.0) ? 0.01 : 0.125; 
    }
}
void ramp_gradient(float *m_delta, float *m_output, unsigned m_size){
    for(unsigned i = 0; i < m_size; i++) {
        m_delta[i] *= float(m_output[i] > 0.0) + 0.1;
    }
}
void relie_gradient(float *m_delta, float *m_output, unsigned m_size){
    for(unsigned i = 0; i < m_size; i++) {
        m_delta[i] *= float(m_output[i] > 0.0 ? 1.0 : 0.01);
    }
}
void relu_gradient(float *m_delta, float *m_output, unsigned m_size){
    for(unsigned i = 0; i < m_size; i++) {
        m_delta[i] *= float(m_output[i] > 0.0); 
    }
}
void stair_gradient(float *m_delta, float *m_output, unsigned m_size){
    for(unsigned i = 0; i < m_size; i++) {
        m_delta[i] *= float(float(int(m_output[i])) != m_output[i]); 
    }
}
void tanh_gradient(float *m_delta, float *m_output, unsigned m_size){
    for(unsigned i = 0; i < m_size; i++) {
        m_delta[i] *= 1.0 - m_output[i] * m_output[i];
    }
}
