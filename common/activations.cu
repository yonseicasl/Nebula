extern "C++"{
#include "activations.h"
#include "def.h"
}

namespace nebula {

__global__ void _elu_activate_(float *m_output, unsigned m_total_size) {
    size_t i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if(i >= m_total_size) { return; }
    m_output[i] = m_output[i] > 0.0 ?
                       m_output[i] * m_output[i] :
                       m_output[i] * (exp(m_output[i]) - 1.0); 
}

__global__ void _hardtan_activate_(float *m_output, unsigned m_total_size) {
    size_t i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if(i >= m_total_size) { return; }
    if((m_output[i] < 1.0) && (m_output[i] > -1.0)) { m_output[i] = 1.0; }
    else { m_output[i] = 0.0; }
}

__global__ void _leaky_activate_(float *m_output, unsigned m_total_size) {
    size_t i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if(i >= m_total_size) { return; }
    m_output[i] = m_output[i] > 0.0 ? m_output[i] : 0.1 * m_output[i];
}

__global__ void _lhtan_activate_(float *m_output, unsigned m_total_size) {
    size_t i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if(i >= m_total_size) { return; }
    if((m_output[i] > 0.0) && (m_output[i] < 1.0)) { m_output[i] = 1.0; }
    else { m_output[i] = 0.0; }
}

__global__ void _loggy_activate_(float *m_output, unsigned m_total_size) {
    size_t i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if(i >= m_total_size) { return; }
    m_output[i] = 2.0 / (1.0 + exp(0.0 - m_output[i])) - 1.0; 
}

__global__ void _logistic_activate_(float *m_output, unsigned m_total_size) {
    size_t i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if(i >= m_total_size) { return; }
    m_output[i] = 1.0 / (1.0 + exp(0.0 - m_output[i])); 
}

__global__ void _plse_activate_(float *m_output, unsigned m_total_size) {
    size_t i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if(i >= m_total_size) { return; }
    if(m_output[i] < -4.0) { m_output[i] = 0.01 * (m_output[i] + 4.0); }
    else if(m_output[i] > 4.0) { m_output[i] = 0.01 * (m_output[i]- 4.0) + 1.0; }
    else { m_output[i] = 0.125 * m_output[i] + 0.5; }
}

__global__ void _ramp_activate_(float *m_output, unsigned m_total_size) {
    size_t i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if(i >= m_total_size) { return; }
    m_output[i] = m_output[i] > 0.0 ? m_output[i] * (m_output[i] + 0.01) :
                                                0.01 * m_output[i];
}

__global__ void _relie_activate_(float *m_output, unsigned m_total_size) {
    size_t i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if(i >= m_total_size) { return; }
    m_output[i] = m_output[i] > 0.0 ? m_output[i] : 0.01 * m_output[i];
}

__global__ void _relu_activate_(float *m_output, unsigned m_total_size) {
    size_t i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if(i >= m_total_size) { return; }
    m_output[i] = m_output[i] > 0.0 ? m_output[i] : 0.0 ; 
}

__global__ void _stair_activate_(float *m_output, unsigned m_total_size) {
    size_t i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if(i >= m_total_size) { return; }
    if(int(m_output[i]) % 2) {
        m_output[i] = m_output[i] - float(int(m_output[i])) +
                           float(int(m_output[i]/2.0));
    }
    else { m_output[i] = float(int(m_output[i]/2.0)); }
}

__global__ void _tanh_activate_(float *m_output, unsigned m_total_size) {
    size_t i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if(i >= m_total_size) { return; }
    m_output[i] = (2.0/ (1 + exp(-2.0 * m_output[i])) - 1);
}


extern "C++" void _elu_activation_(float *m_output, unsigned m_size){
    dim3 cuda_griddim = {(m_size - 1) / BLOCK_SIZE + 1, 1, 1}; 
    _elu_activate_<<<cuda_griddim, BLOCK_SIZE>>>(m_output, m_size);
}

extern "C++" void _hardtan_activation_(float *m_output, unsigned m_size){
    dim3 cuda_griddim = {(m_size - 1) / BLOCK_SIZE + 1, 1, 1}; 
    _hardtan_activate_<<<cuda_griddim, BLOCK_SIZE>>>(m_output, m_size);
}
extern "C++" void _leaky_activation_(float *m_output, unsigned m_size){
    dim3 cuda_griddim = {(m_size - 1) / BLOCK_SIZE + 1, 1, 1}; 
    _leaky_activate_<<<cuda_griddim, BLOCK_SIZE>>>(m_output, m_size);
}
extern "C++" void _lhtan_activation_(float *m_output, unsigned m_size){
    dim3 cuda_griddim = {(m_size - 1) / BLOCK_SIZE + 1, 1, 1}; 
    _lhtan_activate_<<<cuda_griddim, BLOCK_SIZE>>>(m_output, m_size);
}
extern "C++" void _linear_activation_(float *m_output, unsigned m_size){}
extern "C++" void _loggy_activation_(float *m_output, unsigned m_size){
    dim3 cuda_griddim = {(m_size - 1) / BLOCK_SIZE + 1, 1, 1}; 
    _loggy_activate_<<<cuda_griddim, BLOCK_SIZE>>>(m_output, m_size);
}
extern "C++" void _logistic_activation_(float *m_output, unsigned m_size){
    dim3 cuda_griddim = {(m_size - 1) / BLOCK_SIZE + 1, 1, 1}; 
    _logistic_activate_<<<cuda_griddim, BLOCK_SIZE>>>(m_output, m_size);
}
extern "C++" void _plse_activation_(float *m_output, unsigned m_size){
    dim3 cuda_griddim = {(m_size - 1) / BLOCK_SIZE + 1, 1, 1}; 
    _plse_activate_<<<cuda_griddim, BLOCK_SIZE>>>(m_output, m_size);
}
extern "C++" void _ramp_activation_(float *m_output, unsigned m_size){
    dim3 cuda_griddim = {(m_size - 1) / BLOCK_SIZE + 1, 1, 1}; 
    _ramp_activate_<<<cuda_griddim, BLOCK_SIZE>>>(m_output, m_size);
}
extern "C++" void _relie_activation_(float *m_output, unsigned m_size){
    dim3 cuda_griddim = {(m_size - 1) / BLOCK_SIZE + 1, 1, 1}; 
    _relie_activate_<<<cuda_griddim, BLOCK_SIZE>>>(m_output, m_size);
}
extern "C++" void _relu_activation_(float *m_output, unsigned m_size){
    dim3 cuda_griddim = {(m_size - 1) / BLOCK_SIZE + 1, 1, 1}; 
    _relu_activate_<<<cuda_griddim, BLOCK_SIZE>>>(m_output, m_size);
}
extern "C++" void _stair_activation_(float *m_output, unsigned m_size){
    dim3 cuda_griddim = {(m_size - 1) / BLOCK_SIZE + 1, 1, 1}; 
    _stair_activate_<<<cuda_griddim, BLOCK_SIZE>>>(m_output, m_size);
}
extern "C++" void _tanh_activation_(float *m_output, unsigned m_size){
    dim3 cuda_griddim = {(m_size - 1) / BLOCK_SIZE + 1, 1, 1}; 
    _tanh_activate_<<<cuda_griddim, BLOCK_SIZE>>>(m_output, m_size);
}

__global__ void _elu_grad_(float *m_output, float *m_delta, unsigned m_total_size) {
    size_t i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if(i >= m_total_size) { return; }
    m_delta[i] *= float(m_output[i] >= 0.0) +
                  float(m_output[i] < 0.0) * (m_output[i] + 1.0); 
}

__global__ void _hardtan_grad_(float *m_output, float *m_delta, unsigned m_total_size) {
    size_t i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if(i >= m_total_size) { return; }
    m_delta[i] *= float((m_output[i] > - 1.0) && (m_output[i] < 1.0)); 
}

__global__ void _leaky_grad_(float *m_output, float *m_delta, unsigned m_total_size) {
    size_t i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if(i >= m_total_size) { return; }
    m_delta[i] *= m_output[i] > 0.0 ? 1.0 : 0.1; 
}

__global__ void _lhtan_grad_(float *m_output, float *m_delta, unsigned m_total_size) {
    size_t i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if(i >= m_total_size) { return; }
    if((m_output[i] <= 0.0) || (m_output[i] >= 1.0)) { m_delta[i] *= 0.001; }
}

__global__ void _loggy_grad_(float *m_output, float *m_delta, unsigned m_total_size) {
    size_t i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if(i >= m_total_size) { return; }
    m_delta[i] *= 2.0 * (1.0 - (m_output[i] + 1.0) / 2.0) * ((m_output[i] + 1.0) / 2.0); 
}

__global__ void _logistic_grad_(float *m_output, float *m_delta, unsigned m_total_size) {
    size_t i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if(i >= m_total_size) { return; }
    m_delta[i] *= (1.0 - m_output[i]) * m_output[i]; 
}

__global__ void _plse_grad_(float *m_output, float *m_delta, unsigned m_total_size) {
    size_t i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if(i >= m_total_size) { return; }
    m_delta[i] *= (m_output[i] < 0.0) || (m_output[i] > 1.0) ? 0.01 : 0.125; 
}

__global__ void _ramp_grad_(float *m_output, float *m_delta, unsigned m_total_size) {
    size_t i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if(i >= m_total_size) { return; }
    m_delta[i] *= float(m_output[i] > 0.0) + 0.1;
}

__global__ void _relie_grad_(float *m_output, float *m_delta, unsigned m_total_size) {
    size_t i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if(i >= m_total_size) { return; }
    m_delta[i] *= float(m_output[i] > 0.0 ? 1.0 : 0.01);
}

__global__ void _relu_grad_(float *m_output, float *m_delta, unsigned m_total_size) {
    size_t i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if(i >= m_total_size) { return; }
    m_delta[i] *= float(m_output[i] > 0.0); 
}

__global__ void _stair_grad_(float *m_output, float *m_delta, unsigned m_total_size) {
    size_t i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if(i >= m_total_size) { return; }
    m_delta[i] *= float(float(int(m_output[i])) != m_output[i]); 
}

__global__ void _tanh_grad_(float *m_output, float *m_delta, unsigned m_total_size) {
    size_t i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if(i >= m_total_size) { return; }
    m_delta[i] *= 1.0 - m_output[i] * m_output[i];
}
extern "C++" void _elu_gradient_(float *m_output, float *m_delta, unsigned m_size){
    dim3 cuda_griddim = {(m_size - 1) / BLOCK_SIZE + 1, 1, 1}; 
    _elu_grad_<<<cuda_griddim, BLOCK_SIZE>>>(m_output, m_delta, m_size);
}
extern "C++" void _hardtan_gradient_(float *m_output, float *m_delta, unsigned m_size){
    dim3 cuda_griddim = {(m_size - 1) / BLOCK_SIZE + 1, 1, 1}; 
    _hardtan_grad_<<<cuda_griddim, BLOCK_SIZE>>>(m_output, m_delta, m_size);
}
extern "C++" void _leaky_gradient_(float *m_output, float *m_delta, unsigned m_size){
    dim3 cuda_griddim = {(m_size - 1) / BLOCK_SIZE + 1, 1, 1}; 
    _leaky_grad_<<<cuda_griddim, BLOCK_SIZE>>>(m_output, m_delta, m_size);
}
extern "C++" void _lhtan_gradient_(float *m_output, float *m_delta, unsigned m_size){
    dim3 cuda_griddim = {(m_size - 1) / BLOCK_SIZE + 1, 1, 1}; 
    _lhtan_grad_<<<cuda_griddim, BLOCK_SIZE>>>(m_output, m_delta, m_size);
}
extern "C++" void _loggy_gradient_(float *m_output, float *m_delta, unsigned m_size){
    dim3 cuda_griddim = {(m_size - 1) / BLOCK_SIZE + 1, 1, 1}; 
    _loggy_grad_<<<cuda_griddim, BLOCK_SIZE>>>(m_output, m_delta, m_size);
}
extern "C++" void _logistic_gradient_(float *m_output, float *m_delta, unsigned m_size){
    dim3 cuda_griddim = {(m_size - 1) / BLOCK_SIZE + 1, 1, 1}; 
    _logistic_grad_<<<cuda_griddim, BLOCK_SIZE>>>(m_output, m_delta, m_size);
}
extern "C++" void _plse_gradient_(float *m_output, float *m_delta, unsigned m_size){
    dim3 cuda_griddim = {(m_size - 1) / BLOCK_SIZE + 1, 1, 1}; 
    _plse_grad_<<<cuda_griddim, BLOCK_SIZE>>>(m_output, m_delta, m_size);
}

extern "C++" void _ramp_gradient_(float *m_output, float *m_delta, unsigned m_size){
    dim3 cuda_griddim = {(m_size - 1) / BLOCK_SIZE + 1, 1, 1}; 
    _ramp_grad_<<<cuda_griddim, BLOCK_SIZE>>>(m_output, m_delta, m_size);
}
extern "C++" void _relie_gradient_(float *m_output, float *m_delta, unsigned m_size){
    dim3 cuda_griddim = {(m_size - 1) / BLOCK_SIZE + 1, 1, 1}; 
    _relie_grad_<<<cuda_griddim, BLOCK_SIZE>>>(m_output, m_delta, m_size);
}
extern "C++" void _relu_gradient_(float *m_output, float *m_delta, unsigned m_size){
    dim3 cuda_griddim = {(m_size - 1) / BLOCK_SIZE + 1, 1, 1}; 
    _relu_grad_<<<cuda_griddim, BLOCK_SIZE>>>(m_output, m_delta, m_size);
}
extern "C++" void _stair_gradient_(float *m_output, float *m_delta, unsigned m_size){
    dim3 cuda_griddim = {(m_size - 1) / BLOCK_SIZE + 1, 1, 1}; 
    _stair_grad_<<<cuda_griddim, BLOCK_SIZE>>>(m_output, m_delta, m_size);
}
extern "C++" void _tanh_gradient_(float *m_output, float *m_delta, unsigned m_size){
    dim3 cuda_griddim = {(m_size - 1) / BLOCK_SIZE + 1, 1, 1}; 
    _tanh_grad_<<<cuda_griddim, BLOCK_SIZE>>>(m_output, m_delta, m_size);
}

}
// End of namespace nebula
