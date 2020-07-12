#include <algorithm>
#include <functional>
#include <random>
#include <thread>
#ifdef GPU_ENABLED
#include <cuda_runtime.h>
#endif
#include "dropout_layer.h"

namespace nebula {

dropout_layer_t::dropout_layer_t(network_t *m_network, layer_t *m_prev_layer, layer_type_t m_layer_type) :
    layer_t(m_network, m_prev_layer, m_layer_type), 
    probability(0.0),
    rand(NULL) {
#ifdef GPU_ENABLED
    rand_dev = NULL;
#endif
}

dropout_layer_t::~dropout_layer_t() {
    delete [] rand;
#ifdef GPU_ENABLED
    cudaFree(rand_dev);
    curandDestroyGenerator(generator);
#endif

}

// Initialize layer.
void dropout_layer_t::init(section_config_t m_section_config) {
    m_section_config.get_setting("probability", &probability);

    input_size = prev_layer ? prev_layer->output_size : network->input_size;
    output_size = input_size;

    output_data = prev_layer->output_data;
    delta = prev_layer->delta;

    rand = new float[input_size * network->batch_size]();
#ifdef GPU_ENABLED
    cudaMalloc((void**)&rand_dev, input_size * network->batch_size * sizeof(float));
    cudaMemset(rand_dev, 0, input_size * network->batch_size * sizeof(float));

    output_data_dev = prev_layer->output_data_dev;
    delta_dev   = prev_layer->delta_dev;

    curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
    curandGenerateUniform(generator, rand_dev, input_size * network->batch_size);
#endif
}

// Initialize weight from file.
void dropout_layer_t::init_weight(std::fstream &m_weight_file) {
    // Nothing to do
}

// Initialize weight from scratch.
void dropout_layer_t::init_weight() {
    // Nothing to do
}

// Forward propagation
void dropout_layer_t::forward() {
    float scale = 1.0 / (1.0 - probability);

    std::minstd_rand generator(std::random_device{}());
    std::uniform_real_distribution<float> distribution(0.0, 1.0);

// Pass the selected data to the next layer
// in training phase.
    if(network->run_type == TRAIN_RUN) {
        std::vector<std::thread> threads;
        threads.reserve(num_threads); 
        for(unsigned tid = 0; tid < num_threads; tid++) {
            threads.emplace_back(std::bind([&](const unsigned begin, const unsigned end,
                                          const unsigned tid) {
                for(unsigned i = begin; i < end; i++) {
                    rand[i] = distribution(generator);
                    if(rand[i] < probability) { output_data[i] = 0.0; }
                    else { output_data[i] = output_data[i] * scale; }
                }
            }, tid * input_size * network->batch_size / num_threads,
               (tid + 1) * input_size * network->batch_size / num_threads, tid));
        } std::for_each(threads.begin(), threads.end(), [](std::thread& t) { t.join(); });
    }
}

// Backward propagation
void dropout_layer_t::backward() {
    float scale = 1.0 / (1.0 - probability);
    float *prev_delta = prev_layer ? prev_layer->delta : NULL;

    if(prev_delta) {
        for(unsigned i = 0; i < input_size*network->batch_size; i++) {
            if(rand[i] < probability) { prev_delta[i] = 0.0; }
            else { prev_delta[i] *= scale; }
        }
    }
}

// Layer update
void dropout_layer_t::update() {
    // Nothing to do
}

// Store weight.
void dropout_layer_t::store_weight(std::fstream &m_weight_file) {
    // Nothing to do
}

}
// End of namespace nebula.
