#ifndef __CONVOLUTIONAL_LAYER_H__
#define __CONVOLUTIONAL_LAYER_H__

#include "layer.h"

namespace nebula {

class convolutional_layer_t : public layer_t {
public:
    convolutional_layer_t(network_t *m_network, layer_t *m_prev_layer, layer_type_t m_layer_type);
    ~convolutional_layer_t();

    // Initialize layer.
    void init(section_config_t m_section_config);
    // Initialize weight from file.
    void init_weight(std::fstream &m_weight_file);
    // Initialize weight from scratch.
    void init_weight();
    // Store weight.
    void store_weight(std::fstream &m_weight_file);
    // Forward propagation
    void forward();
    // Backward propagation
    void backward();
    // Layer update
    void update();
    // Forward batch normalization.
    void forward_batchnorm();
    // Backward batch normalization.
    void backward_batchnorm();

private:
    float *workspace;               // Workspace
    size_t workspace_size;          // workspace size

    float *bias;
    float *bias_update;

    float *weight_update;           // Weight update
    unsigned weight_size;           // Weight size

    bool batch_normalize;           // Use batch normalize or not.
    float *scale;                   // scale.
    float *scale_update;

    float *normalize_mean;
    float *rolling_mean;            // Renaming
    float *mean_delta;

    float *normalize_variance;
    float *rolling_variance;        // Renaming
    float *variance_delta;

    float *x;                       // Renaming
    float *normalize_x;             // Renaming

};

}
// End of namespace nebula

#endif

