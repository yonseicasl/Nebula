#ifndef __INCEPTION_LAYER_H__
#define __INCEPTION_LAYER_H__

#include "layer.h"

namespace nebula {

class inception_layer_t : public layer_t {
public:
    inception_layer_t(network_t *m_network, layer_t *m_prev_layer, layer_type_t m_layer_type);
    ~inception_layer_t();

    // Initialize layer.
    void init(section_config_t m_section_config);
    // Initialize weight from file.
    void init_weight(std::fstream &m_input_weight);
    // Initialize weight from scratch.
    void init_weight();
    // Store weight.
    void store_weight(std::fstream &m_output_weight);
    // Forward propagation.
    void forward();
    // Backward propagation.
    void backward();
    // Update layer's parameters.
    void update();
    // Forward batch normalization.
    void forward_batchnorm();
    // Backward batch normalization.
    void backward_batchnorm();

#ifdef GPU_ENABLED
    // Forward propagation.
    void _forward_();
    // Backward propagation.
    void _backward_();
    // Update layer's parameters.
    void _update_();
    // Forward batch normalization.
    void _forward_batchnorm_();
    // Backward batch normalization.
    void _backward_batchnorm_();
#endif

private:
    float *workspace;               // Workspace
    size_t workspace_size;          // workspace size

    float *bias;
    float *bias_update;

    float *weight;                  // Weight 
    float *weight_update;           // Weight update
    unsigned weight_size;           // Weight size
    unsigned hops;                  // hops 

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

    layer_t *inception;
#ifdef GPU_ENABLED
    float *bias_dev;                // Bias.
    float *bias_update_dev;         // Bias update.

    float *weight_dev;              // Weight 
    float *weight_update_dev;       // Weight update

    float *scale_dev;
    float *scale_update_dev;

    float *normalize_mean_dev;
    float *rolling_mean_dev;
    float *mean_delta_dev;

    float *normalize_variance_dev;
    float *rolling_variance_dev;
    float *variance_delta_dev;

    float *x_dev;
    float *normalize_x_dev;

    float *workspace_dev;           // Workspace
#endif
};

}


#endif
