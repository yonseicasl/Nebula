#ifndef __RBM_LAYER_H__
#define __RBM_LAYER_H__

#include "layer.h"

#ifdef GPU_ENABLED
#include <curand.h>
#endif

namespace nebula {

class rbm_layer_t : public layer_t {
public:
    rbm_layer_t(network_t *m_network, layer_t *m_prev_layer, layer_type_t m_layer_type);
    ~rbm_layer_t();

    // Initialize layer.
    void init(section_config_t m_section_config);
    // Initialize weight from file.
    void init_weight(std::fstream &m_input_weight);
    // Initialize weight from scratch.
    void init_weight();
    // Sample hidden units using visible units value
    void sample_hidden_units(unsigned m_step);
    // Sample visible units using hidden units value
    void sample_visible_units();
    // Reconstruct the visible units and pretrain weight.
    void pretrain();
    // Forward propagation.
    void forward();
    // Backward propagation.
    void backward();
    // update layer's parameters.
    void update();
    // Store weight.
    void store_weight(std::fstream &m_weight_file);


#ifdef GPU_ENABLED
    // Forward propagation.
    void _forward_();
    // Backward propagation.
    void _backward_();
    // update layer's parameters.
    void _update_();

    // Sample hidden units using visible units value
    void _sample_hidden_units_(unsigned m_step);
    // Sample visible units using hidden units value
    void _sample_visible_units_();
    // Reconstruct the visible units and pretrain weight.
    void _pretrain_();
#endif

private:
#ifdef QUANTIZATION
    float step_size;
#endif
    float *bias;                    // Bias.
    float * weight;                 // Weight
    float * weight_update;          // Weight update
    unsigned weight_size;           // Weight size
    
    float *hidden_units;            // Hidden units
    float *hidden_mean_zero_step;    
    float *hidden_mean_k_step;
    
    float *visible_units_zero_step; // Visible units value for fist step
    float *visible_units_k_step;    // Visible units value for k step
    float *visible_mean;

    float * visible_bias;           // Bias for visible units
    float * hidden_bias;            // Bias for hidden units
    float * visible_bias_update;    // Bias update of visible units
    float * hidden_bias_update;     // Bias update of hidden units

    unsigned k_step;                // k value for k-step contrastive divergence learning

#ifdef GPU_ENABLED
    float * weight_dev;
    float * weight_update_dev;
    
    float *hidden_units_dev;
    float *hidden_mean_zero_step_dev;
    float *hidden_mean_k_step_dev;
    
    float *visible_units_zero_step_dev;
    float *visible_units_k_step_dev;
    float *visible_mean_dev;

    float * visible_bias_dev;
    float * hidden_bias_dev;
    float * visible_bias_update_dev;
    float * hidden_bias_update_dev;
#endif

};

}
// End of namespace nebula.
#endif

