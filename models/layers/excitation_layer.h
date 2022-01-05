#ifndef __EXCITATION_LAYER_H__
#define __EXCITATION_LAYER_H__

#include "layer.h"

namespace nebula {

class excitation_layer_t : public layer_t {
public:
    excitation_layer_t(network_t *m_network, layer_t *m_prev_layer, layer_type_t m_layer_type);
    ~excitation_layer_t();

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

};

}
// End of namespace nebula

#endif

