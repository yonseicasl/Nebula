#ifndef __CONCAT_LAYER_H__
#define __CONCAT_LAYER_H__

#include "layer.h"

namespace nebula {

class concat_layer_t : public layer_t {
public:
    concat_layer_t(network_t *m_netwokr, layer_t *m_prev_layer, layer_type_t m_layer_type);
    ~concat_layer_t();

    // Initialize layer.
    void init(section_config_t m_section_config);
    // Initialize weight from file.
    void init_weight(std::fstream &m_weight_file);
    // Initialize weight from scratch
    void init_weight();
    // Forward propagation.
    void forward();
    // Backward propagation.
    void backward();
    // Layer update
    void update();
    // Store weight.
    void store_weight(std::fstream &m_weight_file);

#ifdef GPU_ENABLED
    // Forward propagation
    void _forward_();
    // Backward propagation
    void _backward_();
    // Layer update
    void _update_();
#endif

private:
    std::vector<layer_t *> connections;
    std::vector<unsigned> hop;
    unsigned num_concats;
};

}


#endif
