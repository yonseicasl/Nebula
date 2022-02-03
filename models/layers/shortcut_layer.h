#ifndef __SHORTCUT_LAYER_H__
#define __SHORTCUT_LAYER_H__

#include "layer.h"

namespace nebula {

class shortcut_layer_t : public layer_t {
public:
    shortcut_layer_t(network_t *m_network, layer_t *m_prev_layer, layer_type_t m_layer_type);
    ~shortcut_layer_t();

    // Initialize layer.
    void init(section_config_t m_section_config);
    // Initialize weight from the file.
    void init_weight(std::fstream &m_weight_file);
    // Initialize weight from scratch.
    void init_weight();
    // Forward propagation
    void forward();
    // Backward propagation
    void backward();
    void update();
    // Store weight.
    void store_weight(std::fstream &m_weight_file);

private:
    layer_t *connection;        // Shortcut connection between layers.
};

}
// End of namespace nebula.
#endif

