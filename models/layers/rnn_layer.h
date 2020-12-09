#ifndef __RNN_LAYER_H__
#define __RNN_LAYER_H__

#include "layer.h"
#include "connected_layer.h"

namespace nebula {

class rnn_layer_t : public layer_t {
public:
    rnn_layer_t(network_t *m_network, layer_t *m_prev_layer, layer_type_t m_layer_type);
    ~rnn_layer_t();

    // Initialize layer.
    void init(section_config_t m_section_config);
    // Initialize weight from file.
    void init_weight(std::fstream &m_input_weight);
    // Initialize weight from scratch.
    void init_weight();
    // Store weight.
    void store_weight(std::fstream &m_weight_file);
    // Forward propagation.
    void forward();
    // Backward propagation.
    void backward();
    // update layer's parameters.
    void update();

private:
    float *state;
    float *prev_state;
    bool batch_normalize; 
    connected_layer_t *input_gate;
    connected_layer_t *hidden_gate;
    connected_layer_t *output_gate;
};

}
// End of namespace nebula.
#endif
