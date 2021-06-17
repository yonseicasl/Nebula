#ifndef __ENCODER_H__
#define __ENCODER_H__

#include "layer.h"


namespace nebula {


class encoder_t : public layer_t {

public:

    encoder_t(network_t *m_netwokr, layer_t *m_prev_layer, layer_type_t m_layer_type);
    ~encoder_t();


    // Initialize the layer
    void init(section_config_t m_section_config);
    // Initialize the weight from the file
    void init_wegiht(std::fstream &m_input_weight);
    // Initialize the weight from the scratch
    void init_weight();
    // Store the weight to the file
    void store_weight(std::fstream &m_output_weight);


};

}
// End of namespace nebula

#endif
