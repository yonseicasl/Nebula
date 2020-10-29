#ifndef __POOLING_LAYER_H__
#define __POOLING_LAYER_H__

#include "layer.h"

namespace nebula {

class pooling_layer_t : public layer_t {
public:
    pooling_layer_t(network_t *m_network, layer_t *m_prev_layer, layer_type_t m_layer_type);
    ~pooling_layer_t();
    
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
    unsigned *index;
#ifdef GPU_ENABLED
    unsigned *index_dev;
#ifdef CUDNN_ENABLED
    cudnnTensorDescriptor_t   input_descriptor;       // Descriptor for input data.
    cudnnTensorDescriptor_t   output_descriptor;      // Descriptor for output data.
    cudnnPoolingDescriptor_t  pooling_descriptor;     // Descriptor for pooling operation.
    cudnnPoolingMode_t        pooling_mode;           // Pooling mode whether max or average.

#endif
#endif

};

}
// End of namespace nebula.

#endif
