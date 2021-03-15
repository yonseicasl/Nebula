#ifndef __LAYER_H__
#define __LAYER_H__

#include <fstream>
#include "config.h"
#include "def.h"
#include "network.h"

namespace nebula {

class layer_t {
public:
    layer_t(network_t *m_network, layer_t *m_prev_layer,
            layer_type_t m_layer_type = UNDEFINED_LAYER);
    virtual ~layer_t();

    // Initialize layer.
    virtual void init(section_config_t m_section_config) = 0;
    // Initialize weight from file.
    virtual void init_weight(std::fstream &m_weight_file) = 0;
    // Initialize weight from scratch.
    virtual void init_weight() = 0;
    // Forward propagation
    virtual void forward() = 0;
    // Backward propagation
    virtual void backward() = 0;
    // Layer update
    virtual void update() = 0;
    // Store weight.
    virtual void store_weight(std::fstream &m_weight_file) = 0;


    layer_type_t layer_type;            // Layer type
    activation_type_t activation_type;  // Activation type

    unsigned input_height;              // Input height
    unsigned input_width;               // Input width
    unsigned input_channel;             // Input channel
    unsigned input_size;                // Input data size

    unsigned output_height;             // Output height
    unsigned output_width;              // Output width
    unsigned output_channel;            // Output channel
    unsigned output_size;               // Output data size

    float *input_data;                  // Input data.
    float *weight;                      // Weight 
    float *output_data;                 // Output data
    float *delta;                       // Delta to update layers

    unsigned padding;                   // Padding size

    unsigned num_filters;               // Number of filters (channels)
    unsigned filter_size;               // Filter size
    unsigned stride;                    // Filter striding distance
	
	unsigned group;					    // Group Convolution
   
    layer_t     *prev_layer;            // Pointer to previous layer
    layer_t     *next_layer;            // Pointer to next layer

    network_t   *network;               // Pointer to main network
protected:
    // Activation function
    void activate();
    // Gradient function
    void gradient();

    unsigned    num_threads;            // Number of threads
};

}
// End of namespace nebula.
#endif
