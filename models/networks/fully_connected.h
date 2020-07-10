#ifndef __FULLY_CONNECTED_H__
#define __FULLY_CONNECTED_H__

#include "network.h"

namespace nebula {

class fully_connected_t : public network_t {
public:
    fully_connected_t();
    ~fully_connected_t();

    // Run network.
    void run(const std::string m_output_weight = "");

private:
    unsigned num_rbm_layers;
    unsigned num_pretrain_iterations;


    // Initialize network.
    void init_network(const std::string m_network_config);
    // Initialize input data.
    void init_data(const std::string m_data_config);
    // Pretrain the network.
    void pretrain(unsigned m_layer_index);
    // Load batch data.
    void load_data(const unsigned m_batch_index);
    // Print reulsts.
    void print_results();

    std::vector<std::string> inputs;        // List of input data
};

}
// End of namespace nebula.
#endif

