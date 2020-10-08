#ifndef __CONVOLUTIONAL_H__
#define __CONVOLUTIONAL_H__

#include "network.h"
#include "box.h"

namespace nebula {

class detection_t : public network_t {
public:
    detection_t();
    ~detection_t();

    // Run network.
    void run(const std::string m_output_weight = "");

private:
    // Initialize network.
    void init_network(const std::string m_network_config);
    //Initialize input data.
    void init_data(const std::string m_data_config);
    // Load batch data.
    void load_data(const unsigned m_batch_index);
    // Print reulsts.
    void print_results();

    std::vector<std::string> inputs;                        // List of input data


    box_t *detection_box;
};

}
// End of namespace nebula.

#endif
