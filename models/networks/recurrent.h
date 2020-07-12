#ifndef __RECURRENT_H__
#define __RECURRENT_H__

#include "network.h"

<<<<<<< HEAD
=======
namespace nebula {
>>>>>>> namespace

class recurrent_t : public network_t {
public:
    recurrent_t();
    ~recurrent_t();

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

    std::vector<std::string> inputs;        // All input words
    std::vector<std::string> labels;        // All label words
    std::vector<std::string>::iterator iter;
    std::vector<std::string> input_words;   // Batched input words.
    std::vector<std::string> label_words;   // Batched label words.

};

}
// End of namespace nebula.
#endif

