#ifndef __NETWORK_H__
#define __NETWORK_H__

#include <vector>
#include "def.h"
#include "stopwatch.h"
#include "config.h"

namespace nebula {

class layer_t;

class network_t {
public:
    network_t();
    virtual ~network_t();

    // Forward propagation
    void forward();
    // Backward propagation
    void backward();
    // Network update
    void update();
    // Initialize network.
    void init(const std::string m_network_config);
    // Run network.
    virtual void run(const std::string m_output_weight = "") = 0;

    run_type_t run_type;                    // Network run type
    unsigned num_threads;                   // Number of CPU threads

    float learning_rate;                    // Learning rate (delta)
    float momentum;                         // Momentum
    float decay;                            // Decay
    float cost;                             // Cost in an iteration

    unsigned *reference_label;              // Correct labels
    unsigned input_height;                  // Input data height
    unsigned input_width;                   // Input data width
    unsigned input_channel;                 // Input data channel
    unsigned input_size;                    // Input data size
    unsigned batch_size;                    // Batch size

    unsigned time_step;                     // Time steps used for recurrent networks
    unsigned num_layers;                    // Number of layers

    float *input_data;                      // Input data
    float *input_label;                     // Input label
    
    std::vector<layer_t*> layers;           // Network layers

    // Initialize network.
    virtual void init_network(const std::string m_network_config) = 0;
    // Load batch data.
    virtual void load_data(const unsigned m_batch_index) = 0;
    // Print reulsts.
    virtual void print_results() = 0;
    // Initialize input data.
	virtual void init_data(section_config_t m_data_config) = 0;
    //virtual void init_data(const std::string m_data_config) = 0;
    // Initialize weight.
    void init_weight(const std::string m_input_weight);
    // Store weight.
    void store_weight(const std::string m_output_weight);

    std::vector<std::string> labels;        // List of labels
    stopwatch_t stopwatch;                  // Stopwatch to measure runtime

    layer_t *input_layer;                   // Input layer
    layer_t *output_layer;                  // Output layer

#ifdef PRUNING
    float weight_threshold;
    float data_threshold;
#endif

    unsigned num_classes;                   // Number of output classes
    unsigned num_iterations;                // Number of iterations to run
    unsigned iteration;                     // Number of processed batches
    unsigned epoch_length;                  // Number of iterations in an epoch
    unsigned top_k;                         // Top-k indices for inference
	unsigned pipe_index;					// Index of the layer.
    float cumulative_cost;                  // Cumulative cost
    std::vector<float> cost_history;        // Latest cost history


};

}
// End of namespace nebula. 

#endif
