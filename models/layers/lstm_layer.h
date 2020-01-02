#ifndef __LSTM_LAYER_H__
#define __LSTM_LAYER_H__

#include "layer.h"
#include "connected_layer.h"

class lstm_layer_t : public layer_t {
public:
    lstm_layer_t(network_t *m_network, layer_t *m_prev_layer, layer_type_t m_layer_type);
    ~lstm_layer_t();

    // Initialize layer.
    void init(section_config_t m_section_config);
    // Initialize weight from file.
    void init_weight(std::fstream &m_input_weight);
    // Initialize weight from scratch.
    void init_weight();
    // Forward propagation.
    void forward();
    // Backward propagation.
    void backward();
    // update layer's parameters.
    void update();
    // Store weight.
    void store_weight(std::fstream &m_weight_file);

#ifdef GPU_ENABLED
    // Forward propagation.
    void _forward_();
    // Backward propagation.
    void _backward_();
    // update layer's parameters.
    void _update_();
#endif

private:
    //float *prev_output_data;
    float *cell_state;                      // Cell state.
    float *current_cell_state;
    float *cell_delta;

    float *hidden_state;                    // Hidden layer's neuron.

    float *input_gate;
    float *forget_gate;
    float *next_forget_gate;
    float *cell_gate;
    float *output_gate;
    bool  batch_normalize;

#ifdef GPU_ENABLED
    //float *prev_output_data_dev;
    float *cell_state_dev;                  // Cell state.
    float *current_cell_state_dev;          // Current cell state.
    float *cell_delta_dev;

    float *hidden_state_dev;                // Hidden layer's neuron.

    float *input_gate_dev;                  
    float *forget_gate_dev;
    float *next_forget_gate_dev;
    float *cell_gate_dev;
    float *output_gate_dev;
    
#endif

    connected_layer_t *input_gate_W;        // Input gate weight from previous hidden layer.
    connected_layer_t *forget_gate_W;       // Forget gate weight from previous hidden layer.
    connected_layer_t *cell_gate_W;         // Cell gate weight from previous hidden layer.
    connected_layer_t *output_gate_W;       // Output gate weight from previous hidden layer.

    connected_layer_t *input_gate_U;        // Input gate weight from input layer.  
    connected_layer_t *forget_gate_U;       // Forget gate weight from input layer.
    connected_layer_t *cell_gate_U;         // Cell gate weight from input layer.
    connected_layer_t *output_gate_U;       // Output gate weight from input layer.

};

#endif
