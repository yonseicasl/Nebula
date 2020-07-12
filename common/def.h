#ifndef __DEFS_H__
#define __DEFS_H__

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#define BLOCK_SIZE 512

namespace nebula {

// Run types
enum run_type_t {
    UNDEFINED_RUN = 0,
    TEST_RUN,
    TRAIN_RUN,
    NUM_RUN_TYPES,
};

static std::vector<std::string> run_type_str __attribute__((unused)) = {
    "undefined_run",
    "test",
    "train",
    "num_run_types",
};

// Layer types
enum layer_type_t { 
    UNDEFINED_LAYER = 0,
    CONNECTED_LAYER,
    CONVOLUTIONAL_LAYER,
    COST_LAYER,
    DROPOUT_LAYER,
    AVGPOOL_LAYER,
    MAXPOOL_LAYER,
    SHORTCUT_LAYER,
    SOFTMAX_LAYER,
    RNN_LAYER,
    LSTM_LAYER,
    RBM_LAYER,
    NUM_LAYER_TYPES,
};

static std::vector<std::string> layer_type_str __attribute__((unused)) = {
    "undefined_layer",
    "connected_layer",
    "convolutional_layer",
    "cost_layer",
    "dropout_layer",
    "avgpool_layer",
    "maxpool_layer",
    "shortcut_layer",
    "softmax_layer",
    "rnn_layer",
    "lstm_layer",
    "rbm_layer",
    "num_layer_types",
};

// Activation type
enum activation_type_t {
    UNDEFINED_ACTIVATION = 0,
    ELU_ACTIVATION,
    HARDTAN_ACTIVATION,
    LEAKY_ACTIVATION,
    LHTAN_ACTIVATION,
    LINEAR_ACTIVATION,
    LOGGY_ACTIVATION,
    LOGISTIC_ACTIVATION,
    PLSE_ACTIVATION,
    RAMP_ACTIVATION,
    RELIE_ACTIVATION,
    RELU_ACTIVATION,
    STAIR_ACTIVATION,
    TANH_ACTIVATION,
    NUM_ACTIVATION_TYPES,
};

static std::vector<std::string> activation_type_str __attribute__((unused)) = {
    "undefined_activation",
    "elu",
    "hardtan",
    "leaky",
    "lhtan",
    "linear",
    "loggy",
    "logistic",
    "plse",
    "ramp",
    "relie",
    "relu",
    "stair",
    "tanh",
    "num_activation_types",
};

// Cost type
enum cost_type_t {
    UNDEFINED_COST = 0,
    L1_COST,
    L2_COST,
    SMOOTH_COST,
    SSE_COST,
    NUM_COST_TYPES,
};

static std::vector<std::string> cost_type_str __attribute__((unused)) = {
    "undefined_cost",
    "l1",
    "l2",
    "smooth",
    "sse",
    "num_cost_types",
};

#define is_valid_type(m_vector, m_string) \
    (find(m_vector.begin(), m_vector.end(), m_string.c_str()) != m_vector.end())

#define get_type(m_vector, m_string) \
    distance(m_vector.begin(), find(m_vector.begin(), m_vector.end(), m_string.c_str()))

}
// End of namespace nebula.
#endif

