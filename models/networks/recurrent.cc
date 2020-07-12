#ifndef CUSTOM_BLAS
    #include <cblas.h>
#endif
#include <string>
#include <fstream>
#include <iostream>
#include <functional>
#include <numeric>
#include <random>
#include <unistd.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#ifdef GPU_ENABLED
    #include <cuda_runtime.h>
#endif
#include "config.h"
#include "layer.h"
#include "recurrent.h"
#include "rnn_layer.h"
#include "lstm_layer.h"
#include "connected_layer.h"
#include "softmax_layer.h"
#include "cost_layer.h"

<<<<<<< HEAD
=======
namespace nebula {

>>>>>>> namespace
recurrent_t::recurrent_t() {
}

recurrent_t::~recurrent_t() {
    for(unsigned i = 0; i < layers.size(); i++) {delete layers[i];}
    delete [] input_data;
    delete [] input_label;
#ifdef GPU_ENABLED
    cublasDestroy(cublas_handle);
    cudaFree(input_data_dev);
    cudaFree(input_label_dev);
#endif
}


// Run network.
void recurrent_t::run(const std::string m_output_weight) {
    std::cout << "Running network ..." << std::endl;
    stopwatch.start();

    // Inference 
    if(run_type == TEST_RUN) {
        unsigned batch_count = num_iterations < inputs.size()/batch_size ?
                               num_iterations : inputs.size()/batch_size;
        for(iteration = 0; iteration < batch_count; iteration++) {
            load_data(iteration);
            // Forward propagation.
            forward();
            //Print results.
            print_results();
        }
    }
    // Training
    else {
        unsigned batch_count = num_iterations;
        for(iteration = 0; iteration < batch_count; iteration++) {
            // Reset cost
            cost = 0.0;
            // Load data.
            load_data(iteration);
            // Forward propagation.
            forward();
            // Backward propagation.
            backward();
            // Update network.
            update();

            cumulative_cost += cost;
            cost_history.push_back(cost);
            if(cost_history.size() > epoch_length) { cost_history.erase(cost_history.begin()); }
            if(iteration % epoch_length == 0) { store_weight(m_output_weight); }
            if(iteration % (epoch_length * 30) == epoch_length * 30 -1) {learning_rate /=10;}
            // Print processing results.
            print_results();
        }
        store_weight(m_output_weight);
    }

    // Display total runtime.
    stopwatch.stop();
    float total_runtime = stopwatch.get_total_time();

    std::cout << std::endl << "Total runtime: ";
    if(total_runtime < 1.0) {
        std::cout << std::fixed << std::setprecision(6) << total_runtime*1e3 << "usec";
    }
    else if(total_runtime < 1e3) {
        std::cout << std::fixed << std::setprecision(6) << total_runtime << "msec";
    }
    else if(total_runtime < 60e3) {
        std::cout << std::fixed << std::setprecision(6) << total_runtime/1e3 << "sec";
    }
    else if(total_runtime < 3600e3) {
        unsigned min = total_runtime/60e3;
        unsigned sec = (total_runtime - min*60e3)/1e3;
        std::cout << min << "min " << sec << "sec";
    }
    else {
        unsigned hour = total_runtime/3600e3;
        unsigned min  = (total_runtime - hour*3600e3)/60e3;
        unsigned sec  = (total_runtime - min*60e3 - hour*3600e3)/1e3;
        std::cout << hour << "h " << min << "min " << sec << "sec";
    }
    std::cout << std::endl;
    std::cout << std::endl << "Network " << run_type_str[run_type] << " done." << std::endl;
}

// Initialize network.
void recurrent_t::init_network(std::string m_network_config) {
    // Parse the configuration file.
    config_t config;
    config.parse(m_network_config);

    // Number of layers is equivalent to the size of sections.
    // -1 counts for generic network setting section.
    num_layers = config.sections.size() - 1;
    layers.reserve(num_layers);

    for(size_t i = 0; i < config.sections.size(); i++) {
        section_config_t section_config = config.sections[i];
        // Network configuration
        if(section_config.name == "net") {
            section_config.get_setting("num_threads", &num_threads);
            section_config.get_setting("learning_rate", &learning_rate);
            section_config.get_setting("momentum", &momentum);
            section_config.get_setting("decay", &decay);
            section_config.get_setting("input_size", &input_size);
            section_config.get_setting("time_steps", &time_step);
            section_config.get_setting("num_iterations", &num_iterations);
                section_config.get_setting("batch", &batch_size);
            if(run_type == TRAIN_RUN) {
            }
            batch_size *= time_step;
        }
        // Layer configuration
        else {
            layer_t *layer = NULL;
            if(section_config.name == "rnn") {
                layer = new rnn_layer_t(this, layers.size()?layers[layers.size()-1]:NULL, RNN_LAYER);
            }
            else if(section_config.name == "lstm") {
                layer = new lstm_layer_t(this, layers.size()?layers[layers.size()-1]:NULL, LSTM_LAYER);
            }
            else if(section_config.name == "connected") {
                layer = new connected_layer_t(this, layers.size()?layers[layers.size()-1]:NULL, CONNECTED_LAYER);
            }
            else if(section_config.name == "softmax") {
                layer = new softmax_layer_t(this, layers.size()?layers[layers.size()-1]:NULL, SOFTMAX_LAYER);
                // Softmax layer is output layer.
                output_layer = layer;
            }
            else if(section_config.name == "cost") {
                layer = new cost_layer_t(this, layers.size()?layers[layers.size()-1]:NULL, COST_LAYER);
            }
            else {
                std::cerr << "Error: unknown layer type " << section_config.name << std::endl;
                exit(1);
            }
            // The first created layer becomes input layer.
            if(!layers.size()) { input_layer = layer; }
            // Initialize layer.
            layer->init(section_config);
            layers.push_back(layer);
        }
    }
#ifndef CUSTOM_BLAS
    // Multi-thread openblas
    openblas_set_num_threads(num_threads);
#endif
}

// Load batch data.
void recurrent_t::load_data(const unsigned m_batch_index) {
    std::vector<std::string> batch_input_sentence;
    batch_input_sentence.reserve(batch_size / time_step);

    memset(input_data, 0.0, input_size * batch_size * sizeof(float));
    memset(input_label, 0.0, input_size * batch_size * sizeof(float));
    unsigned stream = batch_size / time_step;
    // Use one hot encoding when write input value to input data array.
    if(run_type == TEST_RUN) {
        unsigned current_index = m_batch_index * batch_size;
        for(unsigned b = 0; b < stream; b++) {
            for(unsigned step = 0; step < time_step; step++) {
                iter = find(labels.begin(), labels.end(), inputs[current_index % inputs.size()]);
                unsigned index = distance(labels.begin(), iter);
                //cout << labels[index] << endl;
                input_data[step * stream * input_size + b * input_size + index] = 1.0;
                iter = find(labels.begin(), labels.end(), inputs[(current_index+1) % inputs.size()]);
                index = distance(labels.begin(), iter);
                input_label[step * stream * input_size + b * input_size + index] = 1.0;
                current_index++;
            }
        }
    }
    else {
        std::minstd_rand rng(std::random_device{}());
        std::uniform_int_distribution<unsigned> uid(0, input_size - 1);
        //unsigned current_index = uid(rng);
        for(unsigned b = 0; b < stream; b++) {
            unsigned current_index = uid(rng);
            for(unsigned step = 0; step < time_step; step++) {
                iter = find(labels.begin(), labels.end(), inputs[current_index % inputs.size()]);
                unsigned index = distance(labels.begin(), iter);
                input_data[step * stream * input_size + b * input_size + index] = 1.0;
                iter = find(labels.begin(), labels.end(), inputs[(current_index + 1) % inputs.size()]);
                index = distance(labels.begin(), iter);
                input_label[step * stream * input_size + b * input_size + index] = 1.0;
                current_index++;
            }
        }
    }
#ifdef GPU_ENABLED
    cudaMemcpy(input_data_dev, input_data, 
               input_size * batch_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(input_label_dev, input_label, 
               input_size * batch_size * sizeof(float), cudaMemcpyHostToDevice);
#endif
}

// Print results.
void recurrent_t::print_results() {
    if(run_type == TEST_RUN) {
        for(unsigned i = 0; i < output_layer->output_size * batch_size; i++) {
            cumulative_cost += input_label[i] * output_layer->output_data[i];
        }

        std::cout << "Iteration #" << iteration
                  << " (data #" << ((iteration+1) * batch_size) << "):" << std::endl;
        std::cout << "  - perplexity : " << std::fixed << std::setprecision(2)
             //<< cumulative_cost / ((iteration + 1) * batch_size) << endl;
                  << exp((-1) * log2(cumulative_cost / ((iteration + 1) * batch_size))) << std::endl;
    }
    else {
        stopwatch.stop();
        float interval_runtime = stopwatch.get_interval_time();

        // Print results.
        std::cout << "Iteration #" << iteration
                  << " (data #" << ((iteration+1) * batch_size) << "):" << std::endl;
        std::cout << "  - loss (iteration): " << std::fixed << std::setprecision(6)
                  << cost / batch_size << std::endl;
        std::cout << "  - loss (epoch: last " << cost_history.size() << " iterations): "
                  << std::fixed << std::setprecision(6)
                  << accumulate(cost_history.begin(), cost_history.end(), 0.0) /
                     (cost_history.size() * batch_size) << std::endl;
        std::cout << "  - loss (cumulative): " << std::fixed << std::setprecision(6)
                  << cumulative_cost / (iteration + 1) / batch_size << std::endl;
        std::cout << "  - runtime: ";
        if(interval_runtime < 1.0) {
            std::cout << std::fixed << std::setprecision(6) << interval_runtime*1e3 << "usec";
        }
        else if(interval_runtime < 1e3) {
            std::cout << std::fixed << std::setprecision(6) << interval_runtime << "msec";
        }
        else if(interval_runtime < 60e3) {
            std::cout << std::fixed << std::setprecision(6) << interval_runtime/1e3 << "sec";
        }
        else if(interval_runtime < 3600e3) {
            std::cout << interval_runtime << std::endl;
            unsigned min = interval_runtime/60e3;
            unsigned sec = (interval_runtime - min*60e3)/1e3;
            std::cout << min << "min " << sec << "sec";
        }
        else {
            unsigned hour = interval_runtime/3600e3;
            unsigned min  = (interval_runtime - hour*3600e3)/60e3;
            unsigned sec  = (interval_runtime - min*60e3 - hour*3600e3)/1e3;
            std::cout << hour << "h " << min << "min " << sec << "sec";
        }
        std::cout << std::endl;

        // Resume stopwatch.
        stopwatch.start();
    }
}

void recurrent_t::init_data(const std::string m_data_config){
    
    config_t config;
    config.parse(m_data_config);
    section_config_t section_config = config.sections[0];

    if((config.sections.size() != 1) || (config.sections[0].name != "data")) {
        std::cerr << "Error : input config format error in " << m_data_config << std::endl;
        exit(1);
    }
    
    // Read input configuration.
    std::string input_list, label_list;
    if(run_type == TEST_RUN) { section_config.get_setting("test", &input_list);}
    else { section_config.get_setting("train", &input_list);}
    section_config.get_setting("labels", &label_list);

    // Read input file.
    std::fstream input_list_file;
    input_list_file.open(input_list.c_str(), std::fstream::in);
    if(!input_list_file.is_open()) {
        std::cerr << "error : failed to open " << input_list << std::endl;
        exit(1);
    }
    // Read input token from input file.
    // Save input token to vector named inputs.
    std::string input;
    while(!input_list_file.eof()) {
        getline(input_list_file, input, ' ');
        input.erase(remove(input.begin(), input.end(), '\n'), input.end());
        inputs.push_back(lowercase(input));
    }
    epoch_length = inputs.size() / batch_size;
           
    input_list_file.close();

    // Read label configuration.
    std::fstream label_list_file;
    label_list_file.open(label_list.c_str(), std::fstream::in);
    if(!label_list_file.is_open()) {
        std::cerr << "error : failed to open " << label_list << std::endl;
        exit(1);
    }

    // Read label words from label file.
    // Store label words to labels vector.
    std::string label;
    while(!label_list_file.eof()) {
        getline(label_list_file, label, ' ');
        label.erase(remove(label.begin(), label.end(), '\n'), label.end());
        if(find(labels.begin(), labels.end(), lowercase(label)) == labels.end()){
            labels.push_back(label);
        }
    }

    label_list_file.close();

    input_size = labels.size();
    std::cout << input_size  << ' ' << batch_size << std::endl;
    input_data  = new float[input_size * batch_size]();
    input_label = new float[input_size * batch_size]();
#ifdef GPU_ENABLED
    cudaMalloc((void**)&input_data_dev, input_size * batch_size * sizeof(float));
    cudaMalloc((void**)&input_label_dev, input_size * batch_size * sizeof(float));
    cudaMemset(input_data_dev, 0.0, input_size * batch_size * sizeof(float));
    cudaMemset(input_label_dev, 0.0, input_size * batch_size * sizeof(float));
#endif

}

}
<<<<<<< HEAD

=======
// End of namespace nebula.
>>>>>>> namespace
