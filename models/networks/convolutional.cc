#include <string>
#include <fstream>
#include <iostream>
#include <functional>
#include <numeric>
#include <random>
#include <thread>
#include <unistd.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#ifdef GPU_ENABLED
#include <cuda_runtime.h>
#endif
#ifndef CUSTOM_BLAS
#include <cblas.h>
#endif
#include "convolutional.h"
#include "config.h"
#include "layer.h"
#include "convolutional_layer.h"
#include "connected_layer.h"
#include "dropout_layer.h"
#include "softmax_layer.h"
#include "cost_layer.h"
#include "shortcut_layer.h"
#include "pooling_layer.h"

//using namespace std;
//using namespace cv;

convolutional_t::convolutional_t() {
}

convolutional_t::~convolutional_t() {
    for(unsigned i = 0; i < layers.size(); i++) { delete layers[i]; }
    delete [] input_data;
    delete [] input_label;
    delete [] reference_label;
#ifdef GPU_ENABLED
    cublasDestroy(cublas_handle);
    cudaFree(input_data_dev);
    cudaFree(input_label_dev);
#endif
}

// Initialize network.
void convolutional_t::init_network(const std::string m_network_config) {
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
            section_config.get_setting("height", &input_height);
            section_config.get_setting("width", &input_width);
            section_config.get_setting("channels", &input_channel);
            section_config.get_setting("batch", &batch_size);
            // Total number of iterations will be later updated in init_data()
            section_config.get_setting("num_iterations", &num_iterations);
            input_size = input_height * input_width * input_channel;
        }
        // Layer configuration
        else {
            layer_t *layer = NULL;
            if(section_config.name == "convolutional") {
                layer = new convolutional_layer_t(this, layers.size()?layers[layers.size()-1]:NULL, CONVOLUTIONAL_LAYER);
            }
            else if(section_config.name == "connected") {
                layer = new connected_layer_t(this, layers.size()?layers[layers.size()-1]:NULL, CONNECTED_LAYER);
            }
            else if(section_config.name == "dropout") {
                layer = new dropout_layer_t(this, layers.size()?layers[layers.size()-1]:NULL, DROPOUT_LAYER);
            }
            else if(section_config.name == "maxpool") {
                layer = new pooling_layer_t(this, layers.size()?layers[layers.size()-1]:NULL, MAXPOOL_LAYER);
            }
            else if(section_config.name == "avgpool") {
                layer = new pooling_layer_t(this, layers.size()?layers[layers.size()-1]:NULL, AVGPOOL_LAYER);
            }
            else if(section_config.name == "shortcut") {
                layer = new shortcut_layer_t(this, layers.size()?layers[layers.size()-1]:NULL, SHORTCUT_LAYER);
            }
            else if(section_config.name == "softmax") {
                layer = new softmax_layer_t(this, layers.size()?layers[layers.size()-1]:NULL, SOFTMAX_LAYER);
                // Softmax is output layer.
                output_layer = layer;
            }
            else if(section_config.name == "cost") {
                layer = new cost_layer_t(this, layers.size() ? layers[layers.size()-1] : NULL, COST_LAYER);
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

void convolutional_t::init_data(const std::string m_data_config) {
    // Parse input data config.
    config_t config;
    config.parse(m_data_config);
    section_config_t section_config = config.sections[0];

    if((config.sections.size() != 1) || (config.sections[0].name != "data")) {
        std::cerr << "Error: input config format error in " << m_data_config << std::endl;
        exit(1);
    }

    // Input configuration
    std::string input_list, label_list;
    if(run_type == TEST_RUN) { section_config.get_setting("test", &input_list); }
    else { section_config.get_setting("train", &input_list); }
    section_config.get_setting("labels", &label_list);
    section_config.get_setting("top", &top_k);
   
    // Read input list.
    std::fstream input_list_file;
    input_list_file.open(input_list.c_str(), std::fstream::in);
    if(!input_list_file.is_open()) {
        std::cerr << "Error: failed to open " << input_list << std::endl;
        exit(1);
    }
    std::string input;
    while(getline(input_list_file, input)) { inputs.push_back(input); }
    input_list_file.close();

    // Update epoch length and num_iterations
    epoch_length = inputs.size() / batch_size;
    //num_iterations *= inputs.size();

    // Read label list.
    std::fstream label_list_file;
    label_list_file.open(label_list.c_str(), std::fstream::in);
    if(!label_list_file.is_open()) {
        std::cerr << "Error: failed to open " << label_list << std::endl;
        exit(1);
    }
    std::string label;
    while(getline(label_list_file, label)) { labels.push_back(label); }
    num_classes = labels.size();
    label_list_file.close();

    // Reserve memory for input data and labels.
    input_size = input_height * input_width * input_channel;
    input_data = new float[input_size*batch_size];
    input_label = new float[num_classes * batch_size]();
    reference_label = new unsigned[batch_size]();
#ifdef GPU_ENABLED
    cudaMalloc((void**)&input_data_dev, input_size * batch_size * sizeof(float));
    cudaMalloc((void**)&input_label_dev, num_classes * batch_size * sizeof(float));
#endif
}

// Run network.
void convolutional_t::run(const std::string m_output_weight) {
    std::cout << "Running network ..." << std::endl;
    stopwatch.start();

    // Inference 
    if(run_type == TEST_RUN) {
        // Set batch count as num_iterations or inputs.size()/batch_size, whichever is smaller.
        //unsigned batch_count = inputs.size()/batch_size;
		unsigned batch_count = num_iterations -1;
        
        for(iteration = 0; iteration < batch_count; iteration++) {
            // Loda batch data.
            load_data(iteration);
            // Forward propagation
            forward();
            // Print batch processing results.
            print_results();
        }
		print_results();
    }
    // Training
    else {
        //unsigned batch_count = num_iterations ? num_iterations : inputs.size()/batch_size;
        unsigned batch_count = num_iterations;
        for(iteration = 0; iteration < batch_count; iteration++) {
            // Reset cost.
            cost = 0.0;
            // Load batch data.
            load_data(iteration);
            // Forward propagation
            forward();
            // Backward propagation
            backward();
            // Network update
            update();

            // Update cost history
            cumulative_cost += cost;
            cost_history.push_back(cost);
            if(cost_history.size() > epoch_length) { cost_history.erase(cost_history.begin()); }
            if(iteration % epoch_length == 0) { store_weight(m_output_weight); }
            if(iteration % (epoch_length * 20) == epoch_length * 20 -1) { learning_rate /= 10;} 

            // Print batch processing results.
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




// Load batch data.
void convolutional_t::load_data(const unsigned m_batch_index) {
    // Load batch data.
    std::vector<std::string> batch_inputs;
    batch_inputs.reserve(batch_size);

    // Inference
    if(run_type == TEST_RUN) {
        // Sequentially load batch data.
        for(unsigned i = 0; i < batch_size; i++) {
            batch_inputs.push_back(inputs[m_batch_index*batch_size + i]);
        }
    }
    // Training
    else {
        // Randomly load batch data.
        std::minstd_rand rng(std::random_device{}());
        std::uniform_int_distribution<unsigned> uid(0,inputs.size()-1);
        for(unsigned i = 0; i < batch_size; i++) {
            batch_inputs.push_back(inputs[uid(rng)]);
        }
    }

    // Mark matching labels in the batch.
    memset(input_label, 0, batch_size * num_classes * sizeof(float)); 
    for(unsigned i = 0; i < batch_size; i++) {
        for(unsigned j = 0; j < num_classes; j++) {
            if(batch_inputs[i].find(labels[j]) != std::string::npos) {
                input_label[i*num_classes + j] = 1.0;
                reference_label[i] = j;
            }
        }
    }

#ifdef GPU_ENABLED
    // Copy input_label to device.
    cudaMemcpy(input_label_dev, input_label,
               batch_size * num_classes * sizeof(float), cudaMemcpyHostToDevice);
#endif

    // Set opencv flag.
    // flag -1: IMREAD_UNCHANGED
    // flag  0: IMREAD_GRAYSCALE
    // flag  1: IMREAD_COLOR
    int opencv_flag = -1;
    if(input_channel == 1) { opencv_flag = 0; }
    else if(input_channel == 3) { opencv_flag = 1; }
    else {
        std::cerr << "Error: unsupported image channel " << input_channel << std::endl;
        exit(1);
    }

    // Load data in parallel.
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    for(unsigned tid = 0; tid < num_threads; tid++) {
        threads.emplace_back(std::bind([&](const unsigned begin, const unsigned end,
                                           const unsigned tid) {
            for(unsigned i = begin; i < end; i++) {
                cv::Mat src, dst;
                // Check input data format.
                if(batch_inputs[i].find("png") != std::string::npos) { src = cv::imread(batch_inputs[i], -1); }
                else { src = cv::imread(batch_inputs[i], opencv_flag); }
                if(src.empty()) {
                    std::cerr << "Error: failed to load input " << inputs[i] << std::endl;
                    exit(1);
                }

                // Resize data.
                if((input_height != (unsigned)src.size().height) ||
                   (input_width  != (unsigned)src.size().width)) {
                    cv::resize(src, dst, cv::Size(input_width, input_height), 0, 0, cv::INTER_LINEAR);
                }
                else { dst = src; }

                // Flatten data into 1-D array.
                unsigned height  = dst.size().height;
                unsigned width   = dst.size().width;
                unsigned channel = dst.channels();
                float *data = new float[height * width * channel]();

                for(unsigned h = 0; h < height; h++) {
                    for(unsigned c = 0; c < channel; c++) {
                        for(unsigned w = 0; w < width; w++) {
                            data[c * width * height + h * width + w] =
                            dst.data[h * dst.step + w * channel + c]/255.0;
                        }
                    }
                }

                for(unsigned i = 0; i < height * width; i++) {
                    cv::swap(data[i], data[i + 2 * width * height]);
                }

                memcpy(input_data + i * input_size, data,
                       input_height * input_width * input_channel * sizeof(float));
                delete [] data;
            }
        }, tid * batch_size / num_threads, (tid + 1) * batch_size / num_threads, tid));
    } std::for_each(threads.begin(), threads.end(), [](std::thread& t) { t.join(); });

#ifdef GPU_ENABLED
    // Copy input data into device.
    cudaMemcpy(input_data_dev, input_data,
               input_size * batch_size * sizeof(float), cudaMemcpyHostToDevice);
#endif
}

// Print results.
void convolutional_t::print_results() {
    if(run_type == TEST_RUN) {
        // Array indices to sort out top-k classes.
        std::vector<unsigned> indices(num_classes);
        std::vector<unsigned> sorted(num_classes); {
            int x = 0;
            iota(sorted.begin(), sorted.end(), x++);
        }

        // Sort output neuron indices in decending order.
        static unsigned matches = 0;
        for(unsigned i = 0; i < batch_size; i++) {
            indices = sorted;
            std::sort(indices.begin(), indices.end(), [&](unsigned a, unsigned b) {
                return output_layer->output_data[i*num_classes + a] >
                       output_layer->output_data[i*num_classes + b];
            });
            for(unsigned k = 0; k < top_k; k++) {
                if(indices[k] == reference_label[i]) { matches++; }
            }
        }

        // Pause stopwatch.
        stopwatch.stop();
        float interval_runtime = stopwatch.get_interval_time();

        // Print results.
        std::cout << "Iteration #" << iteration
                  << " (data #" << ((iteration+1) * batch_size) << "):" << std::endl;
        std::cout << "  - accuracy: " << std::fixed << std::setprecision(6)
             << (100.0 * matches/((iteration+1) * batch_size)) << "%" << std::endl;
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
    else if(run_type == TRAIN_RUN) {
        // Pause stopwatch.
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
