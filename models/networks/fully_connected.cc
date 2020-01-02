#ifndef CUSTOM_BLAS
	#include <cblas.h>
#endif
#include <string>
#include <fstream>
#include <iostream>
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
#include "fully_connected.h"
#include "config.h"
#include "layer.h"
#include "rbm_layer.h"
#include "connected_layer.h"
#include "softmax_layer.h"
#include "cost_layer.h"

using namespace std;
using namespace cv;

fully_connected_t::fully_connected_t() :
    num_rbm_layers(0),
    num_pretrain_iterations(0) {
}

fully_connected_t::~fully_connected_t() {
    for(unsigned i = 0; i < layers.size(); i++) { delete layers[i]; }
    delete [] input_data;
    delete [] input_label;
    delete [] reference_label;
#ifdef GPU_ENABLED
    cublasDestroy(cublas_handle);
    cudaFree(input_data_dev);
    cudaFree(input_label_dev);
    curandDestroyGenerator(generator);
#endif
}

// Initialize network.
void fully_connected_t::init_network(string m_network_config) {
   
#ifdef GPU_ENABLED
    curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
#endif
    
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
            section_config.get_setting("num_iterations", &num_iterations);
            section_config.get_setting("num_pretrain_iterations", &num_pretrain_iterations);
            input_size = input_height * input_width * input_channel;
        }
        // Layer configuration
        else {
            layer_t *layer = NULL;
            if(section_config.name == "rbm") {
                layer = new rbm_layer_t(this, layers.size() ? layers[layers.size()-1] : NULL, RBM_LAYER);
                num_rbm_layers++;
            }
            else if(section_config.name == "connected") {
                layer = new connected_layer_t(this, layers.size() ? layers[layers.size()-1] : NULL, CONNECTED_LAYER);
            }
            else if(section_config.name == "softmax") {
                layer = new softmax_layer_t(this, layers.size() ? layers[layers.size()-1] : NULL, SOFTMAX_LAYER);
                // Softmax layer is output layer.
                output_layer = layer;
            }
            else if(section_config.name == "cost") {
                layer = new cost_layer_t(this, layers.size() ? layers[layers.size()-1] : NULL, COST_LAYER);
            }
            else {
                cerr << "Error: unknown layer type " << section_config.name << endl;
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

// Initialize input data.
void fully_connected_t::init_data(const string m_data_config){
    
    // Parse input data config.
    config_t config;
    config.parse(m_data_config);
    section_config_t section_config = config.sections[0];

    if((config.sections.size() != 1) || (config.sections[0].name != "data")) {
        cerr << "Error: input config format error in " << m_data_config << endl;
        exit(1);
    }

    // Input configuration
    string input_list, label_list;
    if(run_type == TEST_RUN) { section_config.get_setting("test", &input_list); }
    else { section_config.get_setting("train", &input_list); }
    section_config.get_setting("labels", &label_list);
    section_config.get_setting("top", &top_k);
   
    // Read input list.
    fstream input_list_file;
    input_list_file.open(input_list.c_str(), fstream::in);
    if(!input_list_file.is_open()) {
        cerr << "Error: failed to open " << input_list << endl;
        exit(1);
    }
    string input;
    while(getline(input_list_file, input)) { inputs.push_back(input); }
    input_list_file.close();

    // Update epoch length and num_iterations
    epoch_length = inputs.size() / batch_size;
    //num_iterations *= inputs.size();

    // Read label list.
    fstream label_list_file;
    label_list_file.open(label_list.c_str(), fstream::in);
    if(!label_list_file.is_open()) {
        cerr << "Error: failed to open " << label_list << endl;
        exit(1);
    }
    string label;
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
void fully_connected_t::pretrain(unsigned m_layer_index) {
#ifdef GPU_ENABLED
    for(unsigned j = 0; j < m_layer_index; j++) {
        if(layers[j]->layer_type == RBM_LAYER) {
            ((rbm_layer_t*)layers[j])->_forward_(); 
        }
    }
    if(layers[m_layer_index]->layer_type == RBM_LAYER) {
        ((rbm_layer_t*)layers[m_layer_index])->_pretrain_(); 
    }
#else
    for(unsigned j = 0; j < m_layer_index; j++) {
        if(layers[j]->layer_type == RBM_LAYER) {
            ((rbm_layer_t*)layers[j])->forward(); 
        }
    }
    if(layers[m_layer_index]->layer_type == RBM_LAYER) {
        ((rbm_layer_t*)layers[m_layer_index])->pretrain(); 
    }
#endif
}

// Run network.
void fully_connected_t::run(const string m_output_weight) {
    cout << "Running network ..." << endl;
    stopwatch.start();

    // Inference 
    if(run_type == TEST_RUN) {
        unsigned batch_count = num_iterations > inputs.size() / batch_size ? 
                               inputs.size() / batch_size : num_iterations;

        for(iteration = 0; iteration < batch_count; iteration++) {
            // Loda batch data.
            load_data(iteration);
            // Forward propagation
            forward();
            // Print batch processing results.
            print_results();
        }
    }
    // Training
    else {
        unsigned batch_count = num_iterations ? num_iterations : inputs.size()/batch_size;
         
        // Pretrain the fully-connected network.
        for(unsigned i = 0; i < num_rbm_layers; i++) {
            for(iteration = 0; iteration < num_pretrain_iterations; iteration++) {
                cout << "pretrain : " << iteration <<endl;
                // Load batch data.
                load_data(iteration);
                // Pretrain using rbm layer
                pretrain(i);
            }
        }
         // Fine tuning the fully-connected network.
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

    cout << endl << "Total runtime: ";
    if(total_runtime < 1.0) {
        cout << std::fixed << std::setprecision(6) << total_runtime*1e3 << "usec";
    }
    else if(total_runtime < 1e3) {
        cout << std::fixed << std::setprecision(6) << total_runtime << "msec";
    }
    else if(total_runtime < 60e3) {
        cout << std::fixed << std::setprecision(6) << total_runtime/1e3 << "sec";
    }
    else if(total_runtime < 3600e3) {
        unsigned min = total_runtime/60e3;
        unsigned sec = (total_runtime - min*60e3)/1e3;
        cout << min << "min " << sec << "sec";
    }
    else {
        unsigned hour = total_runtime/3600e3;
        unsigned min  = (total_runtime - hour*3600e3)/60e3;
        unsigned sec  = (total_runtime - min*60e3 - hour*3600e3)/1e3;
        cout << hour << "h " << min << "min " << sec << "sec";
    }
    cout << endl;
    cout << endl << "Network " << run_type_str[run_type] << " done." << endl;
}


// Load batch data.
void fully_connected_t::load_data(const unsigned m_batch_index) {
    // Load batch data.
    vector<string> batch_inputs;
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
        minstd_rand rng(random_device{}());
        uniform_int_distribution<unsigned> uid(0,inputs.size()-1);
        for(unsigned i = 0; i < batch_size; i++) {
            batch_inputs.push_back(inputs[uid(rng)]);
        }
    }

    // Mark matching labels in the batch.
    memset(input_label, 0, batch_size * num_classes * sizeof(float)); 
    for(unsigned i = 0; i < batch_size; i++) {
        for(unsigned j = 0; j < num_classes; j++) {
            if(batch_inputs[i].find(labels[j]) != string::npos) {
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
        cerr << "Error: unsupported image channel " << input_channel << endl;
        exit(1);
    }

    // Load data in parallel.
    vector<thread> threads;
    threads.reserve(num_threads);
    for(unsigned tid = 0; tid < num_threads; tid++) {
        threads.emplace_back(bind([&](const unsigned begin, const unsigned end,
                                      const unsigned tid) {
            for(unsigned i = begin; i < end; i++) {
                Mat src, dst;
                // Check input data format.
                if(batch_inputs[i].find("png") != string::npos) { src = imread(batch_inputs[i], -1); }
                else { src = imread(batch_inputs[i], opencv_flag); }
                if(src.empty()) {
                    cerr << "Error: failed to load input " << inputs[i] << endl;
                    exit(1);
                }

                // Resize data.
                if((input_height != (unsigned)src.size().height) ||
                   (input_width  != (unsigned)src.size().width)) {
                    resize(src, dst, Size(input_width, input_height), 0, 0, INTER_LINEAR);
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
                    swap(data[i], data[i + 2 * width * height]);
                }

                memcpy(input_data + i * input_size, data,
                       input_height * input_width * input_channel * sizeof(float));
                delete [] data;
            }
        }, tid * batch_size / num_threads, (tid + 1) * batch_size / num_threads, tid));
    } for_each(threads.begin(), threads.end(), [](thread& t) { t.join(); });

    for(unsigned i = 0; i < input_size * batch_size;  i++) {
        input_data[i] = 1.0 - input_data[i];
    }
#ifdef GPU_ENABLED
    // Copy input data into device.
    cudaMemcpy(input_data_dev, input_data,
               input_size * batch_size * sizeof(float), cudaMemcpyHostToDevice);
#endif
}

// Print results.
void fully_connected_t::print_results() {
    if(run_type == TEST_RUN) {
        // Array indices to sort out top-k classes.
        vector<unsigned> indices(num_classes);
        vector<unsigned> sorted(num_classes); {
            int x = 0;
            iota(sorted.begin(), sorted.end(), x++);
        }

        // Sort output neuron indices in decending order.
        static unsigned matches = 0;
        for(unsigned i = 0; i < batch_size; i++) {
            indices = sorted;
            sort(indices.begin(), indices.end(), [&](unsigned a, unsigned b) {
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
        cout << "Iteration #" << iteration
             << " (data #" << ((iteration+1) * batch_size) << "):" << endl;
        cout << "  - accuracy: " << std::fixed << std::setprecision(6)
             << (100.0 * matches/((iteration+1) * batch_size)) << "%" << endl;
        cout << "  - runtime: ";
        if(interval_runtime < 1.0) {
            cout << std::fixed << std::setprecision(6) << interval_runtime*1e3 << "usec";
        }
        else if(interval_runtime < 1e3) {
            cout << std::fixed << std::setprecision(6) << interval_runtime << "msec";
        }
        else if(interval_runtime < 60e3) {
            cout << std::fixed << std::setprecision(6) << interval_runtime/1e3 << "sec";
        }
        else if(interval_runtime < 3600e3) {
            cout << interval_runtime << endl;
            unsigned min = interval_runtime/60e3;
            unsigned sec = (interval_runtime - min*60e3)/1e3;
            cout << min << "min " << sec << "sec";
        }
        else {
            unsigned hour = interval_runtime/3600e3;
            unsigned min  = (interval_runtime - hour*3600e3)/60e3;
            unsigned sec  = (interval_runtime - min*60e3 - hour*3600e3)/1e3;
            cout << hour << "h " << min << "min " << sec << "sec";
        }
        cout << endl;

        // Resume stopwatch.
        stopwatch.start();
    }
    else if(run_type == TRAIN_RUN) {
        // Pause stopwatch.
        stopwatch.stop();
        float interval_runtime = stopwatch.get_interval_time();

        // Print results.
        cout << "Iteration #" << iteration
             << " (data #" << ((iteration+1) * batch_size) << "):" << endl;
        cout << "  - loss (iteration): " << std::fixed << std::setprecision(6)
             << cost / batch_size << endl;
        cout << "  - loss (epoch: last " << cost_history.size() << " iterations): "
             << std::fixed << std::setprecision(6)
             << accumulate(cost_history.begin(), cost_history.end(), 0.0) /
                (cost_history.size() * batch_size) << endl;
        cout << "  - loss (cumulative): " << std::fixed << std::setprecision(6)
             << cumulative_cost / (iteration + 1) / batch_size << endl;
        cout << "  - runtime: ";
        if(interval_runtime < 1.0) {
            cout << std::fixed << std::setprecision(6) << interval_runtime*1e3 << "usec";
        }
        else if(interval_runtime < 1e3) {
            cout << std::fixed << std::setprecision(6) << interval_runtime << "msec";
        }
        else if(interval_runtime < 60e3) {
            cout << std::fixed << std::setprecision(6) << interval_runtime/1e3 << "sec";
        }
        else if(interval_runtime < 3600e3) {
            cout << interval_runtime << endl;
            unsigned min = interval_runtime/60e3;
            unsigned sec = (interval_runtime - min*60e3)/1e3;
            cout << min << "min " << sec << "sec";
        }
        else {
            unsigned hour = interval_runtime/3600e3;
            unsigned min  = (interval_runtime - hour*3600e3)/60e3;
            unsigned sec  = (interval_runtime - min*60e3 - hour*3600e3)/1e3;
            cout << hour << "h " << min << "min " << sec << "sec";
        }
        cout << endl;

        // Resume stopwatch.
        stopwatch.start();
    }
}

