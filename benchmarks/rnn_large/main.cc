#include <cstdlib>
#include <iostream>
#include "recurrent.h"

int main(int argc, char **argv) {
    // Usage help.
    if(argc < 4) {
        std::cerr << "Usage: " << argv[0] << " [test/train]"
                  << " [network config] [input data config]"
                  << " [input weight (optional)] [output weight (optional)]"
                  << std::endl;
        exit(1);
    }
    // Input arguments
    std::string run_type       = argv[1];
    std::string network_config = argv[2];
    std::string data_config    = argv[3];
    std::string input_weight   = argc > 4 ? argv[4] : "";
    std::string output_weight  = argc > 5 ? argv[5] : "";

    // Check run type.
    if((run_type != "test") && (run_type != "train")) {
        std::cerr << "Error: unknown run type " << run_type << std::endl;
        exit(1);
    }

    // Create network.
    nebula::network_t *network = new nebula::recurrent_t();
    // Initialize network.
    network->init(run_type, network_config, data_config, input_weight);
    // Run network and save optional output weight.
    network->run(output_weight);
    // Delete network.
    delete network;

    return 0;
}

