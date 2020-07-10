#include <cstdlib>
#include <iostream>
#include "recurrent.h"

using namespace std;

int main(int argc, char **argv) {
    // Usage help.
    if(argc < 4) {
        cerr << "Usage: " << argv[0] << " [test/train]"
             << " [network config] [input data config]"
             << " [input weight (optional)] [output weight (optional)]"
             << endl;
        exit(1);
    }
    // Input arguments
    string run_type       = argv[1];
    string network_config = argv[2];
    string data_config    = argv[3];
    string input_weight   = argc > 4 ? argv[4] : "";
    string output_weight  = argc > 5 ? argv[5] : "";

    // Check run type.
    if((run_type != "test") && (run_type != "train")) {
        cerr << "Error: unknown run type " << run_type << endl;
        exit(1);
    }

    // Create network.
    network_t *network = new recurrent_t();
    // Initialize network.
    network->init(run_type, network_config, data_config, input_weight);
    // Run network and save optional output weight.
    network->run(output_weight);
    // Delete network.
    delete network;

    return 0;
}

