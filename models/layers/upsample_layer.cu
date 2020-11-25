extern "C++" {
#include "upsample_layer.h"
}

namespace nebula {

__global__ void _forward_upsample_() {}

__global__ void _backward_upsample_() {
}

extern "C++" void upsample_layer_t::_forward_() {
    float *input_data_dev = prev_layer ? prev_layer->output_data_dev : network->input_data_dev;

    cudaMemset(output_data_dev, 0.0, output_size * network->batch_size * sizeof(float));
    cudaMemset(delta_dev, 0.0, output_size * network->batch_size * sizeof(float));

    //dim3 cuda_griddim = {(output_size * network->batch_size
}

extern "C++" void upsample_layer_t::_backward_() {

}

extern "C++" void upsample_layer_t::_update_() {
    // Nothing to do.
}




}
