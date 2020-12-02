extern "C++" {
#include "concat_layer.h"
}

namespace nebula {

extern "C++" void concat_layer_t::_forward_() {
    cudaMemset(output_data_dev, 0.0, output_size * network->batch_size * sizeof(float));
    cudaMemset(delta_dev, 0.0, output_size * network->batch_size * sizeof(float));
}

extern "C++" void concat_layer_t::_backward_() {

}

extern "C++" void concat_layer_t::_update_() {
    // Nothing to do.
}





}
