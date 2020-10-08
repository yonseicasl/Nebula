#ifndef __BOX_H__
#define __BOX_H__

#include "layer.h"
#include "network.h"
#include "data.h"

class box_t {

public:
    box_t();
    ~box_t();

    void init();        // initialize the detection box.
    void make_box(unsigned m_height, unsigned m_width);

    unsigned height;       // Height of detection box.
    unsigned width;        // Width of detection box.
    unsigned x_center;     // X coordinate of detection box.
    unsigned x_left;       // X coordinate of detection box.
    unsigned x_right;      // X coordinate of detection box.
    unsigned y_center;     // Y coordinate of detection box.
    unsigned y_bottom;     // Y coordinate of detection box.
    unsigned y_top;         // Y coordinate of detection box.


};

#endif
