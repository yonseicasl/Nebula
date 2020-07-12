#ifndef __DATA_H__
#define __DATA_H__

#include <cstring>
#include <vector>
#include <iostream>

namespace nebula {

template <typename T>
class batch_t {
public:
    batch_t() : 
        rows(0),
        cols(0),
        vals(NULL) {
    }
    ~batch_t() {
        for(size_t i = 0; i < rows; i++) { free(vals[i]); }
        free(vals);
    }
    void init(size_t m_rows, size_t m_cols) {
        rows = m_rows;
        cols = m_cols;
        vals = (T**)malloc(rows*sizeof(T*));
        for(size_t i = 0; i < rows; i++) {
            vals[i] = (T*)malloc(cols*sizeof(T));
            memset(vals[i], 0, cols*sizeof(T));
        }
    }

    size_t rows;
    size_t cols;
    T   ** vals;
};

template <typename T>
class image_t {
public: 
    image_t() :
        height(0),
        width(0),
        channel(0),
        vals(NULL) {}
    ~image_t() {
        free(vals);
    }
    void init(unsigned m_width, unsigned m_height, unsigned m_channel) {
        height = m_height;
        width = m_width;
        channel = m_channel;
        vals = (T*)malloc(height*width*channel*sizeof(T));
        memset(vals, 0, height*width*channel*sizeof(T));
    }

    unsigned height;  // rows
    unsigned width;   // cols
    unsigned channel; // channel
    T     *vals;
};

}
// End of namespace nebula.

#endif
