#include "box.h"

box_t::box_t() :
    height(0),
    width(0),
    x_center(0),
    x_left(0),
    x_right(0),
    y_center(0),
    y_bottom(0),
    y_top(0) {

}

box_t::~box_t(){
}

void box_t::init() {

}

void box_t::make_box(unsigned m_height, unsigned m_width) {
    x_left = x_center - width/2 > 0 ? x_center - width/2 : 0;
    x_right = x_center + width/2 < m_width - 1 ? 
              x_center + width/2 : m_width - 1;
    
    y_bottom = y_center - height/2 > 0 ? y_center - height/2 : 0;
    y_top = y_center + height/2 < m_height -1 ? 
            y_center + height/2 : m_height - 1;
}
