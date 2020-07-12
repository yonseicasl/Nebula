#include <cstring>
#include <iostream>
#include "stopwatch.h"

namespace nebula {

stopwatch_t::stopwatch_t() :
    total_time(0.0),
    interval_time(0.0) {
}

stopwatch_t::~stopwatch_t() {
}

void stopwatch_t::start() {
    gettimeofday(&start_time, 0);
}

void stopwatch_t::stop()  {
    gettimeofday(&end_time, 0);

    // Calculate elapsed time in milliseconds.
    interval_time = 
        ((end_time.tv_sec-start_time.tv_sec)*1e3 +
        (end_time.tv_usec-start_time.tv_usec)/1e3);

    total_time += interval_time;
}

void stopwatch_t::reset() {
    total_time = 0.0;
    interval_time = 0.0;
    memset(&start_time, 0, sizeof(timeval));
    memset(&end_time,   0, sizeof(timeval));
}

float stopwatch_t::get_interval_time() const {
    return interval_time;
}

float stopwatch_t::get_total_time() const {
    return total_time;
}

}
// End of namespace nebula.
