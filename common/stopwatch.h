#ifndef __STOPWATCH_H__
#define __STOPWATCH_H__

#include <sys/time.h>
#include "def.h"
#include <iomanip>

class stopwatch_t {
public:
    stopwatch_t();
    ~stopwatch_t();

    void start();
    void stop();
    void reset();
    float get_interval_time() const;
    float get_total_time() const;

private:
    float total_time;
    float interval_time;
    timeval start_time;
    timeval end_time;
};


#endif

