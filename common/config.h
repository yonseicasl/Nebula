#ifndef __CONFIG_H__
#define __CONFIG_H__

#include <map>
#include <sstream>
#include <string>
#include <vector>
#include "utils.h"

namespace nebula {

// Configuration section
class section_config_t {
public:
    section_config_t(std::string m_name);
    ~section_config_t();

    // Add (key, value) pair to the latest section settings
    void add_setting(std::string m_key, std::string m_value);
    // Check if a setting exists.
    bool exists(std::string m_key);
    // Get the setting value. Return true if found.
    template <typename T>
    bool get_setting(std::string m_key, T *m_var) {
        std::map<std::string, std::string>::iterator it = settings.find(lowercase(m_key));
        if(it == settings.end()) return false;
        std::stringstream ss; ss.str(it->second);
        ss >> *m_var; return true;
    }

    // Section name
    std::string name;

private:
    // Section settings
    std::map<std::string, std::string> settings;
};

// Configuration
class config_t {
public:
    config_t();
    ~config_t();

    // Parse configuration file
    void parse(std::string m_config_name);
    std::vector<section_config_t> sections;
};

}
// End of namespace nebula
#endif

