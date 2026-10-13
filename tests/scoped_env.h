#pragma once

#include <cstdlib>
#include <string>

// Override one environment variable for a scope, preserving unset versus empty.
class scoped_env {
public:
    scoped_env(const char* name, const char* value) : name_(name) {
        if (const char* previous = std::getenv(name)) {
            previous_ = previous;
            had_value_ = true;
        }
        if (value) {
            setenv(name_.c_str(), value, 1);
        } else {
            unsetenv(name_.c_str());
        }
    }

    ~scoped_env() {
        if (had_value_) {
            setenv(name_.c_str(), previous_.c_str(), 1);
        } else {
            unsetenv(name_.c_str());
        }
    }

    scoped_env(const scoped_env&) = delete;
    scoped_env& operator=(const scoped_env&) = delete;
    scoped_env(scoped_env&&) = delete;
    scoped_env& operator=(scoped_env&&) = delete;

private:
    std::string name_;
    std::string previous_;
    bool had_value_ = false;
};
