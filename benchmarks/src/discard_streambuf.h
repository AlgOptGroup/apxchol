#pragma once

#include <iostream>
#include <streambuf>

// RCHOL prints progress to std::cout, from its worker threads in the parallel
// variant. Silencing it through a shared std::ostringstream raced those writers
// on one growing buffer and corrupted the heap; this buffer keeps no state.
struct discard_streambuf : std::streambuf {
    int_type overflow(int_type c) override { return traits_type::not_eof(c); }
    std::streamsize xsputn(const char*, std::streamsize n) override { return n; }
};

// Discards std::cout for its lifetime and restores the previous buffer on every
// exit, including an exception thrown by the silenced code.
class discard_cout {
public:
    discard_cout() : previous_(std::cout.rdbuf(&sink_)) {}
    ~discard_cout() { std::cout.rdbuf(previous_); }
    discard_cout(const discard_cout&) = delete;
    discard_cout& operator=(const discard_cout&) = delete;

private:
    discard_streambuf sink_;
    std::streambuf* previous_;
};
