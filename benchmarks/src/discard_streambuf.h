#pragma once

#include <streambuf>

// RCHOL prints progress to std::cout, from its worker threads in the parallel
// variant. Silencing it through a shared std::ostringstream raced those writers
// on one growing buffer and corrupted the heap; this buffer keeps no state.
struct discard_streambuf : std::streambuf {
    int_type overflow(int_type c) override { return traits_type::not_eof(c); }
    std::streamsize xsputn(const char*, std::streamsize n) override { return n; }
};
