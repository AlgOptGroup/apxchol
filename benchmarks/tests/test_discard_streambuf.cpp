// Regression test for the rchol_par heap race: parallel RCHOL writes progress to
// std::cout from many threads while the adapter has silenced it. With the old
// shared std::ostringstream this crashed or corrupted the heap; the discard
// buffer must survive the same load and leak nothing to the real stream.
#include "discard_streambuf.h"

#include <cstdio>
#include <iostream>
#include <sstream>
#include <thread>
#include <vector>

int main() {
    std::ostringstream real;  // stands in for the process stdout
    std::streambuf* original = std::cout.rdbuf(real.rdbuf());
    {
        std::streambuf* old = std::cout.rdbuf();
        discard_streambuf devnull;
        std::cout.rdbuf(&devnull);
        std::vector<std::thread> workers;
        for (int t = 0; t < 64; ++t)
            workers.emplace_back([t] {
                for (int i = 0; i < 20000; ++i)
                    std::cout << "depth(separator): " << t << " nztotal " << i * 1.5 << "\n";
            });
        for (auto& w : workers) w.join();
        std::cout.rdbuf(old);
    }
    std::cout << "after";
    std::cout.rdbuf(original);
    if (real.str() != "after") {
        std::fprintf(stderr, "silenced output leaked or restore failed: '%s'\n", real.str().c_str());
        return 1;
    }
    std::puts("discard_streambuf: 64 concurrent writers, nothing leaked");
    return 0;
}
