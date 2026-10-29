// Regression test for the rchol_par heap race: parallel RCHOL writes progress to
// std::cout from many threads while the adapter silences it with discard_cout.
// With the old shared std::ostringstream this crashed or corrupted the heap.
// The guard must survive the same load, discard everything, and restore the
// previous buffer on normal exit and when the silenced code throws.
#include "discard_streambuf.h"

#include <cstdio>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <vector>

int main() {
    std::ostringstream real;  // stands in for the process stdout
    std::streambuf* original = std::cout.rdbuf(real.rdbuf());
    {
        discard_cout quiet;
        std::vector<std::thread> workers;
        for (int t = 0; t < 64; ++t)
            workers.emplace_back([t] {
                for (int i = 0; i < 20000; ++i)
                    std::cout << "depth(separator): " << t << " nztotal " << i * 1.5 << "\n";
            });
        for (auto& w : workers) w.join();
    }
    std::cout << "after-threads;";
    try {
        discard_cout quiet;
        std::cout << "discarded before throw";
        throw std::runtime_error("silenced code failed");
    } catch (const std::runtime_error&) {
    }
    std::cout << "after-throw";
    std::cout.rdbuf(original);
    if (real.str() != "after-threads;after-throw") {
        std::fprintf(stderr, "unexpected stdout: '%s'\n", real.str().c_str());
        return 1;
    }
    std::puts("discard_cout: 64 concurrent writers discarded; stdout restored after exit and throw");
    return 0;
}
