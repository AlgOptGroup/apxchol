#pragma once
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <map>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>
#include <ranges>

namespace apxchol {

/// Accumulating per-label timer with hierarchical grouping.
///
/// Maintains a persistent timing tree.  Use descend()/ascend() to set
/// the current scope, then operator()(label) to record elapsed time
/// under that scope.  report() prints the full tree with per-node
/// subtotals; total(prefix) returns a subtree's total.
///
/// Example:
///   cp.descend("setup");
///   // ... work ...
///   cp("build_adj");              // records under setup.build_adj
///   // ... work ...
///   cp("eliminate");              // records under setup.eliminate
///   cp.ascend();
struct checkpoint {
    using Clock = std::chrono::high_resolution_clock;

    struct node {
        double time = 0.0;
        std::map<std::string, node> children;
    };

    checkpoint() : last_(Clock::now()), cur_(&root_) {}

    /// Record elapsed time since last call under `label` at the current scope.
    void operator()(const std::string& label) {
        auto now = Clock::now();
        cur_->children[label].time +=
            std::chrono::duration<double>(now - last_).count();
        last_ = now;
    }

    /// Descend into a child scope.
    void descend(const std::string& name) {
        stack_.push_back(cur_);
        cur_ = &cur_->children[name];
    }

    /// Return to the parent scope.
    void ascend() {
        cur_ = stack_.back();
        stack_.pop_back();
    }

    /// Reset the clock without recording.
    void tick() { last_ = Clock::now(); }

    /// Sum of all times in the subtree rooted at the given dotted path.
    double total(const std::string& path) const {
        const node* n = find(path);
        if (!n) return 0.0;
        return subtotal(*n);
    }

    /// Human-readable breakdown with recursive multi-level grouping.
    void report(std::ostream& os) const {
        struct line { int depth; std::string name; double subtotal; };
        std::vector<line> lines;

        auto collect = [&](this auto const& collect, const node& n,
                           int depth) -> double {
            double s = n.time;
            for (const auto& [name, c] : n.children) {
                const double child_subtotal = collect(c, depth + 1);
                lines.push_back({depth, name, child_subtotal});
                s += child_subtotal;
            }
            return s;
        };
        double grand = collect(root_, 0);

        os << std::fixed;
        for (const auto& [depth, name, st] : lines | std::views::reverse) {
            double pct = 100.0 * st / grand;
            int indent = 2 * depth;
            int name_w = std::max(1, 16 - indent);
            os << std::string(indent, ' ')
               << std::left  << std::setw(name_w) << name
               << std::right << std::setprecision(3) << std::setw(10)
               << (st * 1000) << " ms"
               << std::setprecision(1) << std::setw(7) << pct << "%\n";
        }
        os << std::left << std::setw(16) << "total"
           << std::right << std::setprecision(3) << std::setw(10)
           << (grand * 1000) << " ms\n";
    }

    /// Convenience overload returning a string.
    std::string report() const {
        std::ostringstream os;
        report(os);
        return os.str();
    }

private:
    /// Compute total time in a subtree (node's own time + all descendants).
    static double subtotal(const node& n) {
        double s = n.time;
        for (const auto& [_, c] : n.children)
            s += subtotal(c);
        return s;
    }

    /// Walk a dotted path from root.  Returns nullptr if not found.
    const node* find(const std::string& path) const {
        const node* n = &root_;
        std::string_view sv(path);
        while (!sv.empty()) {
            auto dot = sv.find('.');
            auto key = std::string(sv.substr(0, dot));
            auto it = n->children.find(key);
            if (it == n->children.end()) return nullptr;
            n = &it->second;
            if (dot == std::string_view::npos) break;
            sv.remove_prefix(dot + 1);
        }
        return n;
    }

    Clock::time_point last_;
    node root_;
    node* cur_;
    std::vector<node*> stack_;
};

} // namespace apxchol
