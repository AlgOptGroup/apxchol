#include <Eigen/Core>

// The alignment the core may assume for Eigen data, which consumers must share.
int apxchol_consumer_core_alignment() {
    return EIGEN_MAX_ALIGN_BYTES;
}

// Allocated by the core's Eigen and freed by the consumer's.
Eigen::VectorXd apxchol_consumer_core_vector(Eigen::Index n) {
    return Eigen::VectorXd::Ones(n);
}

// Allocated by the consumer's Eigen and freed by the core's.
double apxchol_consumer_core_take(Eigen::VectorXd v) {
    const double sum = v.sum();
    v.resize(0);
    return sum;
}
