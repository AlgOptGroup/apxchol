#include <Eigen/Core>

int apxchol_consumer_core_alignment() {
    return EIGEN_DEFAULT_ALIGN_BYTES;
}

// Allocated by the core's Eigen and freed by the consumer's.
Eigen::VectorXd apxchol_consumer_core_vector(Eigen::Index n) {
    return Eigen::VectorXd::Ones(n);
}
