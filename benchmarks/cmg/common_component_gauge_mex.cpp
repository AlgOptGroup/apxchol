// A no-copy view of MATLAB CSC arrays for the shared native CMG gauge rule.
// Compile with MATLAB mex -R2018a.
#include "mex.h"
#include "component_gauge.h"
#include <climits>

namespace {
struct MatlabSparse {
    const mxArray* matrix;
    int rows() const { return static_cast<int>(mxGetM(matrix)); }
    int cols() const { return static_cast<int>(mxGetN(matrix)); }
    struct InnerIterator {
        const mwIndex* indices;
        const double* values;
        mwIndex position, end;
        InnerIterator(const MatlabSparse& a, int column)
            : indices(mxGetIr(a.matrix)), values(mxGetDoubles(a.matrix)),
              position(mxGetJc(a.matrix)[column]), end(mxGetJc(a.matrix)[column+1]) {}
        explicit operator bool() const { return position < end; }
        void operator++() { ++position; }
        mwIndex row() const { return indices[position]; }
        double value() const { return values[position]; }
    };
};
struct MatlabVector {
    const mxArray* vector;
    int size() const { return static_cast<int>(mxGetNumberOfElements(vector)); }
    double operator[](int index) const { return mxGetDoubles(vector)[index]; }
};
}

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    if (nrhs != 2 || nlhs != 3)
        mexErrMsgIdAndTxt("cmg_common:gaugeArguments", "Expected [keep, components, grounded] = common_component_gauge_mex(A,b)");
    if (!mxIsSparse(prhs[0]) || !mxIsDouble(prhs[0]) || mxIsComplex(prhs[0]) ||
        mxGetM(prhs[0]) != mxGetN(prhs[0]) || mxGetM(prhs[0]) > INT_MAX ||
        mxGetJc(prhs[0])[mxGetN(prhs[0])] > INT_MAX ||
        mxIsSparse(prhs[1]) || !mxIsDouble(prhs[1]) || mxIsComplex(prhs[1]) ||
        mxGetNumberOfElements(prhs[1]) != mxGetM(prhs[0]))
        mexErrMsgIdAndTxt("cmg_common:gaugeInput", "Expected square real sparse double A, dense double b, and signed-32-bit dimensions/entry count");
    try {
        const auto gauge = cmg_native::select_component_gauge(MatlabSparse{prhs[0]}, MatlabVector{prhs[1]});
        plhs[0] = mxCreateDoubleMatrix(gauge.retained.size(), 1, mxREAL);
        auto* keep = mxGetDoubles(plhs[0]);
        for (std::size_t i = 0; i < gauge.retained.size(); ++i) keep[i] = gauge.retained[i] + 1;
        plhs[1] = mxCreateDoubleScalar(gauge.components);
        plhs[2] = mxCreateDoubleScalar(gauge.grounded);
    } catch (const std::exception& error) {
        mexErrMsgIdAndTxt("cmg_common:gaugeFailure", "%s", error.what());
    }
}
