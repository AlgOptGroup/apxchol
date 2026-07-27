# matlab-deps:r2026a + a C/C++ toolchain so MATLAB's `mex` can compile the CMG MEX.
# MATLAB itself is bind-mounted at runtime (-v <host MATLAB>:/opt/MATLAB/R2026a:ro);
# the host license works via `docker run --network=host` (license MAC matches).
FROM mathworks/matlab-deps:r2026a
RUN apt-get update \
 && apt-get install -y --no-install-recommends gcc g++ make ca-certificates \
 && rm -rf /var/lib/apt/lists/*
