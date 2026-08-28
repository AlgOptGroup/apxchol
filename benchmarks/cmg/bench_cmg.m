% bench_cmg.m — CMG-preconditioned PCG bench driver for Octave or MATLAB
% Usage (Octave):  octave --no-gui --quiet --eval "bench_cmg('/path/to/A.mtx', 1e-8, 500, 42)"
% Usage (MATLAB):  matlab -batch "bench_cmg('/path/to/A.mtx', 1e-8, 500, 42)"
%
% Emits ONE CSV row to stdout in the bench schema:
%   solver,graph,n,nnz,setup_s,solve_s,total_s,iters,rel_res,fillin,us_per_nnz
function bench_cmg(mtx_path, tol, maxiter, seed, reg_rel, as_operator)
    % Default argument values
    if nargin < 2 || isempty(tol);     tol     = 1e-8; end
    if nargin < 3 || isempty(maxiter); maxiter = 500;  end
    if nargin < 4 || isempty(seed);    seed    = 42;   end
    % reg_rel > 0: add reg_rel*mean(diag)*I to make the Laplacian strictly SDDM
    % (unified regularization), solve the FULL system with no pinning -- so CMG
    % is measured against the same operator as every other solver.
    if nargin < 5 || isempty(reg_rel); reg_rel = 0;    end
    % as_operator = 1: read mtx_path AS AN OPERATOR -- every stored value kept,
    % the diagonal included -- and solve it as it stands: no pin, no eps*I. This
    % is what a kind=operator matrix (apache2, G3_circuit, thermal2,
    % parabolic_fem, the IPM matrices) needs. The Laplacian reader below REBUILDS
    % the diagonal as the degree, which for such a matrix throws away the
    % diagonal excess and hands CMG a DIFFERENT system than every other solver in
    % the cell. as_operator = 0 keeps the Laplacian reading, which for a
    % kind=graph dump reproduces exactly the L that was dumped.
    if nargin < 6 || isempty(as_operator); as_operator = 0; end

    % Locate cmg-solver
    cmg_root = getenv('CMG_ROOT');
    if isempty(cmg_root)
        cmg_root = fullfile(getenv('HOME'), 'cmg-solver');
    end
    addpath(cmg_root);
    addpath(fullfile(cmg_root, 'matlab', 'cmg'));
    addpath(fullfile(cmg_root, 'matlab', 'cmg', 'mex'));

    % Load the matrix from Matrix Market file.
    % We use a built-in reader so no external toolboxes are required.
    if as_operator
        L = read_mtx_as_operator(mtx_path);
    else
        L = read_mtx_as_laplacian(mtx_path);
    end
    % A full-rank operator has no null space: solve it whole, and do not
    % mean-centre its right-hand side or its residual.
    is_singular = ~as_operator;
    full_system = as_operator || (reg_rel > 0);

    n = size(L, 1);
    nnz_L = nnz(L);
    [~, graph_name, ~] = fileparts(mtx_path);

    % Skip matrices too small for CMG (it silently returns empty for n<500).
    % n/a, not a failure: emit the sentinel row (iters = rel_res = -1) plus the
    % reason on stderr, which the runner lifts into the cell.
    if n < 500
        emit_na(graph_name, n, nnz_L, 0, sprintf( ...
            ['n=%d is below the ~500 nodes CMG needs: cmg_precondition returns an ' ...
             'empty hierarchy below that size'], n));
        return;
    end

    % Generate structured RHS: b = L*g, mean-centered, normalized.
    % Matches the C++ make_rhs() convention.
    if exist('OCTAVE_VERSION', 'builtin')
        randn('state', double(seed));
    else
        rng(seed);
    end
    g = randn(n, 1);
    b = L * g;
    if is_singular
        b = b - mean(b);
    end
    nb = norm(b);
    if nb > 0
        b = b / nb;
    end

    % CMG requires a strictly SDD solve operator.  Start setup before the
    % solver-specific pin/slice/regularization that produces that input; loading
    % the common MatrixMarket operator and constructing the common RHS stay out.
    tic;
    if as_operator
        % Published full-rank operator: solve it as it stands. No pin (there is
        % no null space to remove) and no eps*I (it is already strictly SDDM),
        % so this is the very matrix the rest of the cell was scored on.
        Lsub = L;
        m = n;
        bsub = b;
        Lshift = L;
    elseif reg_rel > 0
        % Unified SDDM regularization: solve the FULL n×n system, no pinning.
        eps = reg_rel * mean(abs(diag(L)));
        Lsub = L + eps * speye(n);
        m = n;
        bsub = b;
        Lshift = Lsub;   % already strictly SDDM; CMG preconditions the same operator
    else
        % Pin last vertex: solve the (n-1)×(n-1) sub-system (legacy singular path).
        m = n - 1;
        Lsub = L(1:m, 1:m);
        bsub = b(1:m);
        bsub = bsub - mean(bsub);
        Lshift = Lsub + 1e-12 * speye(m);
    end

    % CMG hierarchy setup
    pfun = cmg_sdd(Lshift);
    setup_s = toc;

    % An empty preconditioner is CMG DECLINING the matrix, not a failed run — it
    % prints its own reason ("The current version of CMG does not support
    % positive off-diagonals") and returns []. Report it as n/a WITH the reason,
    % determined here from the matrix itself so the cell says something checkable.
    if isempty(pfun)
        npos = nnz(triu(Lshift, 1) > 0);
        if npos > 0
            reason = sprintf(['cmg_precondition declined this matrix: CMG does not support ' ...
                              'positive off-diagonal entries, and it has %d of them ' ...
                              '(upper triangle)'], npos);
        else
            reason = ['cmg_precondition returned an empty preconditioner (see its own ' ...
                      'message on stdout)'];
        end
        emit_na(graph_name, n, nnz_L, setup_s, reason);
        return;
    end

    % Solve
    tic;
    [x_sub, ~, ~, iters] = pcg(Lsub, bsub, tol, maxiter, pfun);
    solve_s = toc;
    total_s = setup_s + solve_s;

    % True residual.
    if full_system
        % SDDM: full system, no centering — same metric as every other solver.
        res = bsub - Lsub * x_sub;
        bnorm = norm(bsub);
    else
        % Singular Laplacian: reconstruct full x and mean-center.
        x = zeros(n, 1);
        x(1:m) = x_sub;
        x = x - mean(x);
        res = b - L * x;
        res = res - mean(res);
        bnorm = norm(b);
    end
    if bnorm > 0
        relres_true = norm(res) / bnorm;
    else
        relres_true = 1.0;
    end

    us_per_nnz = total_s / nnz_L * 1e6;

    backend = octave_or_matlab();
    fprintf('CMG+PCG [Koutis10;%s],%s,%d,%d,%e,%e,%e,%d,%e,%e,%e\n', ...
        backend, graph_name, n, nnz_L, ...
        setup_s, solve_s, total_s, iters, relres_true, 0.0, us_per_nnz);
end

% ---------------------------------------------------------------------------
% Helper: read a Matrix Market file and return it as a Laplacian.
%
% For "real" (or "integer") symmetric matrices the file already stores
% the Laplacian directly (negative off-diagonals, positive diagonal).
% The diagonal entries are discarded and the Laplacian is reconstructed
% from |w_ij| edge weights — matching the C++ load_mtx_as_adjacency logic.
%
% For "pattern" symmetric matrices all edge weights are treated as 1.
% ---------------------------------------------------------------------------
function [L, is_pattern] = read_mtx_as_laplacian(fname)
    fid = fopen(fname, 'r');
    if fid < 0
        error('Cannot open file: %s', fname);
    end

    % Parse header line
    header = fgetl(fid);
    if isempty(header) || ~strncmp(header, '%%MatrixMarket', 14)
        fclose(fid);
        error('Not a MatrixMarket file: %s', fname);
    end
    header_lc = lower(header);
    is_pattern  = ~isempty(strfind(header_lc, 'pattern'));
    is_symmetric = ~isempty(strfind(header_lc, 'symmetric'));

    % Skip comment lines
    line = fgetl(fid);
    while ischar(line) && ~isempty(line) && line(1) == '%'
        line = fgetl(fid);
    end
    if ~ischar(line)
        fclose(fid);
        error('Unexpected end of file after comments in %s', fname);
    end

    % Dimension line
    dims = sscanf(line, '%d %d %d');
    n = dims(1); nz = dims(3);

    % Read entries
    if is_pattern
        data = fscanf(fid, '%d %d', [2, nz]);
        rows = data(1, :)';
        cols = data(2, :)';
        vals = ones(nz, 1);
    else
        data = fscanf(fid, '%d %d %g', [3, nz]);
        rows = data(1, :)';
        cols = data(2, :)';
        vals = data(3, :)';
    end
    fclose(fid);

    % Build adjacency: drop diagonal entries; use |w| as edge weight.
    off  = (rows ~= cols);
    ri   = rows(off);
    ci   = cols(off);
    wi   = abs(vals(off));

    if is_symmetric
        % Lower triangle → mirror to upper
        ri2 = [ri; ci];
        ci2 = [ci; ri];
        wi2 = [wi; wi];
    else
        ri2 = ri; ci2 = ci; wi2 = wi;
    end

    % Laplacian = D - W
    deg = accumarray(ri2, wi2, [n, 1]);
    W   = sparse(ri2, ci2, wi2, n, n);
    D   = spdiags(deg, 0, n, n);
    L   = D - W;
end

% ---------------------------------------------------------------------------
% Helper: emit the n/a sentinel row (iters = rel_res = -1, which the runner's
% classify() turns into an "n/a" cell) plus the reason on stderr, tagged so
% cmg_matlab_runner.py can lift it into the cell. An unsupported combination has
% to be visible as unsupported, not as a crash and not as not_converged.
% ---------------------------------------------------------------------------
function emit_na(graph_name, n, nnz_L, setup_s, reason)
    fprintf(2, '[n/a] cmg on %s: %s\n', graph_name, reason);
    fprintf('CMG+PCG [Koutis10;%s] [n/a],%s,%d,%d,%e,0,%e,-1,-1,0,0\n', ...
        octave_or_matlab(), graph_name, n, nnz_L, setup_s, setup_s);
end

% ---------------------------------------------------------------------------
% Helper: read a Matrix Market file AS AN OPERATOR.
%
% Every stored entry keeps its published value, the diagonal included, and a
% `symmetric` header is expanded into both triangles. The mirror image of the
% C++ load_mtx_as_operator and of load_mtx_operator in
% benchmarks/julia/bench_laplacians.jl, so a kind=operator matrix reaches CMG as
% the same system every other solver in the cell was given.
% ---------------------------------------------------------------------------
function A = read_mtx_as_operator(fname)
    fid = fopen(fname, 'r');
    if fid < 0
        error('Cannot open file: %s', fname);
    end

    header = fgetl(fid);
    if isempty(header) || ~strncmp(header, '%%MatrixMarket', 14)
        fclose(fid);
        error('Not a MatrixMarket file: %s', fname);
    end
    header_lc = lower(header);
    if ~isempty(strfind(header_lc, 'pattern'))
        fclose(fid);
        error('%s is a `pattern` file: it carries no values, so it is not an operator', fname);
    end
    is_symmetric = ~isempty(strfind(header_lc, 'symmetric'));

    line = fgetl(fid);
    while ischar(line) && ~isempty(line) && line(1) == '%'
        line = fgetl(fid);
    end
    if ~ischar(line)
        fclose(fid);
        error('Unexpected end of file after comments in %s', fname);
    end

    dims = sscanf(line, '%d %d %d');
    n = dims(1); nz = dims(3);

    data = fscanf(fid, '%d %d %g', [3, nz]);
    fclose(fid);
    rows = data(1, :)';
    cols = data(2, :)';
    vals = data(3, :)';

    if is_symmetric
        off = (rows ~= cols);
        ri = rows(off); ci = cols(off); vi = vals(off);
        rows = [rows; ci];
        cols = [cols; ri];
        vals = [vals; vi];
    end
    A = sparse(rows, cols, vals, n, n);   % duplicates are summed, as MatrixMarket specifies
end

% ---------------------------------------------------------------------------
function s = octave_or_matlab()
    if exist('OCTAVE_VERSION', 'builtin')
        s = 'octave';
    else
        s = 'matlab';
    end
end
