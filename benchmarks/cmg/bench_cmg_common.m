function bench_cmg_common(operator_path, rhs_path, output_dir, retained_repeats, tol, maxiter, threads)
% Canonical CMG on explicit unchanged A,b: one full warmup and retained rebuilds.
% Component certification/slicing and hierarchy creation are timed as setup.
% Reconstruction and destruction are timed as solve; true residuals and I/O are
% outside those intervals. No shift, RHS regeneration, or residual projection.
    if retained_repeats < 1 || fix(retained_repeats) ~= retained_repeats
        error('cmg_common:repeats', 'At least one retained repeat is required');
    end
    maxNumCompThreads(threads);
    cmg_root = getenv('CMG_ROOT');
    addpath(cmg_root, fullfile(cmg_root, 'matlab', 'cmg'), fullfile(cmg_root, 'matlab', 'cmg', 'mex'));
    A = read_common_mtx(operator_path);
    b = full(read_common_mtx(rhs_path));
    if size(A,1) ~= size(A,2) || ~isequal(size(b), [size(A,1),1]) || ...
            any(~isfinite(nonzeros(A))) || any(~isfinite(b)) || ~isequal(A,A')
        error('cmg_common:input', 'Expected an exactly symmetric finite operator and finite explicit column RHS');
    end
    n = size(A,1); nza = nnz(A); bnorm = norm(b);
    for repetition = 0:retained_repeats
        rec = struct('repetition', repetition, 'retained', repetition > 0, ...
            'status', 'failed', 'n', n, 'nnz', nza, 'effective_threads', maxNumCompThreads, ...
            'matlab_version', version, 'setup_calls', 0);
        try
            started = tic;
            [keep, components, grounded] = common_component_gauge_mex(A,b);
            rec.components = components; rec.grounded_components = grounded;
            if grounded > 0
                S = A(keep,keep); sb = b(keep);
            else
                S = A; sb = b;
            end
            rec.solve_n = size(S,1);
            rec.adaptation_s = toc(started);
            % Request the canonical public one-output contract: unsupported
            % inputs return before upstream initializes its optional H output.
            pfun = cmg_precondition(S);
            rec.setup_s = toc(started); rec.setup_calls = 1;
            if isempty(pfun)
                rec.status = 'n/a'; rec.reason = 'canonical cmg_precondition returned an empty hierarchy';
                rec.solve_s = 0; rec.total_s = rec.setup_s;
            else
                started = tic;
                [sx, pcg_flag, reported_relres, iterations] = pcg(S,sb,tol,maxiter,pfun);
                if grounded > 0
                    x = zeros(n,1); x(keep) = sx;
                else
                    x = sx;
                end
                clear pfun S sb sx keep;
                rec.solve_s = toc(started); rec.total_s = rec.setup_s + rec.solve_s;
                rec.pcg_flag = pcg_flag; rec.reported_relres = reported_relres; rec.iters = iterations;
                if bnorm > 0
                    rec.rel_res = norm(A*x-b)/bnorm;
                else
                    rec.rel_res = norm(A*x-b);
                end
                rec.status = 'not_converged';
                if pcg_flag == 0 && isfinite(rec.rel_res) && rec.rel_res <= tol
                    rec.status = 'complete';
                end
                fp = fopen(fullfile(output_dir,sprintf('rep%d.solution.f64',repetition)), 'w', 'ieee-le');
                if fp < 0; error('cmg_common:output','Cannot open solution'); end
                count = fwrite(fp,x,'double'); fclose(fp);
                if count ~= n; error('cmg_common:output','Incomplete solution'); end
            end
        catch err
            rec.status = 'failed'; rec.error_id = err.identifier; rec.error = err.message;
        end
        write_record(fullfile(output_dir,sprintf('rep%d.json',repetition)), rec);
        fprintf('CMG_COMMON repetition=%d status=%s\n', repetition, rec.status);
        if ~strcmp(rec.status,'complete'); return; end
    end
end

function write_record(path, record)
    fp = fopen(path,'w');
    if fp < 0; error('cmg_common:output','Cannot open result'); end
    fprintf(fp,'%s\n',jsonencode(record)); fclose(fp);
end

function A = read_common_mtx(path)
% The shared exporter emits real coordinate or array MatrixMarket. Preserve
% all represented values, including diagonal slack; expand symmetry once.
    fp = fopen(path,'r');
    if fp < 0; error('cmg_common:input','Cannot open %s',path); end
    cleanup = onCleanup(@() fclose(fp));
    header = strsplit(lower(strtrim(fgetl(fp))));
    if numel(header) ~= 5 || ~strcmp(header{1},'%%matrixmarket') || ...
            ~strcmp(header{2},'matrix') || ~any(strcmp(header{4},{'real','integer'})) || ...
            ~any(strcmp(header{5},{'general','symmetric'}))
        error('cmg_common:input','Unsupported MatrixMarket header in %s',path);
    end
    line = fgetl(fp);
    while ischar(line) && (isempty(strtrim(line)) || startsWith(strtrim(line),'%'))
        line = fgetl(fp);
    end
    dims = sscanf(line,'%d');
    if strcmp(header{3},'coordinate')
        if numel(dims) ~= 3; error('cmg_common:input','Invalid coordinate dimensions'); end
        data = fscanf(fp,'%g',[3,dims(3)]);
        if size(data,2) ~= dims(3) || size(data,1) ~= 3
            error('cmg_common:input','Truncated coordinate data');
        end
        if any(data(1,:) < 1 | data(1,:) > dims(1) | fix(data(1,:)) ~= data(1,:)) || ...
                any(data(2,:) < 1 | data(2,:) > dims(2) | fix(data(2,:)) ~= data(2,:))
            error('cmg_common:input','Invalid coordinate indices');
        end
        A = sparse(data(1,:), data(2,:), data(3,:), dims(1), dims(2));
        clear data;
        if strcmp(header{5},'symmetric')
            if dims(1) ~= dims(2); error('cmg_common:input','Nonsquare symmetric data'); end
            D = spdiags(diag(A),0,dims(1),dims(2));
            off = A-D; clear A;
            A = off+off'+D; % never double a diagonal before subtracting it
        end
    elseif strcmp(header{3},'array') && strcmp(header{5},'general')
        if numel(dims) ~= 2; error('cmg_common:input','Invalid array dimensions'); end
        A = fscanf(fp,'%g',prod(dims));
        if numel(A) ~= prod(dims); error('cmg_common:input','Truncated array data'); end
        A = reshape(A,dims(1),dims(2));
    else
        error('cmg_common:input','Unsupported MatrixMarket storage');
    end
    if ~isempty(strtrim(fread(fp,Inf,'*char')'))
        error('cmg_common:input','Unexpected trailing MatrixMarket data');
    end
end
