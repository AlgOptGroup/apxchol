function [x, stop] = checked_cmg_solve(A, b, S, sb, keep, pfun, tol, maxiter)
% Keep the canonical hierarchy across bounded warm-started PCG retries.
% Original-system checks and reconstruction belong to the solve interval.
    started = tic;
    sx = zeros(size(sb)); x = zeros(size(b));
    bn = norm(b); if bn == 0; bn = 1; end
    stop = struct('iters',0,'solve_passes',0,'stop_check_s',0, ...
                  'pcg_flag',0,'reported_relres',Inf,'rel_res',Inf);
    request = tol;
    while true
        if stop.iters < maxiter
            remaining = maxiter - stop.iters;
            [sx, flag, reported, used] = pcg(S,sb,request,remaining,pfun,[],sx);
            if ~isscalar(used) || ~isfinite(used) || used < 0 || ...
                    fix(used) ~= used || used > remaining
                error('cmg_common:budget','Native PCG violated its iteration budget');
            end
            stop.iters = stop.iters + used;
            stop.solve_passes = stop.solve_passes + 1;
            stop.pcg_flag = flag; stop.reported_relres = reported;
        end
        if isempty(keep)
            x = sx;
        else
            x(:) = 0; x(keep) = sx;
        end
        checked = tic;
        stop.rel_res = norm(A*x-b)/bn;
        stop.stop_check_s = stop.stop_check_s + toc(checked);
        if ~isfinite(stop.rel_res) || stop.rel_res <= tol || ...
                stop.iters >= maxiter || stop.solve_passes >= 8 || ...
                any(stop.pcg_flag == [2,4])
            break;
        end
        next = max(eps, request * 0.1);
        if next >= request; break; end
        request = next;
    end
    stop.solve_s = toc(started);
end
