function [x,flag,relres,used] = pcg(A,b,tol,maxit,M,M2,x0)
    global cmg_test_mode cmg_test_calls;
    cmg_test_calls{end+1} = {tol,maxit,x0};
    flag=0; relres=0; used=1;
    switch cmg_test_mode
        case 'optimistic'
            if numel(cmg_test_calls)==1; x=b/2; else; x=A\b; end
        case 'zero_progress'
            x=x0; used=0;
        case 'breakdown'
            x=x0; flag=4;
        case 'budget'
            x=x0; used=maxit;
        otherwise
            error('unknown mode');
    end
end
