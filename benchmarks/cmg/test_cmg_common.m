function test_cmg_common()
% Functional tests for the no-copy binding to the shared exact gauge rule.
    A = sparse([1 -1 0 0 0; -1 1 0 0 0; 0 0 2 -2 0; 0 0 -2 2 0; 0 0 0 0 0]);
    v = [1;3;-2;5;0];
    check(A,A*v,3,3,[2;4]);
    A(5,5)=0.5; check(A,A*v,3,2,[2;4;5]);
    A(1,1)=1+eps; check(A,A*v,3,1,[1;2;4;5]);
    A(1,1)=1;
    b=A*v; b(1)=b(1)+1;
    caught=false;
    try
        [~,~,~]=common_component_gauge_mex(A,b);
    catch error
        caught=strcmp(error.identifier,'cmg_common:gaugeFailure');
    end
    assert(caught,'Incompatible component RHS was accepted');
    check(sparse([1 -1;-1 1]),[1;-1],1,0,zeros(0,1));
    A=speye(3); A(1,2)=0.1; A(2,1)=0.1;
    check(A,[1;2;3],2,0,zeros(0,1));
    % A positive represented slack hidden by ordinary extended-precision
    % cancellation must not be erased by component grounding.
    small=4096; last=small+2; n=small+3; w=pow2(1.5,60); q=2^-4+2^-16;
    weights=[repmat(q,small,1);w-512];
    A=sparse([1;ones(last-1,1);(2:last)';(2:last)'], ...
             [1;(2:last)';ones(last-1,1);(2:last)'], ...
             [w;-weights;-weights;weights],n,n);
    [keep,components,grounded]=common_component_gauge_mex(A,zeros(n,1));
    assert(components==2 && grounded==1 && keep(1)==1 && numel(keep)==n-1);
    fprintf('MATLAB component gauge: 7/7 contract cases passed\n');
end

function check(A,b,components,grounded,expected)
    [keep,c,g]=common_component_gauge_mex(A,b);
    assert(c==components && g==grounded && isequal(keep,expected));
    if grounded>0
        x=zeros(size(b)); x(keep)=A(keep,keep)\b(keep);
        assert(norm(A*x-b)<1e-12*max(1,norm(b)));
    end
end
