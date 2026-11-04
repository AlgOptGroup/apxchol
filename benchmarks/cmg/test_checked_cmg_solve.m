function test_checked_cmg_solve()
A=sparse([1 -1 0 0;-1 1 0 0;0 0 2 -2;0 0 -2 2]);
b=[1;-1;2;-2]; keep=[2;4]; S=A(keep,keep); sb=b(keep);
[x,r]=checked_cmg_solve(A,b,S,sb,keep,@(x)x,1e-10,10);
assert(r.rel_res<1e-10 && norm(A*x-b)<1e-10 && r.solve_s>=r.stop_check_s);
[x,r]=checked_cmg_solve(sparse(1,1),0,sparse(1,1),0,[],@(x)x,1e-10,10);
assert(r.rel_res==0 && x==0);
A=blkdiag(sparse(1),sparse([1 -1;-1 1]));b=[1;1;-1];keep=[1;3];
[x,r]=checked_cmg_solve(A,b,A(keep,keep),b(keep),keep,@(x)x,1e-10,10);
assert(r.rel_res<1e-10 && norm(A*x-b)<1e-10);
mockdir = fullfile(fileparts(mfilename('fullpath')),'tests','checked_solve_mock');
addpath(mockdir,'-begin');
cleanup = onCleanup(@() rmpath(mockdir));
clear pcg;
global cmg_test_mode cmg_test_calls;
for mode={'optimistic','zero_progress','breakdown','budget'}
 cmg_test_mode=mode{1};cmg_test_calls={};
 [x,r]=checked_cmg_solve(speye(2),[1;2],speye(2),[1;2],[],@(x)x,1e-8,5);
 assert(r.solve_s>=r.stop_check_s);
 switch cmg_test_mode
 case 'optimistic'
  assert(r.rel_res==0 && r.solve_passes==2 && r.iters==2);
  assert(cmg_test_calls{2}{2}==4 && norm(cmg_test_calls{2}{3}-[.5;1])==0);
 case 'zero_progress';assert(r.solve_passes==8 && r.iters==0 && r.rel_res==1);
 case 'breakdown';assert(r.solve_passes==1 && r.iters==1 && r.rel_res==1);
 case 'budget';assert(r.solve_passes==1 && r.iters==5 && r.rel_res==1);
 end
 fprintf('%s passes=%d iterations=%d residual=%g\n',cmg_test_mode,r.solve_passes,r.iters,r.rel_res);
end
fprintf('7 checked-solve cases passed\n');
end
