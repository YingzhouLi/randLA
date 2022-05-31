% Test with randomized SVD
addpath('../src')

r = 20;
A = randn(2000,r)*randn(r,2000)/10+randn(2000);

tic;
[Utrue, Strue, Vtrue] = svd(A);
toc;

tic;
[U, S, V] = rsvd(A,r);
toc;

fprintf("Error in singular values: %.2e\n", ...
    norm(diag(Strue(1:r,1:r))-diag(S)));
fprintf("Error in left singular vectors: %.2e\n", ...
    norm(abs(U'*Utrue(:,1:r))-eye(r)));
fprintf("Error in right singular vectors: %.2e\n", ...
    norm(abs(V'*Vtrue(:,1:r))-eye(r)));