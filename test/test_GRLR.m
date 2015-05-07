% Test with Gaussian Random low rank matrix
addpath('../src')

Niter = 100;
N = 300;
tol = 1e-4;
r = 5;

relerr = NaN(3,Niter);
time   = NaN(3,Niter);

for iter = 1:Niter
    A = randn(N,r+10);
    A = A*A';
    tic;
    [Usvd,Ssvd,~] = svd(A);
    time(3,iter) = toc;
    relerr(3,iter) = Ssvd(r+1,r+1)/Ssvd(1,1);
    
    tic;
    [U,S] = Uni_Sampling_mat(A,tol,r);
    time(1,iter) = toc;
    relerr(1,iter) = norm(A-U*S*U')/Ssvd(1,1);
    
    tic;
    [U,S] = PQR_Sampling_mat(A,tol,r);
    time(2,iter) = toc;
    relerr(2,iter) = norm(A-U*S*U')/Ssvd(1,1);
end

figure(1)
plot(relerr','.');
title('relative error');
legend('Uni Sampling','PQR Sampling','SVD');

figure(2)
plot(time','.');
title('time');
legend('Uni Sampling','PQR Sampling','SVD');