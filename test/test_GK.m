% Test with Gaussian Kernel
addpath('../src')

Niter = 100;
N = 300;
dim = 10;
tol = 1e-4;
r = 5;
h = 3;
fun = @(x,y)exp(-pdist2(x,y).^2/h^2).*(exp(-pdist2(x,y).^2/h^2)>0.1);

relerr = NaN(4,Niter);
time   = NaN(4,Niter);

for iter = 1:Niter
    X = randn(N,dim);
    A = fun(X,X);
    tic;
    [Usvd,Ssvd,~] = svd(A);
    time(4,iter) = toc;
    relerr(4,iter) = Ssvd(r+1,r+1)/Ssvd(1,1);
    
    tic;
    [U,S] = Uni_Sampling_fun(fun,X,tol,r);
    time(1,iter) = toc;
    relerr(1,iter) = norm(A-U*S*U')/Ssvd(1,1);
    
    tic;
    [U,S] = PQR_Sampling_fun(fun,X,tol,r);
    time(2,iter) = toc;
    relerr(2,iter) = norm(A-U*S*U')/Ssvd(1,1);
    
    tic;
    [U,S] = Kmeans_Sampling_fun(fun,X,tol,r);
    time(3,iter) = toc;
    relerr(3,iter) = norm(A-U*S*U')/Ssvd(1,1);
end

figure(1)
plot(relerr','.');
title('relative error');
legend('Uni Sampling','PQR Sampling','Kmeans Sampling','SVD');

figure(2)
plot(time','.');
title('time');
legend('Uni Sampling','PQR Sampling','Kmeans Sampling','SVD');

% REMARK: When h is very small such that the matrix A=fun(X,X) is close to
% identity, the approximation are very bad!