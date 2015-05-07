function [U,S] = Uni_Sampling_mat(A,tol,r)

Nx = size(A,1);

tR = 3*r;

if( Nx==0 )
    U = zeros(Nx,0);
    S = zeros(0,0);
    return;
end

if( tR < Nx )
    Idx = randsample(Nx,tR);
else
    Idx = 1:Nx;
end

M = A(:,Idx);

[Q,~,~] = qr(M,0);

if( tR < Nx )
    idx = randsample(Nx,5);
    idx = union(idx,Idx);
else
    idx = 1:Nx;
end

pMQ = pinv(Q(idx,:));
MM = A(idx,idx);
MD = pMQ * MM* pMQ';
[U,S,V] = svd(MD,0);
if ~isempty(S)
    idx = find(find(diag(S)>tol*S(1,1))<=r);
    U = Q*U(:,idx);
    S = S(idx,idx);
else
    U = zeros(Nx,0);
    S = zeros(0,0);
end

end