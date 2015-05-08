function [U,S] = PQR_Sampling_mat(A,tol,r)

Nx = size(A,1);

tR = 3*r;

if( Nx==0 )
    U = zeros(Nx,0);
    S = zeros(0,0);
    return;
end

if( tR < Nx )
    Idx = [];
    for iter = 1:2
        idx = randsample(Nx,tR);
        idx = union(idx,Idx);
        M = A(idx,:);
        [~,R,E] = qr(M,0);
        Idx = E(find(abs(diag(R))>tol*abs(R(1)))<=tR);
    end
else
    Idx = 1:Nx;
end

M = A(:,Idx);

[Q,~,~] = qr(M,0);

if( tR < Nx )
    idx = randsample(Nx,tR);
    idx = union(idx,Idx);
else
    idx = 1:Nx;
end

pQ = pinv(Q(idx,:));
MM = A(idx,idx);
MD = pQ * MM* pQ';
[U,S,~] = svd(MD,0);
if ~isempty(S)
    idx = find(find(diag(S)>tol*S(1,1))<=r);
    U = Q*U(:,idx);
    S = S(idx,idx);
else
    U = zeros(Nx,0);
    S = zeros(0,0);
end

end

