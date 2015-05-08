function [U,S] = Uni_Sampling_fun(fun,x,tol,r)

Nx = size(x,1);

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

M = fun(x,x(Idx,:));

[Q,~,~] = qr(M,0);

if( tR < Nx )
    idx = randsample(Nx,tR);
    idx = union(idx,Idx);
else
    idx = 1:Nx;
end

pMQ = pinv(Q(idx,:));
MM = fun(x(idx,:),x(idx,:));
MD = pMQ * MM* pMQ';
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
