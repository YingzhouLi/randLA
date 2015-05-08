function [U,S,V] = Uni_Sampling_mat(A,tol,r)

Nx = size(A,1);
Np = size(A,2);

tR = 3*r;

if( Nx==0 || Np==0 )
    U = zeros(Nx,0);
    S = zeros(0,0);
    V = zeros(Np,0);
    return;
end

if( tR < Np && tR < Nx )
    Ridx = randsample(Nx,tR);
    Cidx = randsample(Np,tR);
else
    Ridx = 1:Nx;
    Cidx = 1:Np;
end

MR = A(Ridx,:);
MC = A(:,Cidx);

[QC,~,~] = qr(MC,0);
[QR,~,~] = qr(MR',0);

if( tR < Np && tR < Nx )
    rs = randsample(Nx,tR);
    rs = union(rs,Ridx);
    cs = randsample(Np,tR);
    cs = union(cs,Cidx);
else
    rs = 1:Nx;
    cs = 1:Np;
end

M1 = QC(rs,:);
M2 = QR(cs,:);
M3 = A(rs,cs);
MD = pinv(M1) * (M3* pinv(M2'));
[U,S,V] = svd(MD,0);
if ~isempty(S)
    idx = find(find(diag(S)>tol*S(1,1))<=r);
    U = QC*U(:,idx);
    S = S(idx,idx);
    V = QR*V(:,idx);
else
    U = zeros(Nx,0);
    S = zeros(0,0);
    V = zeros(Np,0);
end

end