function [U,S,V] = PQR_Sampling_mat(A,tol,r)

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
    
    Ridx = [];
    Cidx = [];
    for iter = 1:2
        %get columns
        rs = randsample(Nx,tR);
        rs = union(rs,Ridx);
        M2 = A(rs,:);
        [~,R2,E2] = qr(M2,0);
        Cidx = E2(find(abs(diag(R2))>tol*abs(R2(1)))<=tR);

        %get rows
        cs = randsample(Np,tR);
        cs = union(cs,Cidx);
        M1 = A(:,cs);
        [~,R1,E1] = qr(M1',0);
        Ridx = E1(find(abs(diag(R1))>tol*abs(R1(1)))<=tR);
    end
    
else
    Ridx = 1:Nx;
    Cidx = 1:Np;
end

MR = A(Ridx,:);
MC = A(:,Cidx);

[QC,~,~] = qr(MC,0);
[QR,~,~] = qr(MR',0);

if( tR < Np && tR < Nx )
    cs = randsample(Np,tR);
    cs = union(cs,Cidx);
    rs = randsample(Nx,tR);
    rs = union(rs,Ridx);
else
    cs = 1:Np;
    rs = 1:Nx;
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

