function [U,S,V] = Kmeans_Sampling_fun(fun,x,p,tol,r)

    function center = eff_kmeans(data, m, MaxIter)
        n = size(data,1);
        center = data(randsample(n,m),:);
        for i = 1:MaxIter
            nul = zeros(m,1);
            [~,idx] = min(pdist2(center,data,'euclidean'));
            for j = 1:m
                dex = find(idx == j);
                l = length(dex);
                cltr = data(dex,:);
                if l > 1
                    center(j,:) = mean(cltr);
                elseif l == 1
                    center(j,:) = cltr;
                else
                    nul(j) = 1;
                end
            end
            dex = find(nul == 0);
            m = length(dex);
            center = center(dex,:);
        end
    end

Nx = size(x,1);
Np = size(p,1);

tR = 3*r;

centerp = eff_kmeans(p,tR,5);
centerx = eff_kmeans(x,tR,5);

MR = fun(centerx,p);
MC = fun(x,centerp);

[QC,~,~] = qr(MC,0);
[QR,~,~] = qr(MR',0);

if( tR+5 < Np && tR+5 < Nx )
    cs = randsample(Np,tR+5);
    rs = randsample(Nx,tR+5);
else
    cs = 1:Np;
    rs = 1:Nx;
end

M1 = QC(rs,:);
M2 = QR(cs,:);
M3 = fun(x(rs,:),p(cs,:));
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