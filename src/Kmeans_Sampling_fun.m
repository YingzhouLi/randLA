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

MD = fun(centerx,centerp);
[U,S,V] = svd(MD,0);
if ~isempty(S)
    idx = find(find(diag(S)>tol*S(1,1))<=r);
    S = S(idx,idx);
    U = MC*U(:,idx)*diag(diag(S).^(-1));
    V = MR'*V(:,idx)*diag(diag(S).^(-1));
else
    U = zeros(Nx,0);
    S = zeros(0,0);
    V = zeros(Np,0);
end

end