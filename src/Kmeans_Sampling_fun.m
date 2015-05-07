function [U,S] = Kmeans_Sampling_fun(fun,x,tol,r)

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

tR = 3*r;

centerx = eff_kmeans(x,tR,5);

M = fun(x,centerx);

[U,S,~] = svd(fun(centerx,centerx));
if ~isempty(S)
    idx = find(find(diag(S)>tol*S(1,1))<=r);
    U = M*U(:,idx)*(diag(diag(S(idx,idx)).^(-1)));
    S = S(idx,idx);
else
    U = zeros(Nx,0);
    S = zeros(0,0);
end

end