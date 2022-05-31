function varargout = rsvd(A,r,varargin)
%RSVD    Randomized singular value decomposition.
%   [U,S,V] = RSVD(X,r) produces a rank r singular value decomposition of
%   X.
%
%   S = RSVD(X,r) returns a vector containing the top r singular values.
%
%   The X,r pairs can be followed by parameter,value pairs to specify
%   additional properties. Supported properties include:
%     - tolerance (default: 1e-8)  : tolerance for convergence;
%     - oversampling (default: 0.1): if the oversampling parameter k is
%                                    smaller than one, then r(1+k) columns
%                                    will be used in randomized SVD; if k
%                                    is greater or equal to one, then r+k
%                                    columns will be used;
%     - maxiter (default: 1000)    : maximum number of iteration;
%     - U0, V0 (default: [])       : initial guess for U and V vectors
%                                    respectively, the number of columns
%                                    must be greater than r; if both of
%                                    them are empty, the algorithm starts
%                                    from the side with larger size with a
%                                    Gaussian random matrix.
%
%   See also SVD, SVDS.

%   Copyright 2022 Yingzhou Li.


validfunc = @(x) isnumeric(x) && isscalar(x) && (x > 0);
par = inputParser;
addParameter(par, 'tolerance', 1e-8, validfunc);
addParameter(par, 'oversampling', 0.1, validfunc);
addParameter(par, 'maxiter', 1000, validfunc);
addParameter(par, 'u0', []);
addParameter(par, 'v0', []);
parse(par,varargin{:});

tol = par.Results.tolerance;
k = par.Results.oversampling;
if k < 1
    k = k*r;
end
maxit = par.Results.maxiter;
[m,n] = size(A);
if ~isempty(par.Results.u0) || ~isempty(par.Results.v0)
    if (~isempty(par.Results.u0) && ~isempty(par.Results.v0) && m >= n) ...
            || isempty(par.Results.v0)
        U = par.Results.u0;
        st = 2;
    else
        V = par.Results.v0;
        st = 1;
    end
else
    if m >= n
        [U,~] = qr(randn(m,r+k),0);
        st = 2;
    else
        [V,~] = qr(randn(n,r+k),0);
        st = 1;
    end
end

Sold = zeros(r+k,1);
for it = st:maxit+st-1
    if mod(it,2) == 1
        [U,S,Q] = svdecon(A*V);
        S = diag(S);
        diff = norm(S(1:r)-Sold(1:r),'inf')/S(1);
        if diff <= tol
            if nargout == 3
                varargout{1} = U(:,1:r);
                varargout{2} = diag(S(1:r));
                varargout{3} = V*Q(:,1:r);
            else
                varargout{1} = S(1:r);
            end
            return;
        end
        Sold = S;
    else
        [Q,S,V] = svdecon(U'*A);
        S = diag(S);
        diff = norm(S(1:r)-Sold(1:r),'inf')/S(1);
        if diff <= tol
            if nargout == 3
                varargout{1} = U*Q(:,1:r);
                varargout{2} = diag(S(1:r));
                varargout{3} = V;
            else
                varargout{1} = S(1:r);
            end
            return;
        end
        Sold = S;
    end
end

    function [UU,SS,VV] = svdecon(AA)
        [mm,nn] = size(AA);
        if mm >= nn
            [UU,SS,VV] = svd(AA,0);
        else
            [VV,SS,UU] = svd(AA',0);
        end
    end

end