function X_rec = recoverData(Z, U, K)

%   X_rec = RECOVERDATA(Z, U, K) recovers an approximation the 
%   original data that has been reduced to K dimensions. It returns the
%   approximate reconstruction in X_rec.
%

% Return the following variables correctly.
X_rec = zeros(size(Z, 1), size(U, 1));


% DIMENSIONS: 
%    Z = m x K
%    U = n x n
%    U_reduce = n x k
%    K = scalar
%    X_rec = m x n

U_reduce = U(:, 1:K); % n x K

X_rec = Z * U_reduce';  % m x n


end