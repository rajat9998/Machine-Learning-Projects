function Z = projectData(X, U, K)

%   Z = projectData(X, U, K) computes the projection of 
%   the normalized inputs X into the reduced dimensional space spanned by
%   the first K columns of U. It returns the projected examples in Z.
%

% Return the following variables correctly.
Z = zeros(size(X, 1), K);

% DIMENSIONS:
%    X = m x n
%    U = n x n
%    U_reduce = n x K
%    K = scalar

U_reduce = U(:, 1:K);   % n x K

Z = X * U_reduce;   % m x K


end