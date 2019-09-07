function [U, S] = pca(X)

%   [U, S, X] = pca(X) computes eigenvectors of the covariance matrix of X
%   Returns the eigenvectors U, the eigenvalues (on diagonal) in S


% Useful values
[m, n] = size(X);

% Return the following variables correctly.
U = zeros(n);
S = zeros(n);



Sigma = 1/m * X' * X;   % Covariance Matrix

[U, S, V] = svd(Sigma);


end