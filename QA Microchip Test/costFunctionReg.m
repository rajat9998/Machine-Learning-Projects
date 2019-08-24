function [J, grad] = costFunctionReg(theta, X, y, lambda)

%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

%DIMENSIONS: 
%   theta = (n+1) x 1
%   X     = m x (n+1)
%   y     = m x 1
%   grad  = (n+1) x 1
%   J     = Scalar

z = X * theta;	% m x 1

hx = sigmoid(z);	% m x 1

reg_term = (lambda / (2*m)) * sum(theta(2:end) .^ 2);

J = (1/m) * sum(-y' * log(hx) - (1-y)' * log(1-hx)) + reg_term;	% scalar


grad(1) = (1/m) * (X(:,1)' * (hx - y));		% 1 x 1

grad(2:end) = (1/m) * (X(:, 2:end)' * (hx - y)) + (lambda/m) * theta(2:end);	% n x 1

end