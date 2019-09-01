function [J, grad] = linearRegCostFunction(X, y, theta, lambda)

%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% Return the following variables correctly 

J = 0;	% cost
grad = zeros(size(theta));	% gradient

%DIMENSIONS:
%   X = 12x2 = m x 1
%   y = 12x1 = m x 1
%   theta = 2x1 = (n+1) x 1
%   grad = 2x1 = (n+1) x 1

hx = X * theta;		% 12 x 1

reg_term = lambda/(2*m) * sum(theta(2:end).^ 2);	% scalar

J = 1/(2*m) * sum((hx - y).^ 2) + reg_term;		% scalar


grad(1) = 1/m * sum(X(:,1)' * (hx - y));	% scalar == 1x1

grad(2:end) = 1/m * sum(X(:,2:end)' * (hx - y)) + (lambda/m) * theta(2:end);	% n x 1


grad = grad(:);

end