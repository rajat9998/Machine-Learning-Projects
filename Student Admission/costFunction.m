function [J, grad] = costFunction(theta, X, y)

%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

J = 0;
grad = zeros(size(theta));

% DIMENSIONS:
%	theta = (n+1) x 1
%	X = m x (n+1)
%	y = m x 1
%	grad = (n+1) x 1
%	J = scalar

z = X * theta;	% m x 1

hx = sigmoid(z);	% m x 1

J = 1/m * sum(-y' * log(hx) - (1-y)' * log(1-hx));	% scalar


grad = 1/m * (X' * (hx - y)); % (n+1) x 1

end