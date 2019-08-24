function p = predict(theta, X)

%   p = PREDICT(theta, X) computes the predictions for X using a 
%   threshold at 0.5

m = size(X, 1); % Number of training examples

% Initialization
p = zeros(m, 1);

% DIMENSIONS:
%	X =  m x (n+1)
%	theta = (n+1) x 1

hx = sigmoid(X * theta);

p = (hx >= 0.5);

end