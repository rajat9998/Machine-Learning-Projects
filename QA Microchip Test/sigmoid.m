function g = sigmoid(z)

%   g = SIGMOID(z) computes the sigmoid of z.

% Initialization
g = zeros(size(z));

% Compute the sigmoid of each value of z (z can be a matrix, vector or scalar).
g = 1 ./ (1 + exp(-z));

end