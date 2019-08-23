function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)

%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    
    fprintf('Theta: %f, %f\n',theta(1),theta(2));

    error = (X * theta) - y;
    
    theta = theta - (alpha/m) * X' * error;

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
