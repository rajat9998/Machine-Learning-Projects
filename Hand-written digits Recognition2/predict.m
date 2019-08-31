function p = predict(Theta1, Theta2, X)

%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)
%	Implemented Forward Propagation

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly
p = zeros(size(X, 1), 1);

h1 = sigmoid([ones(m, 1) X] * Theta1');
h2 = sigmoid([ones(m, 1) h1] * Theta2');
[dummy, p] = max(h2, [], 2);

%%%%%%%%% Detailed Answer %%%%%%%%%
% a1 = [ones(m,1) X]; % 5000 x 401 == no_of_input_images x no_of_features % Adding 1 in X 
%   %No. of rows = no. of input images
%   %No. of Column = No. of features in each image
%   
% z2 = a1 * Theta1';  % 5000 x 25
% a2 = sigmoid(z2);   % 5000 x 25
%  
% a2 =  [ones(size(a2,1),1) a2];  % 5000 x 26
%   
% z3 = a2 * Theta2';  % 5000 x 10
% a3 = sigmoid(z3);  % 5000 x 10
%   
% [~, p] = max(a3,[],2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end