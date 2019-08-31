function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)

%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 

%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1)); % 25 x 401

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1)); % 10 x 26

% Setup some useful variables
m = size(X, 1); % 5000
         
% Return the following variables correctly
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

%%%%%%%%%%% Implementing Backpropogation for Theta_gra with Regularization %%%%%%%%%%%%%

%%%%%%%%%%% Forward Propagation %%%%%%%%%%%%%%%

X = [ones(m,1), X];  % Adding 1 as first column in X
  
a1 = X; % 5000 x 401
  
z2 = a1 * Theta1';  % m x hidden_layer_size == 5000 x 25
a2 = sigmoid(z2); % m x hidden_layer_size == 5000 x 25
a2 = [ones(size(a2,1),1), a2]; % Adding 1 as first column in z = (Adding bias unit) 
                    % m x (hidden_layer_size + 1) == 5000 x 26
  
z3 = a2 * Theta2';  % m x num_labels == 5000 x 10
a3 = sigmoid(z3); % m x num_labels == 5000 x 10
  
hx = a3; % m x num_labels == 5000 x 10

% Converting y into vector of 0's and 1's for multi-class classification
% y_Vec = zeros(m, num_labels);     % non-vectorized
% for i = 1:m
%     y_Vec(i, y(i)) = 1;
% end

y_Vec = (1:num_labels)==y; % m x num_labels == 5000 x 10

%%%%%%%%%%% Calculating J w/o Regularization %%%%%%%%%%%  

J = (1/m) * sum(sum(-y_Vec .* log(hx) - (1-y_Vec) .* log(1-hx)));  %scalar

%%%%%%%%%%% Calculating delta values(error) %%%%%%%%%%%%
delta_3 = a3 - y_Vec;	% 5000 x 10
    
delta_2 = (delta_3 * Theta2) .* [ones(size(z2,1),1) sigmoidGradient(z2)];	% 5000 x 26
delta_2 = delta_2(:, 2:end);	% 5000 x 25 %Removing delta2 for bias node

Theta1_grad = (1/m) * (delta_2' * a1); % 25 x 401
Theta2_grad = (1/m) * (delta_3' * a2); % 10 x 26

%%%%%%% Backpropogation using for loop %%%%%%%
  % for t=1:m
  %     % Here X is including 1 column at begining
  %     
  %     % for layer-1
  %     a1 = X(t,:)'; % (n+1) x 1 == 401 x 1
  %     
  %     % for layer-2
  %     z2 = Theta1 * a1;  % hidden_layer_size x 1 == 25 x 1
  %     a2 = [1; sigmoid(z2)]; % (hidden_layer_size+1) x 1 == 26 x 1
  %   
  %     % for layer-3
  %     z3 = Theta2 * a2; % num_labels x 1 == 10 x 1    
  %     a3 = sigmoid(z3); % num_labels x 1 == 10 x 1    
  % 
  %     yVector = (1:num_labels)'==y(t); % num_labels x 1 == 10 x 1    
  %     
  %     %calculating delta values
  %     delta_3 = a3 - yVector; % num_labels x 1 == 10 x 1    
  %     
  %     delta_2 = (Theta2' * delta_3) .* [1; sigmoidGradient(z2)]; % (hidden_layer_size+1) x 1 == 26 x 1
  %     
  %     delta_2 = delta_2(2:end); % hidden_layer_size x 1 == 25 x 1 %Removing delta2 for bias node  
  %     
  %     % delta_1 is not calculated because we do not associate error with the input  
  %     
  %     % CAPITAL delta update
  %     Theta1_grad = Theta1_grad + (delta_2 * a1'); % 25 x 401
  %     Theta2_grad = Theta2_grad + (delta_3 * a2'); % 10 x 26
  %  
  % end
  % 
  % Theta1_grad = (1/m) * Theta1_grad; % 25 x 401
  % Theta2_grad = (1/m) * Theta2_grad; % 10 x 26
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
%%%%%%%%%%%% Adding Regularisation term in J and Theta_grad %%%%%%%%%%%%%
reg_term = (lambda/(2*m)) * (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2))); %scalar
  
% Costfunction With regularization
J = J + reg_term; %scalar
  
% Calculating gradients for the regularization
Theta1_grad_reg_term = (lambda/m) * [zeros(size(Theta1, 1), 1) Theta1(:,2:end)]; % 25 x 401
Theta2_grad_reg_term = (lambda/m) * [zeros(size(Theta2, 1), 1) Theta2(:,2:end)]; % 10 x 26
  
% Adding regularization term to earlier calculated Theta_grad
Theta1_grad = Theta1_grad + Theta1_grad_reg_term;
Theta2_grad = Theta2_grad + Theta2_grad_reg_term;
  
% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end							   
