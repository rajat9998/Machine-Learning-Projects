
%% Initialization
clear; close all; clc

%% Load Data
data = load('data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);


%% ================ Feature Normalization =================

% Scale features and set them to zero mean
fprintf('Normalizing Features ...\n');

[X mu sigma] = featureNormalize(X);

% Add intercept term to X
X = [ones(m, 1) X];


%% ================ Part 2: Gradient Descent ================

fprintf('Running gradient descent ...\n');

% Choose some alpha value
alpha = 0.01;
num_iters = 400;

% Init Theta and Run Gradient Descent 
theta = zeros(3, 1);
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);

% Plot the convergence graph
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');

% Display gradient descent's result
fprintf('Theta computed from gradient descent: \n');
fprintf(' %f \n', theta);
fprintf('\n');

% Estimate the price of a 1650 sq-ft, 3 br house
price = 0;
temp = [1 1650 3];
temp(1,2) = (temp(1,2) - mu(1,1))/(sigma(1,1));
temp(1,3) = (temp(1,3) - mu(1,2))/(sigma(1,2));
price = temp * theta;

fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using gradient descent):\n $%f\n'], price);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================ Normal Equations ================

fprintf('Solving with normal equations...\n');

%% Load Data
data = csvread('data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Add intercept term to X
X = [ones(m, 1) X];

% Calculate the parameters from the normal equation
theta = normalEqn(X, y);

% Display normal equation's result
fprintf('Theta computed from the normal equations: \n');
fprintf(' %f \n', theta);
fprintf('\n');

% Estimate the price of a 1650 sq-ft, 3 br house
price = 0;
temp = [1 1650 3];
temp(1,2) = (temp(1,2) - mu(1,1))/(sigma(1,1));
temp(1,3) = (temp(1,3) - mu(1,2))/(sigma(1,2));
price = temp * theta;

fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using gradient descent):\n $%f\n'], price);
