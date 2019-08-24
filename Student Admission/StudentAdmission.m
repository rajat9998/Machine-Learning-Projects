
%% Initialization
clear;	close all; clc

%% Load Data
%  The first two columns contains the exam scores and the third column
%  contains the label.

data = load('data.txt');
X = data(:, [1, 2]);
y = data(:, 3);


%% ==================== Plotting graph ====================

fprintf(['Plotting data with + indicating (y = 1) examples and o ' ...
         'indicating (y = 0) examples.\n']);
plotData(X, y);

fprintf('\nProgram paused. Press enter to continue.\n');
pause;


%% ================= Logistic Regression ==================

%  Setup the data matrix appropriately, and add ones for the intercept term
[m, n] = size(X);

X = [ones(m, 1) X];

% Initialize fitting parameters
initial_theta = zeros(n + 1, 1);

% Compute and display initial cost and gradient
[cost, grad] = costFunction(initial_theta, X, y);

fprintf('Cost at initial theta (zeros): %f\n', cost);
fprintf('Gradient at initial theta (zeros): \n');
fprintf(' %f \n', grad);

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% ============== Optimizing using fminunc  ===============

%  Use a built-in function (fminunc) to find the optimal parameters theta.

%  Set options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', 400);

%  Run fminunc to obtain the optimal theta
%  This function will return theta and the cost 
[theta, cost] = ...
	fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);

fprintf('Cost at theta found by fminunc: %f\n', cost);
fprintf('theta: \n');
fprintf(' %f \n', theta);

% Plot Boundary
plotDecisionBoundary(theta, X, y);

fprintf('\nProgram paused. Press enter to continue.\n');
pause;


%% ================ Predict and Accuracies ================

%  Use the logistic regression model to predict the probability 
%  that a student with score 45 on exam 1 and 
%  score 85 on exam 2 will be admitted.

%  Predict probability for a student with score 45 on exam 1 
%  and score 85 on exam 2 
prob = sigmoid([1 45 85] * theta);
fprintf(['For a student with scores 45 and 85, we predict an admission ' ...
         'probability of %f\n\n'], prob);

% Compute accuracy on our training set
p = predict(theta, X);

fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);

fprintf('\nProgram paused. Press enter to continue.\n');
pause;


