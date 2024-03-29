
%% Initialization
clear ; close all; clc

%% =================  Visualizing Dataset 1 ==================

%  We start this section by first loading and visualizing the dataset. 
%  The following code will load the dataset into your environment and plot
%  the data.

fprintf('Loading and Visualizing Data ...\n')

% X, y
load('data1.mat');

% Plot training data
plotData(X, y);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ==================== Training Linear SVM ====================

%  The following code will train a linear SVM on the dataset and plot the
%  decision boundary learned.

% X, y
load('data1.mat');

fprintf('\nTraining Linear SVM ...\n')

% Now, try to change the C value below and see how the decision
% boundary varies (e.g., try C = 1000)
C = 1;
model = svmTrain(X, y, C, @linearKernel, 1e-3, 20);
visualizeBoundaryLinear(X, y, model);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =============== Implementing Gaussian Kernel ===============

%  We will now implement the Gaussian kernel to use
%  with the SVM.

fprintf('\nEvaluating the Gaussian Kernel ...\n')

x1 = [1 2 1]; x2 = [0 4 -1]; sigma = 2;
sim = gaussianKernel(x1, x2, sigma);

fprintf(['Gaussian Kernel between x1 = [1; 2; 1], x2 = [0; 4; -1], sigma = 0.5 :' ...
         '\n\t%f\n(this value should be about 0.324652)\n'], sim);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =================  Visualizing Dataset 2 ==================

%  The following code will load the next dataset into environment and 
%  plot the data. 

fprintf('Loading and Visualizing Data ...\n')
 
% X, y
load('data2.mat');

% Plot training data
plotData(X, y);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ========== Training SVM with RBF Kernel (Dataset 2) ==========

%  After implementing the kernel, we can now use it to train the 
%  SVM classifier.
% 
fprintf('\nTraining SVM with RBF Kernel (this may take 1 to 2 minutes) ...\n');


% We will have X, y in your environment
load('data2.mat');

% SVM Parameters
C = 1; sigma = 0.1;

% We set the tolerance and max_passes lower here so that the code will run
% faster. However, in practice,  we want to run the training to
% convergence.
model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
visualizeBoundary(X, y, model);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =================  Visualizing Dataset 3 ==================

%  The following code will load the next dataset into environment and 
%  plot the data. 

fprintf('Loading and Visualizing Data ...\n')

% X, y
load('data3.mat');

% Plot training data
plotData(X, y);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ========== Training SVM with RBF Kernel (Dataset 3) ==========

%  This is a different dataset that you can use to experiment with. Try
%  different values of C and sigma here.
% 

% X, y
load('data3.mat');

% Try different SVM Parameters here
[C, sigma] = svmParams(X, y, Xval, yval);

% Train the SVM
model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
visualizeBoundary(X, y, model);


fprintf('Program paused. Press enter to continue.\n');
pause;
