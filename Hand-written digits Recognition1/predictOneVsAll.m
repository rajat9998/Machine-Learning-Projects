function p = predictOneVsAll(all_theta, X)

%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K 

m = size(X, 1);
num_labels = size(all_theta, 1);

% Return the following variables
p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

%size(X)
%size(all_theta)

prob_matrix = X * all_theta';


[~, p] = max(prob_matrix, [], 2); 	% obtain the max for each row.


  %%%%%%%% WORKING: Computation per input image %%%%%%%%%
  % for i = 1:m                               % To iterate through each input sample
  %     one_image = X(i,:);                   % 1 x 401 == 1 x no_of_features
  %     prob_mat = one_image * all_theta';    % 1 x 10  == 1 x num_labels
  %     [prob, out] = max(prob_mat);
  %     %out: predicted output
  %     %prob: probability of predicted output
  %     p(i) = out;
  % end
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  %%%%%%%% WORKING %%%%%%%%%
  % for i = 1:m
  %     RX = repmat(X(i,:),num_labels,1);
  %     RX = RX .* all_theta;
  %     SX = sum(RX,2);
  %     [val, index] = max(SX);
  %     p(i) = index;
  % end
  %%%%%%%%%%%%%%%%%%%%%%%%%%

end