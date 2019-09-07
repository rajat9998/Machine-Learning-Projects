function centroids = computeCentroids(X, idx, K)

%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% Return the following variables correctly.
centroids = zeros(K, n);


for i = 1:K
    
    idx_i = find(idx==i);
    
    centroids(i, :) = mean(X(idx_i, :));
    
end   


end