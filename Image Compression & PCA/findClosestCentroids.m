function idx = findClosestCentroids(X, centroids)

%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])


% Set K
K = size(centroids, 1);

% Return the following variables correctly.
idx = zeros(size(X,1), 1);


for i = 1:length(X)
    temp = zeros(K, 1);
    for j = 1:K 
    
        temp(j) = sum(abs(X(i, :) - centroids(j, :)));
    end
    [~, idx(i)] = min(temp);
end


end