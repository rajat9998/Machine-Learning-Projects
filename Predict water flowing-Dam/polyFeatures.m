function [X_poly] = polyFeatures(X, p)

%   [X_poly] = POLYFEATURES(X, p) takes a data matrix X (size m x 1) and
%   maps each example into its polynomial features where
%   X_poly(i, :) = [X(i) X(i).^2 X(i).^3 ...  X(i).^p];

% Return the following variables correctly.
X_poly = zeros(numel(X), p);


for i = 1:p
    X_poly(:,i) = X(:,1).^i;
end

% X_poly(:,1:p) = X(:,1).^(1:p); % w/o for loop

end