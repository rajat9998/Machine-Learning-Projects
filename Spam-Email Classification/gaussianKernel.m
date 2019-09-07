function sim = gaussianKernel(x1, x2, sigma)

%   sim = gaussianKernel(x1, x2) returns a gaussian kernel between x1 and x2
%   and returns the value in sim

% Ensure that x1 and x2 are column vectors
x1 = x1(:); x2 = x2(:);

% Return the following variables correctly.
sim = 0;

sim = exp(-sum(abs(x1 - x2).^2)./(2*sigma*sigma));


end
