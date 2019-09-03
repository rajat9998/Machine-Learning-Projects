function [C, sigma] = svmParams(X, y, Xval, yval)

%   [C, sigma] = SVMPARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma.

% Return the following variables correctly.
C = 1;
sigma = 0.3;


error = 1.0;

list = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

for i = 1:length(list)
    for j = 1:length(list)
        C_test = list(i);
        sigma_test = list(j);
        
        model = svmTrain(X, y, C, @(x1, x2)gaussianKernel(x1, x2, sigma));

        predictions = svmPredict(model, Xval);
        
        err = mean(double(predictions ~= yval));
        
        if(err < error)
            C = C_test;
            sigma = sigma_test;
            error = err;
        end
    end
end

end
