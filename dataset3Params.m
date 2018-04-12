function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns our choice of C and sigma for Part 3 of the exercise
%where we select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns our choice of C and 
%   sigma. We should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% We need to return the following variables correctly.
C = 0.03;
sigma = 0.03;

% Fill in this function to return the optimal C and sigma
% learning parameters found using the cross validation set.
% You can use svmPredict to predict the labels on the cross
% validation set. For example, 
% predictions = svmPredict(model, Xval);
% will return the predictions on the cross validation set.
%
%  Note: We can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
CTemp = 0.03;
sigmaTemp = 0.03;
errorMin = intmax;

for i = [0.01 0.03 0.1 0.3 1 3 10 30]
    for j = [0.01 0.03 0.1 0.3 1 3 10 30]
        model= svmTrain(X, y, i, @(x1, x2) gaussianKernel(x1, x2, j));
        predictions = svmPredict(model, Xval);
        errorCurr = mean(double(predictions ~= yval));
        if errorCurr < errorMin
            CTemp = i;
            sigmaTemp = j;
            errorMin = errorCurr;
        end
    end
end
C = CTemp;
sigma = sigmaTemp;

end
