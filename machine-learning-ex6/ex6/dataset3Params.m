function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and
%   sigma. You should complete this function to return the optimal C and
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.1;

values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
values2 = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example,
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using
%        mean(double(predictions ~= yval))
%
#{
minError = -1;
for i = 1:length(values)
  cTest = values(i);
  for j = 1:length(values)
    sigmaTest = values(j);
    model = svmTrain(X, y, cTest, @(x1, x2) gaussianKernel(x1, x2, sigmaTest));
    %fprintf('\nTesting C: %f and sigma: %f', cTest, sigmaTest);
    predictions = svmPredict(model, Xval);
    errTest = mean(double(predictions ~= yval));
    if (minError < 0 || errTest < minError)
      fprintf('\nPrediction error is %f with C %f and sigma %f', errTest, cTest, sigmaTest);
      minError = errTest;
      C = cTest;
      sigma = cTest;
    endif
  end
end
#}






% =========================================================================

end
