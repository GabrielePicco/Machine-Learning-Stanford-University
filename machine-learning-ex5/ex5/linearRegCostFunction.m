function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the
%   cost of using theta as the parameter for linear regression to fit the
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


predictionMatrix = X * theta;
errorMatrix = y - predictionMatrix;
errorMatrix = errorMatrix.^2;
sumOfErrors = sum(errorMatrix);
reg = (lambda/(2*m))*sum(theta(2:end,:) .^ 2);
J = (sumOfErrors / (2*m)) + reg;

grad = ((1/m)*sum(repmat((predictionMatrix - y), 1, size(X, 2)) .* X))';
regGrad = (lambda/m) .* theta;
if (length(regGrad) >= 1)
  regGrad(1) = 0;
endif
grad = grad .+ regGrad;



% =========================================================================

grad = grad(:);

end
