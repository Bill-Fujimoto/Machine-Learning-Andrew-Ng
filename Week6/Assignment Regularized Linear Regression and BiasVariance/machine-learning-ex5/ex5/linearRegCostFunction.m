function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta);

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.

e = (X * theta) - y; % Since X contains features in rows, and theta is a column vector, X*theta has the proper inner dimensions, returns a column vector, y is also a column vector. k is therefore a column vector. theta'*x applies to the original column vector of x, but x is transposed to rows in the matrix X.

% By setting theta0 = 0, this prevents regularization of theta0.                  
theta(1) = 0;   

J = e' * e/2/m + (theta' * theta) * lambda/2/m; 
% Or J = sum(k .^2)/2/m; .^2 gives square of each element
% k' * k is the magnitude squared of the vector k, i.e. dot product

% Remember that Matlab/Octave use starting index of 1
% therefore theta(2:n) indexes all but the theta0 element (index = 1)

%grad(1) =  X(:,1)' * k/m; % Computes theta0 gradient (index=1) separately by slicing x0 column only
%grad(2:n) = X(:,2:n)' * k/m + theta(2:n) * lambda/m; % Computes gradients theta1 to theta(n-1) (index = 2 to n) by slicing columns x1 to x(n-1) 

% Again, by setting theta0 = 0, this prevents regularization of theta0 term.  This line is simpler than the equivalent indexed grad solution above.
grad = (X' * e + theta * lambda)/m; 

% =========================================================================

%grad = grad(:);

end
