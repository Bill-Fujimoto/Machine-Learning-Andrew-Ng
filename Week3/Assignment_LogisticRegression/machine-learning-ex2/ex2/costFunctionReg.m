function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
n = length(theta);
h = sigmoid(X*theta);
% Remember that Matlab/Octave use starting index of 1
% therefore theta(2:n) indexes all but the theta0 element (index = 1)

J = (-y'*log(h)-(1-y)'*log(1-h))/m + (theta(2:n)'*theta(2:n))*lambda/2/m; % uses natural log.  The second term is the regularization term starting from theta1 (index=2).

grad(1) =  X(:,1)'*(h - y)/m; % Computes theta0 gradient (index=1) separately by slicing x0 column only

grad(2:n) = X(:,2:n)'*(h - y)/m + theta(2:n)*lambda/m; % Computes gradients theta(1) to theta(n-1) (index = 2 to n) by slicing columns x(1) to x(n-1) 


% =============================================================

end
