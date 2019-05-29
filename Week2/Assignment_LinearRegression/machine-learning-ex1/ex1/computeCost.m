function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.
m = length(y);
k = (X*theta) - y; % Since X contains features in rows, and theta is a column
                   % vector, X*theta has the proper inner dimensions, 
                   % returns a column vector, y is also a column vector.
                   % k is therefore a column vector.
                   % theta'*x applies to the original column vector of x, 
                   % but x is transposed to rows in the matrix X.
J = k'* k/2/m;  % Or J = sum(k .^2)/2/m; .^2 gives square of each element
                % k' * k is the magnitude squared of the vector k, i.e. dot product

% =========================================================================

end
