function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter 'grad' should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
% Expand the 'y' output values into a matrix of single values.  Syntax explanation: Reading left to right, it creates an eye matrix of num_labels size. Then it uses 'y' as a vectorized index. For each value in y, it copies that row of the eye matrix into y_matrix. That's what the (y,:) does - the colon means "use all columns".
y_matrix = eye(num_labels)(y,:);

a1 = X; 
% Add ones to the a1 data matrix
a1 = [ones(m, 1) a1];
z2 = a1*Theta1'; % size 5000 x 25
a2 = sigmoid(z2); % size 5000 x 25
% Add ones to the a2 data matrix
a2 = [ones(m, 1) a2]; % size 5000 x26
z3 = a2*Theta2'; % size 5000 x 10
a3 = sigmoid(z3);
h = a3;  % size 5000 x 10
  
% From tutorial: cost is a scalar value. Since y_matrix and a3 are both matrices, you need to compute the double-sum. Remember to use element-wise multiplication with the log() function. The sum(A(:)) adds all elements in a matrix which is a bit cleaner than using a double-sum.
J = sum((-y_matrix.*log(h) - (1-y_matrix).*log(1-h))(:))/m;

%For a discussion of how you can use (a less intuitive) matrix multiplication, see this thread: https://www.coursera.org/learn/machine-learning/discussions/all/threads/AzIrrO7wEeaV3gonaJwAFA
% This implements the cost function using matrix multiplication, then sums the diagonal elements with trace(); this also eliminates the unwanted elements.  This is less computationally efficient than element-wise multiplication since it discards 90% of the matrix except for the diagonal.
% J = trace((-y_matrix' *log(h) - (1-y_matrix)' *log(1-h)))/m;

% Delete theta0 bias term column
Theta1_nobias = Theta1(:,2:end); 
Theta2_nobias = Theta2(:,2:end);

% Square each element of ThetaN:
Theta1_sqr = Theta1_nobias.^2;
Theta2_sqr = Theta2_nobias.^2;

% Sum all elements of each ThetaN_sqr matrix, then add and scale by lambda/2/m
J_reg = (sum(Theta1_sqr(:)) + sum(Theta2_sqr(:)))*lambda/2/m;

% Add cost + regularization
J = J + J_reg;

% Backpropagation 
% Based on guidance from ex4 Tutorial for nn CostFunction and back propagation: https://www.coursera.org/learn/machine-learning/discussions/all/threads/a8Kce_WxEeS16yIACyoj1Q

d3 = a3 - y_matrix; % size 5000 x 10
  %size(d3)
d2 = d3*Theta2_nobias .* sigmoidGradient(z2); % size 5000 x 25
  %size(d2)
Delta2 = d3'*a2; % size 10 x 26
  %size(delta2)
Delta1 = d2'*a1; % size 25 x 401
  %size(delta1)
% Unregularized gradients
Theta2_grad = Delta2/m;
Theta1_grad = Delta1/m;

% Regularized gradients
Theta2(:,1) = 0; % Set first column of ThetaN to 0's
Theta1(:,1) = 0;

Theta2_grad = Theta2_grad + (lambda/m)*Theta2;
Theta1_grad = Theta1_grad + (lambda/m)*Theta1;

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
