function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% X is (5000 x 401; Theta1 is 25 x 401; so transpose Theta1 to match inner dimension.  Resultant a_2 is 5000 x 25
a_2 = sigmoid(X*Theta1');

% Add 1's column to a_2 hidden layer matrix, resultant a_2 is 5000 x 26
a_2 = [ones(m, 1) a_2];

% a_2 is 5000 x 26; Theta2 is 10 x 26; so transpose Theta2 to match inner dimension.  Resultant a_3 is 5000 x 10, each column is the probability of each digit.
a_3 = sigmoid(a_2*Theta2');

[pred, p] = max(a_3, [], 2); % max() in this config returns the max probability in each row, and the column index of the max proba which is equal to the predicted digit value.

% =========================================================================


end
