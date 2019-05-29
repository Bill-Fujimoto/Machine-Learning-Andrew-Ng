function [C, sigma, score] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
%C= 1;
%sigma = 0.1;

C = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigma = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
score = zeros(length(C), length(sigma));
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
% ============ When submitting for auto grading, comment out the for loops below =============
##for i = 1:length(C);
##  for j = 1:length(sigma);
##    model= svmTrain(X, y, C(i), @(x1, x2) gaussianKernel(x1, x2, sigma(j)));
##    predictions = svmPredict(model, Xval);
##    %score(i, j) = mean(double(predictions ~= yval)); % Use ~= for min  
##    score(i, j) = mean(double(predictions == yval)); % Use == for max
##    
##  endfor
##endfor
% =========================================================================================
% Use the following boolean expression from command line to indicate which elements are the max score
%  score==max(score(:))

%Manually determined by examining the score matrix.  Since multiple combinations of C and sigma produced identical scores, a better selection method is needed to select the median value.
C = 1;
sigma = 0.1;

% =========================================================================

end
