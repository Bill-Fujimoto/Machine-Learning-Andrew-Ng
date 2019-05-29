fprintf('Loading ex6data3.mat Data ...\n')

%% ========== Part 7: Training SVM with RBF Kernel (Dataset 3) ==========

%  This is a different dataset that you can use to experiment with. Try
%  different values of C and sigma here.
% 

% Load from ex6data3: 
% You will have X, y in your environment
load('ex6data3.mat');

% Try different SVM Parameters here
%[C, sigma] = dataset3Params(X, y, Xval, yval);
C = 1
sigma = .1
% C = 1, sigma =0.1 most closely approximates the graph in ex6.pdf Fig 7

fprintf('Training SVM Using selected C and sigma ...\n')
% Train the SVM
model= svmTrain(X, y, C(1), @(x1, x2) gaussianKernel(x1, x2, sigma(1)));
predictions = svmPredict(model, Xval);
%score(i, j) = mean(double(predictions ~= yval)); % Use ~= for min  
score = mean(double(predictions == yval)) % Use == for max

fprintf('Plotting training data and decision boundary ...\n')
visualizeBoundary(X, y, model);
