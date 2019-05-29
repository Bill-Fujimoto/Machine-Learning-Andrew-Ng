function centroids = kMeansInitCentroids(X, K)
%KMEANSINITCENTROIDS This function initializes K centroids that are to be 
%used in K-Means on the dataset X
%   centroids = KMEANSINITCENTROIDS(X, K) returns K initial centroids to be
%   used with the K-Means on the dataset X
%

% You should return this values correctly
centroids = zeros(K, size(X, 2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should set centroids to randomly chosen examples from
%               the dataset X
%
X = unique(X, 'rows'); % Ensures no duplicate rows in X which ensures unique initial centroids.

##m = size(X, 1);
##index = randi(m, K, 1); % Generate vector of random integer between 1 and m (number of samples), K rows, 1 col
##centroids = X(index,:);  % Use vector indexing to select K samples to use as initial centroids.

% Initialize the centroids to be random examples
% Randomly reorder the indices of examples
randidx = randperm(size(X, 1));
% Take the first K examples as centroids
centroids = X(randidx(1:K), :);
% =============================================================

end

