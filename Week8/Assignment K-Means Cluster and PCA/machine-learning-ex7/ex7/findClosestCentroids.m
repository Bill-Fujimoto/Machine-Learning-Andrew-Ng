function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);
m = size(X,1);
delta_mag = zeros(m, K);

% You need to return the following variables correctly.
idx = zeros(m, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%
for k = 1:K
  delta = bsxfun(@minus, X, centroids(k,:)); % bsxfun() is a broadcast function, see doc for details.  It returns a matrix (size(X)) of the difference between each of m samples and centroid(k).
  delta_mag(:,k) = sum(delta.^2, 2);  % This computes the delta magnitude^2 of each sample(row) and loads this vector to k-th column of delta_mag.
endfor

[min_val idx] = min(delta_mag, [], 2); % Finds minimum value and k-th index of each row in delta_mag.

% =============================================================

end

