function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta
%

##theta
##X
##y

n = size(grad)(1); % number of features


#### VECTORIZED WAY - J
h = @(x) sigmoid(x*theta);
J = y'*log(h(X)) + (1-y)'*log(1-h(X));
J = -J/m;

#### ITERATIVE WAY - J
##h = @(x) sigmoid(x*theta)
##for i = 1:m
##  x_i = X(i, :), y_i = y(i);
##  hyp = h(x_i);
##  J = J + y_i*log(hyp) + (1-y_i)*log(1-hyp)
##endfor
##J = -J/m


#### VECTORIZED WAY - grad
grad = sum((h(X).-y).*X)/m';
grad = grad';


#### ITERATIVE WAY - grad
##for j = 1:n
##  grad(j,1) = sum((h(X).-y).*X(:,j))/m
##endfor


% =============================================================

end
