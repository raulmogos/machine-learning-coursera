function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

n = size(theta)(1,1)

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

    
    

    new_theta = zeros(n,1);
    
    A = X*theta - y;
    
    for i = 1:n
      new_theta(i,1) = theta(i,1) - alpha*(1/m)*sum(A.*X(:,i));
    endfor
    
    theta = new_theta;

    
    % VECTORIZED FORM
    
    % new_theta = theta;
    
    % A = X*theta - y;
    % R = (1/m) * sum(A.*X);
    % new_theta = theta - alpha*R';
    
    % theta = new_theta;
    
    % END VECTORIZED FORM
    
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

##plot(1:num_iters, J_history)

end
