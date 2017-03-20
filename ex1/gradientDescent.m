function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    suma1=0;
    suma2=0;
    for i=1:m
        suma1 = (X(i,1:2)*theta - y(i))*X(i,1)+suma1;
    end
    theta1 = theta(1) - (alpha/m) * suma1;
    for i=1:m
        suma2 = (X(i,1:2)*theta- y(i))*X(i,2)+suma2;
    end
    theta2=theta(2) - (alpha/m) *suma2;
    
    theta(1) = theta1;
    theta(2) = theta2;
    




    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
