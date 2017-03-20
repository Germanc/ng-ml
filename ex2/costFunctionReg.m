function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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

JJ=0;
for i=1:m
    J = J+(1/m)*(-y(i)*log(sigmoid(X(i,:)*theta))-(1-y(i))*log(1-sigmoid(X(i,:)*theta)));
end
for i=2:(size(theta))
    JJ = JJ+(lambda/(2*m))*theta(i)^2;
end
J = J+JJ;


for l=1:m
   grad(1) = grad(1)+(1/m)*((sigmoid(X(l,:)*theta)-y(l))*X(l,1));
end


for i=2:(size(theta))
    for l=1:m
       grad(i) = grad(i)+(1/m)*((sigmoid(X(l,:)*theta)-y(l))*X(l,i));
    end
    grad(i) = grad(i) + (lambda/m)*theta(i);
end






% =============================================================

end
