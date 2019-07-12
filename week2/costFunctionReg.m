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

one = ones(size(y));
h = sigmoid(X*theta);
J = -sum((y.*log(h) + (one-y).*(log(one-h)))) + lambda*sum(theta.^2)/2 - lambda*theta(1)^2/2;
J = J/m;

error = h - y;
%changedX = error'*X;
%size(changedX);

%for i=1:size(theta)
%	if i==1
%		grad(i) = changedX(i)/m;
%	else
%		grad = (changedX(i) + lambda*theta(i))/m;
%	end	
%end
x = theta;
theta(1) = 0;
grad = ((X'*error)+ lambda*theta)/m;
theta=x;



% =============================================================

end
