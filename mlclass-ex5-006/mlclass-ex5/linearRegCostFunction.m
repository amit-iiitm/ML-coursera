function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful valuesb4upZCnFjK
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta))
t=theta
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
cost=0;
for i=1:m
	hypo=X(i,:)*theta;
	cost=cost+(hypo-y(i))*(hypo-y(i));
end
cost=(1/(2*m))*cost;
s=0
for j=2:size(theta)
	s=s+theta(j)^2;
end
s=(lambda/(2*m))*s;
J=cost+s
q=size(theta)	
for i=1:size(theta)
	temp=0;
	for j=1:m
		temp=temp+(X(j,:)*theta-y(j))*X(j,i);
	end
	if i>1
	   grad(i)=(1/m)*temp+(lambda/m)*theta(i)
	else
	   grad(i)=(1/m)*temp
end	










% =========================================================================

grad = grad(:);

end
