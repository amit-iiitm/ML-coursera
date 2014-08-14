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
cost=0;
for i=1:m
	z=X(i,:)*theta;
	g=1/(1+e^(-z));
	cost=cost+y(i)*log(g)+(1-y(i))*log(1-g);
end
J=(-1/m)*cost
for i=1:size(theta)
	err=0;
	for j=1:m
		z=X(j,:)*theta;
		g=1/(1+e^(-z));
		err=err+(g-y(j))*X(j,i);
	end
	grad(i)=(1/m)*err;
end

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%








% =============================================================

end
