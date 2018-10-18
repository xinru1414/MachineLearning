function main
addpath(genpath(pwd));

% Get training and testing data
load('../data/mnist2.mat')

rng(42);
% training
[n, p] = size(xtrain);
w0 = zeros(1, p);
T = 100*n;
lambda = 100;
w = train(w0, xtrain, ytrain, T, lambda);

% evaluation
fprintf('Train Accuracy: %f, Test Accuracy: %f\n', accuracy(xtrain, ytrain, w), accuracy(xtest, ytest, w));

rmpath(genpath(pwd));
end

% computes the accuracy on a test dataset
function v = accuracy(X, y, w)
[n, ~] = size(X);
n_mistakes = 0;
for j =1: n
    yhat = pred(w, X(j,:));
    if (y(j) > 0 && yhat < 0) || (y(j) < 0 && yhat >= 0)
        n_mistakes = n_mistakes + 1.0;
    end
end
v =  1 - n_mistakes/n;
end

function [v] = pred(w, X)
yhat = dot(w, X);
if yhat >= 0
    v =  1;
else
    v = -1;
end
end
