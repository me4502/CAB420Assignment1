function plot2DLinear(obj, X, Y)
% plot2DLinear(obj, X,Y)
%   plot a linear classifier (data and decision boundary) when features X are 2-dim
%   wts are 1x3,  wts(1)+wts(2)*X(1)+wts(3)*X(2)
%
[n,d] = size(X);
if (d~=2) error('Sorry -- plot2DLogistic only works on 2D data...'); end;

%%% TODO: Fill in the rest of this function...
hold on;

% Plot class data.
classes = unique(Y);

classAIndicies = find(Y==classes(1));
xPointsClassA = X(classAIndicies, 1:end);
scatter(xPointsClassA(:, 1), xPointsClassA(:, 2));

classBIndicies = find(Y==classes(2));
xPointsClassB = X(classBIndicies, 1:end);
scatter(xPointsClassB(:, 1), xPointsClassB(:, 2));


% Plot decision boundary.
wts = getWeights(obj);
f = @(x1, x2) wts(1) + wts(2)*x1 + wts(3)*x2;
ezplot(f,[-3.5,3.5])

legend('Class 0','Class 1', 'Decision Bondary');
hold off;
