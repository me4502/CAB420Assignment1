%% CAB420 Assignment 1
%
% Authors: 
% - Jarrod Williams, n9722068
% - Madeline Miller, n9342401
% 
%
%% 1. Features, Classes, and Linear Regression

% Load motorcycle training dataset
mTrain = load('data/mcycleTrain.txt');

% Separate features; x = single fature, y = target value
ytr = mTrain(: ,1); xtr = mTrain(: ,2);

% Add contant feature and quatratic feature of x
Xtr = polyx(xtr, 2);

% Create and learn a regression predictor from the data Xtr, ytr.
learner = linearReg(Xtr, ytr);

% Use learner to predict the y-values at the original training data points.
yhat = predict(learner, Xtr);

% Plot newly created linear predictor output at x new points.
xline = [0:.01:2]' ; % Transpose
yline = predict(learner, polyx(xline, 2)); % Assuming quadratic features
figure('name', 'Quadratic linear predictor');
plot(xline, yline, 'ro');

% Plot training data and label figure.
hold on
plot (xtr, ytr, 'bo'); % Plot training data
legend('Linear Predictor', 'Training Data');
title('Quadratic linear predictor');

% Calculate Mean Squared Error (MSE) for quadratic model using training data.
mseQuadTrain = immse(yhat, ytr);

% Calculate MSE for quadratic model using test data.
mTest = load('data/mcycleTest.txt');
ytest = mTest(: ,1); xtest = mTest(: ,2);
Xtest = polyx(xtest, 2);
learner = linearReg(Xtest, ytest);
yhat = predict(learner, Xtest);
mseQuadTest = immse(yhat, ytest);

% Repeat process for a fifth-degree polynomial.
Xtr = polyx(xtr, 5);
learner = linearReg ( Xtr , ytr );
yhat = predict ( learner , Xtr );
yline = predict ( learner , polyx ( xline ,5) ); % assuming quintic features
figure('name', 'Quintic linear predictor');
plot ( xline , yline ,'ro ');
hold on
plot (xtr, ytr, 'bo');
legend('Linear Predictor', 'Training Data');
title('Quintic linear predictor');
mseQuinTrain = immse(yhat, ytr); % Calculate MSE for training data
Xtest = polyx(xtest, 5); % Calculate MSE for test data
learner = linearReg(Xtest, ytest);
yhat = predict(learner, Xtest);
mseQuinTest = immse(yhat, ytest);

fprintf('The MSE for the quadratic linear predictor was: %.2f (training data), %.2f (test data)\n', mseQuadTrain, mseQuadTest);
fprintf('The MSE for the quintic linear predictor was: %.2f (training data), %.2f (test data)\n', mseQuinTrain, mseQuinTest);


%% 2. kNN Regression

% Create a list of K values
Ks = [1, 2, 3, 5, 10, 50];
ytr = mTrain(: ,1); xtr = mTrain(: ,2);

% Plot training data
figure('name', 'kNN Regression predictor');
plot (xtr, ytr, 'bo'); % Plot training data
legend('Training Data');
hold on

xline = [0:.01:2]' ; % Transpose

% Create and learn a kNN regression predictor from the data Xtr, ytr for each K.
for i=Ks
    learner = knnRegress(i, xtr, ytr);
    
    yhat = predict(learner, xtr);
    
    % Plot newly created linear predictor output at x new points.
    yline = predict(learner, polyx(xline, 2)); % Assuming quadratic features
    plot(xline, yline, 'o', 'DisplayName', strcat('K=', num2str(i)));
end

%% 3. Hold-out and Cross-validation









%% 4. Nearest Neighbor Classifiers

% Import the dataset.
iris = load('data/iris.txt');
pi = randperm(size(iris, 1));
Y= iris(pi, 5); X = iris(pi, 1:2);

% (A) - Plot dataset by feature values.
figure('Name','Feature Values of Iris Dataset');
hold on;
colours = unique(Y);
for colour = 1:length(colours)
    scatterColour = [0,0,0];
    scatterColour(colour) = 1;
    featureIndicies = find(Y==colours(colour));
    points = X(featureIndicies, 1:end);
    pointsX = points(1:end, 1);
    pointsY = points(1:end, 2);
    scatter(pointsX, pointsY, [], scatterColour, 'filled');
end
legend('Class 0', 'Class 1', 'Class 2');
title('Feature Values of Iris Dataset');
hold off;

% (B) Learn and plot 1-nearest-neighbour predictor
learner = knnClassify(1, X, Y);
class2DPlot(learner,X,Y);
title('1-nearest-neighbour predictor');

% (C) Repeat for several values of k
kValues = [3,10,30]; % k=1 is already plotted above in (B).
for index = 1:length(kValues)
    learner = knnClassify(kValues(index), X, Y);
    class2DPlot(learner,X,Y);
    title(strcat(int2str(kValues(index)), '-nearest-neighbour predictor'));
end

% (D) Split data into training (80%) and valuation (20%) data. Train and 
%     validate model for multiple values of k and calculate its
%     performance.
Xtrain = X(1:118, 1:end);
Xvalid = X(119:end, 1:end);
Ytrain = Y(1:118);
Yvalid = Y(119:end);
kValues = [1, 2, 5, 10, 50, 100, 200];
errors = [];
for index = 1:length(kValues)
    learner = knnClassify(kValues(index), Xtrain, Ytrain); % train model on X/Ytrain
    Yhat = predict(learner, Xvalid); % predict results on Xtrain/Yvalid
    errors = [errors, numel(find(Yhat~=Yvalid))]; % count what fraction of predictions are wrong   
    hold on;
end

figure('Name','Performace of k'); % Plot performance of k against error
plot(kValues, errors,'-o')
title('Performance of k against error rate');
xlabel('Value of k')
ylabel('Number of data points classified incorrectly') % y-axis label





%% 5. Perceptrons and Logistic Regression


iris = load('data/iris.txt'); % load the text file
X = iris(:, 1:2); Y = iris(:, end); % get first two features
[X Y] = shuffleData(X, Y); % reorder randomly
X = rescale(X); % works much better for rescaled data
XA = X(Y<2, :); YA=Y(Y<2); % get class 0 vs 1
XB = X(Y>0, :); YB=Y(Y>0); % get class 1 vs 2

% (A) Show the two classes in a scatter plot and verify that one is 
% linearly separable while the other is not.

figure('Name','Class A');
hold on;
classZeroIndicies = find(YA==0);
xPointsClassZero = XA(classZeroIndicies, 1:end);
scatter(xPointsClassZero(:, 1), xPointsClassZero(:, 2));
classOneIndicies = find(YA==1);
xPointsClassOne = XA(classOneIndicies, 1:end);
scatter(xPointsClassOne(:, 1), xPointsClassOne(:, 2));
hold off;

figure('Name','Class B');
hold on;
% classOneIndicies = find(YB==1);
% xPointsClassOne = XB(classOneIndicies, 1:end);
scatter(xPointsClassOne(:, 1), xPointsClassOne(:, 2));
classTwoIndicies = find(YB==2);
xPointsClassTwo = XB(classTwoIndicies, 1:end);
scatter(xPointsClassTwo(:, 1), xPointsClassTwo(:, 2));
hold off;


%% (B) 

learner=logisticClassify2(); % create "blank" learner
learner=setClasses(learner, unique(YA)); % define class labels using YA or YB
wts = [0.5 1 -0.25]; % TODO: fill in values
learner=setWeights(learner, wts); % set the learner's parameters
plot2DLinear(learner, XA, YA);







% Note: Be sure to shuffle your data before doing SGD in part (f)  otherwise, 
% if the data are in a pathological ordering (e.g., ordered by class), you 
% may experience strange behavior and slow convergence during the optimization.

























