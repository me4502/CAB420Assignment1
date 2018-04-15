%% CAB420 Assignment 1
%
% Authors: 
% - Jarrod Williams, n9722068
% - Madeline Miller, n9342401
% 
%
%% 1. Features, Classes, and Linear Regression
disp('1. Features, Classes, and Linear Regression');

% (a) Plot the training data in a scatter plot.
mTrain = load('data/mcycleTrain.txt'); % Load motorcycle training dataset
ytr = mTrain(: ,1); xtr = mTrain(: ,2); % Separate features; x = single fature, y = target value
figure('name', 'Training Data');
plot (xtr, ytr, 'bo'); % Plot training data
xlabel('x');
ylabel('y');
title('Scatter Plot of Training Data');

% (b) Create a linear predictor (slope and intercept) using the above
%    functions. Plot it on the same plot as the training data.
Xtr = polyx(xtr, 2);
learner_quadratic = linearReg(Xtr, ytr); % Create and learn a regression predictor from the data Xtr, ytr.
xline = [0:.01:2]' ; % Transpose
yline = predict(learner_quadratic, polyx(xline, 2)); % Assuming quadratic features
figure('name', 'Quadratic linear predictor');
plot(xline, yline, 'ro');
hold on % Plot training data and label figure.
plot (xtr, ytr, 'bo');
legend('Linear Predictor', 'Training Data');
title('Quadratic linear predictor');

% (c) Create another plot with the data and a fifth-degree polynomial.
Xtr = polyx(xtr, 5);
learner_quintic = linearReg (Xtr , ytr);
yline = predict (learner_quintic , polyx(xline ,5)); % assuming quintic features
figure('name', 'Quintic linear predictor');
plot ( xline , yline ,'ro ');
hold on
plot (xtr, ytr, 'bo');
legend('Linear Predictor', 'Training Data');
title('Quintic linear predictor');

% (d) Calculate the mean squared error associated with each of your learned 
%     models on the training data.
% Quadratic
Xtr = polyx(xtr, 2);
yhat = predict(learner_quadratic, Xtr);
mseQuadTrain = immse(yhat, ytr);
fprintf('The MSE for the quadratic linear predictor on training data was: %.2f\n', mseQuadTrain);
% Quintic
Xtr = polyx(xtr, 5);
yhat = predict(learner_quintic, Xtr);
mseQinTrain = immse(yhat, ytr);
fprintf('The MSE for the quintic linear predictor on training data was: %.2f\n', mseQinTrain);

% (e) Calculate the MSE for each model on the test data (in mcycleTest.txt).
mTest = load('data/mcycleTest.txt');
ytest = mTest(: ,1); xtest = mTest(: ,2);
% Quadratic
Xtest = polyx(xtest, 2);
yhat = predict(learner_quadratic, Xtest);
mseQuadTest = immse(yhat, ytest);
fprintf('The MSE for the quadratic linear predictor on test data was: %.2f\n', mseQuadTest);
% Quintic
Xtest = polyx(xtest, 5);
yhat = predict(learner_quintic, Xtest);
mseQuinTest = immse(yhat, ytest);
fprintf('The MSE for the quintic linear predictor on test data was: %.2f\n', mseQuinTest);


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
    
    % Plot newly created knn regress output at x points.
    yline = predict(learner, xline);
    plot(xline, yline, '', 'DisplayName', strcat('K=', num2str(i)));
end

%% 3. Hold-out and Cross-validation

ytr = mTrain(: ,1); xtr = mTrain(: ,2);
ytest = mTest(: ,1); xtest = mTest(: ,2);

xtrMin = xtr(1:20, :); ytrMin = ytr(1:20, :);

knnMses = zeros(3, 100);

for k=1:100
    learnerMin = knnRegress(k, xtrMin, ytrMin);
    yhatMin = predict(learner, xtest);
    
    knnMses(1,k) = immse(yhatMin, ytest);
    
    learnerAll = knnRegress(k, xtr, ytr);    
    yhatAll = predict(learner, xtest);

    knnMses(2,k) = immse(yhatAll, ytest);
    
    for i=1:4
        start = 20*(i - 1) + 1;
        endIndex = start + 19;
        crossTest = mTrain(start:endIndex, :);
        crossTrain = setdiff(1:80, crossTest);
        ytrCross = crossTrain(: ,1); xtrCross = crossTrain(: ,2);
        ytestCross = crossTest(: ,1); xtestCross = crossTest(: ,2);
        learnerCross = knnRegress(k, xtrCross, ytrCross);
        
        yhatCross = predict(learnerCross, xtestCross);

        knnMses(3,k) = immse(yhatCross, ytestCross);
    end
end

% Plot training data
figure('name', 'kNN MSE');

loglog(1:100, knnMses(1, :));
hold on;
loglog(1:100, knnMses(2, :));
loglog(1:100, knnMses(3, :));

xlabel('K');
ylabel('Mean Squared Error');
legend('20 Training Data Points', 'All Training Data Points', 'Cross Validation');

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
title('Class A');
hold off;

figure('Name','Class B');
hold on;
% classOneIndicies = find(YB==1);
% xPointsClassOne = XB(classOneIndicies, 1:end);
scatter(xPointsClassOne(:, 1), xPointsClassOne(:, 2));
classTwoIndicies = find(YB==2);
xPointsClassTwo = XB(classTwoIndicies, 1:end);
scatter(xPointsClassTwo(:, 1), xPointsClassTwo(:, 2));
title('Class B');
hold off;


%% (B) Write the function @logisticClassify2/plot2DLinear.m such that it 
%      can Plot the two classes of data in different colors, along with the 
%      decision boundary (a line). To demo your function plot the decision 
%       boundary corresponding to the classifier sign( .5 + 1x1 ? .25x2 )
%       along with the A data, and again with the B data.

learner=logisticClassify2(); % create "blank" learner
learner=setClasses(learner, unique(YA)); % define class labels using YA or YB
wts = [0.5 1 -0.25]; % TODO: fill in values
learner=setWeights(learner, wts); % set the learner's parameters
figure('Name','Linear Plot');
plot2DLinear(learner, XA, YA);
title('Class A with Decision Boundary');
figure('Name','Linear Plot');
plot2DLinear(learner, XB, YB);
title('Class B with Decision Boundary');


%% (C) Complete the predict.m function to make predictions for your linear 
%      classifier.  Verify that your function works by computing & 
%      reporting the error rate of the classifier in the previous
%      part on both data sets A and B. (The error rate on data set A should
%      be ? 0.0505.)

% Data set A
yte = predict(learner,XA);
classError = 0;
for i=1:size(YA,1);
    if(YA(i) ~= yte(i));
        classError = classError + 1;
    end;
end;
finalClassError = classError/size(YA,1); % = 0.0505
disp(strcat({'The error rate for class A is:'},{' '},{num2str(finalClassError,' %.4f')}));
% Data set B
yte = predict(learner,XB);
classError = 0;
for i=1:size(YB,1);
    if(YB(i) ~= yte(i));
        classError = classError + 1;
    end;
end;
finalClassError = classError/size(YB,1); % = 0.54555
disp(strcat({'The error rate for class B is:'},{' '},{num2str(finalClassError,' %.4f')}));

%% (D) Refer to report.


%% (E) Implemented train.m


%% (F) 

% Train Class A
train(learner, XA, YA);
legend('Error Rate', 'Surrogate Loss');
% Plot final converged classifier decision boundaries.
figure();
plotClassify2D(learner, XA, YA);

%% Train Class B
train(learner, XB, YB);
legend('Error Rate', 'Surrogate Loss');
% Plot final converged classifier decision boundaries.
figure();
plotClassify2D(learner, XB, YB);
