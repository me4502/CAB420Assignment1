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




% 5. Perceptrons and Logistic Regression

