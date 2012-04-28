function [ recItems ] = mf_gd( trainMatrix, featureNumber, maxEpoch, learnRate, lambdaU, lambdaV, k)
%risk function: trainMatrix <- Ut * Vt'

%get the size the train matrix
[userNumber,itemNumber] = size(trainMatrix);

%init user factors and item factors
Ut = 0.01 * randn(userNumber, featureNumber);
Vt = 0.01 * randn(itemNumber, featureNumber);

logitMatrix = trainMatrix > 0;

%calculate the gradient of user factors and item factors
%and user sgd to optimize the risk function
%alternative update user factors and item factors alternative
for round = 1:maxEpoch,
   dU = -(logitMatrix  .* trainMatrix)  * Vt + (Ut * Vt' .* logitMatrix ) * Vt + lambdaU * Ut;
   dV = -(logitMatrix' .* trainMatrix') * Ut + (Vt * Ut' .* logitMatrix') * Ut + lambdaV * Vt;
   Ut = Ut - learnRate * dU * 2;
   Vt = Vt - learnRate * dV * 2;
end

%predict the rating of each item given by each user
predictMatrix = Ut * Vt';

%sort the score of items for each user
[sortedMatrix, sortedItems] = sort(predictMatrix, 2, 'descend');

%get the top-k items for each suer
recItems = sortedItems(:, 1:k);

end
