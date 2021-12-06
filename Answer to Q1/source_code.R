library(caret)
library(ISLR)
library(boot)
library(class)
library(e1071)
library(neuralnet)
library(Boruta)
library(Rcpp)

##Data preparation. Take a fraction 0.8 of digit 0 and a fraction of 0.2 of digit 7.
Data = read.csv(file="/Users/as/Desktop/mnist_test.csv")
D0 = Data[Data$label == 0,]
D7 = Data[Data$label == 7,]
set.seed(1)
index0 = sample(c(1:length(D0)),80, replace = FALSE)
index7 = sample(c(1:length(D7)),20, replace = FALSE)
Bag = rbind(D0[index0,], D7[index7,])
label = Bag$label


##Train a model

# Feature selection with Boruta algorithm.
Bag$label = as.factor(Bag$label)
boruta <- Boruta(label~.,data=Bag, doTrace=2,maxRuns = 500)
print(boruta) #79 attritubes are comfirmed important.
Bag = cbind(label,Bag[,(boruta$finalDecision=="Confirmed")] )#A feature matrix is obtained with size 79X100.


# Stage Three: Neural Network for prediction of Y.
set.seed(1)
Set <- createFolds(Bag$X6x14,k=10,list=TRUE, returnTrain=FALSE)
error = c()
for (layer in 1:10){
  E=0
  for (i in 1:10){
    validation <- Bag[Set[[i]],]
    train <- Bag[-Set[[i]],]
    nn = neuralnet(label~.,data=train,hidden=layer,act.fct="logistic",linear.output = FALSE)
    predict=compute(nn, validation)
    prob <- predict$net.result
    prob[prob == max(prob)] = 7
    prob[prob != max(prob)] = 0
    Predicted_Y = sum(prob==0)/length(prob)
    True_Y = sum(validation==0)/length(validation)
    
    E <- E+ (Predicted_Y - True_Y)^2
  }
  error <- c(error,E)
}
error <- error/10

nn = neuralnet(label~.,data=train,hidden=10,act.fct="logistic",linear.output = FALSE)
plot(nn) #To plot the result model.



## Test the model.

Error = c()
for (i in c(1:10)){
  set.seed(i)
  index0 = sample(c(1:length(D0)),80, replace = FALSE)
  index7 = sample(c(1:length(D7)),20, replace = FALSE)
  test = rbind(D0[index0,], D7[index7,])
  nn = neuralnet(label~.,data=test,hidden=10,act.fct="logistic",linear.output = FALSE)
  predict=compute(nn, test)
  prob[prob == max(prob)] = 7
  prob[prob != max(prob)] = 0
  Predicted_Y = sum(prob==0)/length(prob)
  True_Y = sum(validation==0)/length(validation)
  
  Error = c(Error, (Predicted_Y - True_Y)^2)
}
RSS = 1/10*(sum(Error))





