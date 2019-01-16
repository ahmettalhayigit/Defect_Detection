##### Introduction #####

# Ahmet Talha Yiðit
# 070130262

# The codes should be runned in the order as they are written, otherwise problems can occur,
#because i used same names for different arrays sometimes.


##### Libraries #####

library(openxlsx)
library("caret")
library("FactoMineR")
library("e1071")
library("gbm")
library(data.table)
library(mlr)
library("xgboost")
library(randomForest)
library(rpart)
library(tree)
library(class)



##### Data Preparetion #####
### Read and Prepare Data  ###

set.seed(1)

Defect<- read.xlsx("defect_competition.xlsx") # Read full dataset
smp_size <- floor(0.80 * nrow(Defect))      # Define % of training and test set
train_ind <- sample(seq_len(nrow(Defect)), size = smp_size)   # Sample rows
train <- Defect[train_ind, ]      # Get training set
test <- Defect[-train_ind, ]      # Get test set

write.xlsx(train, "train.xlsx")    # Write and Read again the subsets to have them ready to check 
write.xlsx(test, "test.xlsx")      # in case of a problem
train<-read.xlsx("train.xlsx")
test<-read.xlsx("test.xlsx")

setDT(Defect) # Setting data tables
setDT(train)
setDT(test)

# Create model matrixes which convert all factor variables to dummy variables,  
# and exclude "IsDefective".
new_df <- model.matrix(~.+0,data = Defect[,-c("IsDefective"),with=F])
new_tr <- model.matrix(~.+0,data = train[,-c("IsDefective"),with=F])
new_ts <- model.matrix(~.+0,data = test[,-c("IsDefective"),with=F])


k=1 # Check if we have the same variables both in train and test sets
for (i in colnames(new_tr)) {
  check<-(i %in% colnames(new_ts))
  print(check)
  print(k)
  k=k+1
} # Yes, all variables in train set found in test set too.

labels <- train$IsDefective #Define labels
ts_label <- test$IsDefective
df_label <- Defect$IsDefective

# Convert labels to numeric "0" and "1" format.
labels<-(labels=="Defective")
labels<-as.numeric(labels)
ts_label<-(ts_label=="Defective")
ts_label<-as.numeric(ts_label)
df_label<-(df_label=="Defective")
df_label<-as.numeric(df_label)

# Create xgb.DMatrixes to use in XGBoost method.
ddefect <- xgb.DMatrix(data = new_df,label = df_label)
dtrain <- xgb.DMatrix(data = new_tr,label = labels)
dtest <- xgb.DMatrix(data = new_ts,label= ts_label)



##### XGBoost #####
### Caret Based Parameter Selection ###

# Define the grid with parameters.
xgbGrid <-  expand.grid(nrounds = c(10,100), 
                        max_depth = c(4,5,6, 10), 
                        eta = 0.3,
                        gamma = c(0,1), colsample_bytree=1,
                        min_child_weight=1, subsample=1)

# Define the control method, here it is 5 fold crossvalidation.
fitControl <- trainControl(
  method = "repeatedcv",
  number = 5,
  # Repeated ten times
  repeats = 10)


# Create all possible models with possible parameters.
gbmFit <- caret::train(new_tr, as.factor(labels), method = "xgbTree", 
                       trControl = fitControl, verbose = T, 
                       tuneGrid = xgbGrid)

varImp(gbmFit) # Check variable importance.

plot(gbmFit) # Plot results to see accuracy levels.                      
plot(gbmFit, plotType = "level")

gbmFit$results # Print all results.

best(gbmFit$results, metric="Accuracy", maximize=T) # Find the best model and parameters.



### XGBoost

# Set the parameters found from above method.
params <- list(booster = "gbtree", objective = "binary:logistic", 
               eta=0.3, gamma=1, max_depth=4, min_child_weight=1, 
               subsample=1, colsample_bytree=1)

# Train XGBoost model with these parameters
xgbcv <- xgb.cv( params = params, data = dtrain, nrounds = 100, 
                 nfold = 5, showsd = T, stratified = T, 
                 print_every_n = 1, early_stopping_rounds = 20, maximize = F)

xgbcv$best_iteration # Find the best iteration created in model

# Take the best iteration as a model
xgb1 <- xgb.train (params = params, data = dtrain, 
                   nrounds = xgbcv$best_iteration)

# Predictions on test set
xgbpred <- predict (xgb1,dtest)
# Assign results to "1" or "0" by looking if they are higher or lower than 0.5
lastxgbpred <- ifelse (xgbpred > 0.5,1,0)

#Create confusion matrix and calculate accuracy
table(lastxgbpred, ts_label)
mean(lastxgbpred==ts_label) # 100% accuracy on test set

# Define Variable importance matrix
mat <- xgb.importance (feature_names = colnames(new_tr),model = xgb1)
# Print the most important five variables
xgb.plot.importance (importance_matrix = mat[1:5])
# Only "Thickness" variable is important in this model

# Plot the tree created by XGBoost 
xgb.plot.tree(model = xgb1) 



### Test on Full Dataset

# Predictions on Full Data set
xgbpredfull <- predict (xgb1,ddefect)
# Assign results to "1" or "0" by looking if they are higher or lower than 0.5
lastxgbpredfull <- ifelse (xgbpredfull > 0.5,1,0)

#Create confusion matrix
table(lastxgbpredfull, df_label)

#Calculate accuracy
mean(lastxgbpredfull==df_label)



##### Logistic Regression #####

# The dataset has already read in the beginning.
set.seed(1)
smp_size <- floor(0.80 * nrow(Defect))      # Define % of training and test set
train_ind <- sample(seq_len(nrow(Defect)), size = smp_size)   # Sample rows
train3 <- Defect[train_ind, ]      # Get training set
test3 <- Defect[-train_ind, ]      # Get test set

# Turn format of "IsDefective" to numeric "0" and "1".
labels <- train3$IsDefective
ts_label <- test3$IsDefective
train3$IsDefective=(labels=="Defective")
train3$IsDefective<-as.numeric(train3$IsDefective)
test3$IsDefective=(ts_label=="Defective")
test3$IsDefective<-as.numeric(test3$IsDefective)

# Create the logistic model.
logreg=glm(IsDefective~.,data=train3, family = binomial)
summary(logreg)

# Predictions on Test set.
glm.probs=predict(logreg ,type="response",newdata = test3)
glm.probs
glm.pred=rep("0", 66)
glm.pred[glm.probs > 0.5]="1"
# Create confusion matrix.
table(glm.pred, test3$IsDefective)
# Calculate Accuracy
mean(glm.pred==test3$IsDefective)

# Predictions on Full data set.
glm.probs=predict(logreg ,type="response",newdata = Defect)
glm.probs
glm.predfull=rep("0", 327)
glm.predfull[glm.probs > 0.5]="1"
# Create confusion matrix.
table(glm.predfull, df_label)
# Calculate Accuracy on full dataset.
mean(glm.predfull==df_label)



##### Random Forest #####

set.seed(1)

#Create a random forest model.
forest.model <- randomForest(x = new_tr, y = as.factor(labels), 
                             importance=TRUE,ntree=1500,
                             mtry=19)
#Summary of the model.
forest.model

#Check importance of variables.
importance(forest.model)        
varImpPlot(forest.model)  

#Find the model with the best parameters and accuracy. 
a=c()
i=1
for (i in 1:19) {
  forest.model.param <- randomForest(x = new_tr, y = as.factor(labels),  ntree = 500, mtry = i, importance = TRUE)
  predValid <- predict(forest.model.param, new_ts, type = "class")
  a[i-2] = mean(predValid == as.factor(ts_label))
}
# Best accuracy on test data
max(a)

df_label1 <- Defect$IsDefective
#See result on Full dataset.
rflastpredfull<-predict(forest.model.param, new_df, type = "class")
table(rflastpredfull, df_label1)
mean (rflastpredfull==df_label1)



##### K- Nearest Neighbors #####
set.seed(1)
labels1<-as.factor(labels)
ts_label1<-as.factor(ts_label)

#Find the best number of "K" value.
accuracy<-1:200
for (i in 1:200){
knn.pred=knn(new_tr,new_ts,labels1  ,k=i)
accuracy[i]<-mean(knn.pred ==ts_label1)
}
print(which.max(accuracy))

# k=18 maximize
# Prediction on test data
knn.pred=knn(new_tr,new_ts,labels1  ,k=18)
mean(knn.pred ==ts_label1)
table(knn.pred,ts_label1)

# Prediction on full data
knn.predfull<-knn(new_tr,new_df,labels1  ,k=18)
mean(knn.predfull ==df_label1)
table(knn.predfull,df_label1)



##### Decision Trees #####

#Data preparetion
set.seed(1)
defecttreetrain =data.frame(new_tr ,labels1)
defecttreetest <- data.frame(new_ts ,ts_label1)
defecttreefull <- data.frame(new_df ,df_label1)

#Create the decision tree model
model_dt = tree(labels1~.,data=defecttreetrain  )
summary(model_dt )
plot(model_dt )
text(model_dt ,pretty =0)

#Prediction on test data.
model_dt_pred <- predict(model_dt ,defecttreetest ,type="class")
table(model_dt_pred ,ts_label1)
mean (model_dt_pred==ts_label1)

#Prediction on full dataset.
model_dt_predfull <- predict(model_dt ,defecttreefull ,type="class")
table(model_dt_predfull ,df_label1)
mean (model_dt_predfull==df_label1)



##### Ensemble Model by Majority Vote Method #####
set.seed(1)

# Turn all result to "0" and "1" format.
rflastpredfull<- ifelse (rflastpredfull == "Defective",1,0)
knn.predfull<- ifelse (knn.predfull == "Defective",1,0)
model_dt_predfull<- ifelse (model_dt_predfull == "Defective",1,0)
glm.predfull<-as.numeric(glm.predfull)


lastxgbpredfull
glm.predfull
rflastpredfull
knn.predfull
model_dt_predfull
#First ensemble of random forest, knn, and dt.
pred_majority1<-as.factor(ifelse(rflastpredfull==1 & knn.predfull==1,1,ifelse(knn.predfull==1 & model_dt_predfull==1,1,ifelse(model_dt_predfull==1 & rflastpredfull==1,1,0))))
pred_majority1
#Second ensemble of first ensemble, XGBoost, and logistic
pred_majority2<-as.factor(ifelse(lastxgbpredfull==1 & glm.predfull==1,1,ifelse(glm.predfull==1 & pred_majority1==1,1,ifelse(pred_majority1==1 & lastxgbpredfull==1,1,0))))
pred_majority2
#Final ensemble of first ensemble, second ensemble, and XGBoost
pred_majority3<-as.factor(ifelse(lastxgbpredfull==1 & glm.predfull==1,1,ifelse(glm.predfull==1 & pred_majority2==1,1,ifelse(pred_majority1==2 & lastxgbpredfull==1,1,0))))
pred_majority3

#Create confusion matrix of final ensemble
table(pred_majority3, df_label)

#Calculate accuracy of final ensemble
mean(pred_majority3==df_label)



##### Comments #####

# 5 different classification methods has been used to find the best solution for the problem.

# For "XGBoost" method the accuracy for full dataset = 0.9938838 
# For "Logistic Regression" method the accuracy for full dataset = 0.9449541
# For "Random Forest" method he accuracy for full dataset = 0.6299694
# For "K- Nearest Neighbors" method the accuracy for full dataset = 0.6666667
# For "Decision Trees" method the accuracy for full dataset = 0.7155963

# After looking at the resulted accuracies, XGBoost method is the best method.
# But it made the classification only regarding "Thickness" variable, and ignored others.
# In case of overfitting, and to utilize other variables as well an ensemble model created.

# It is created by "majority vote" method. and the accuracy on full data is 0.9602446.
# This ensemble method can have better results on other dataset, 
#because it is less likely to overfit our existing data.

# In conclusion the best result is 0.9938838 accuracy but ensemble model can be a better model
#in terms of using on other datasets.