
#Stacking -- Running xgboost again with dnn blend output as input variable

install.packages('zoo')
setwd("/Users/jp/Documents/Kaggle/All State Severity Claims")

library(data.table)
library(Matrix)
library(Metrics)
library(xgboost)
library(e1071)
library(scales)
library(stringr)
library(dplyr)
library(Hmisc)
library(caret)
library(ggplot2)
library(forecast)
TRAIN_FILE = "C:/Users/HP/Desktop/Kaggle/All State Severity Claims/train.csv"
TEST_FILE = "C:/Users/HP/Desktop/Kaggle/All State Severity Claims/test.csv"
SUBMISSION_FILE = "C:/Users/HP/Desktop/Kaggle/All State Severity Claims/sample_submission.csv"


train = fread(TRAIN_FILE, showProgress = TRUE)
test = fread(TEST_FILE, showProgress = TRUE)
#test$loss = 'NA'#
submission = fread(SUBMISSION_FILE, showProgress = TRUE)
ID = 'id'
TARGET = 'loss'
SEED = 0
SHIFT = 200

ntrain = nrow(train)
y_train = log(train[,TARGET, with = FALSE] + SHIFT)[[TARGET]]
y_train_stack1 = log(train[1:(ntrain/2),TARGET, with = FALSE] + SHIFT)[[TARGET]]
y_train_stack2 = log(train[((ntrain/2)+1):ntrain,TARGET, with = FALSE] + SHIFT)[[TARGET]]

#y_train = as.data.frame(y_train)
train[, c(ID, TARGET) := NULL]
test[, c(ID) := NULL]
train_test = rbind(train,test,fill=TRUE)


# remove skewness
for (f in colnames(train_test)[colnames(train_test) %like% "^cont"]) {
  
  tst <- e1071::skewness(train_test[, eval(as.name(f))])
  if (tst > .25) {
    if (is.na(train_test[, BoxCoxTrans(eval(as.name(f)))$lambda])) next
    lambda = BoxCoxTrans((train_test[,eval(as.name(f))]))$lambda
    train_test[,f] = BoxCox(train_test[,eval(as.name(f))],lambda)
  }
}

# scale

for (f in colnames(train_test)[colnames(train_test) %like% "^cont"]){
  train_test[, f] = scale(train_test[,eval(as.name(f))])
}


features = names(train)

for (f in features) {
  if (class(train_test[[f]])=="character") {
    #cat("VARIABLE : ",f,"\n")
    levels <- sort(unique(train_test[[f]]))
    train_test[[f]] <- as.integer(factor(train_test[[f]], levels=levels))
  }
}


# in order to speed up fit within Kaggle scripts have removed 30
# least important factors as identified from local run
features_to_drop <- c("cat67","cat21","cat60","cat65", "cat32", "cat30",
                      "cat24", "cat74", "cat85", "cat17", "cat14", "cat18",
                      "cat59", "cat22", "cat63", "cat56", "cat58", "cat55",
                      "cat33", "cat34", "cat46", "cat47", "cat48", "cat68",
                      "cat35", "cat20", "cat69", "cat70", "cat15", "cat62")

x_train = train_test[1:ntrain,-features_to_drop, with = FALSE]
x_test = train_test[(ntrain+1):nrow(train_test),-features_to_drop, with = FALSE]

# Splitting train dataset into two


x_train_set1 = x_train[1:(ntrain/2),]
x_train_set2 = x_train[((ntrain/2)+1):ntrain,]



dtrain_stack1 = xgb.DMatrix(as.matrix(x_train_set1), label=y_train_stack1)
dtrain_stack2 = xgb.DMatrix(as.matrix(x_train_set2), label=y_train_stack2)
dtest_stack1 = xgb.DMatrix(as.matrix(x_train_set1))
dtest_stack2 = xgb.DMatrix(as.matrix(x_train_set2))

xg_eval_mae <- function (yhat, dtrain) {
  y = getinfo(dtrain, "label")
  err= mae(exp(y),exp(yhat) )
  return (list(metric = "error", value = err))
}

# fair objective 2 for XGBoost

amo.fairobj2 <- function(preds, dtrain) {
  
  labels <- getinfo(dtrain, "label")
  con <- 2
  x <- preds - labels
  grad <- con * x / (abs(x) + con)
  hess <- con ^ 2 / (abs(x) + con) ^ 2
  
  return(list(grad = grad, hess = hess))
  
}

xgb_params = list(
  colsample_bytree = 0.7,
  subsample = 0.7,
  eta = 0.03, # replace this with 0.01 for local run to achieve 1113.93
  objective = amo.fairobj2,
  max_depth = 12,
  alpha = 1,
  gamma = 2,
  min_child_weight = 100,
  booster='gbtree'
)

best_nrounds = 1500


gbdt_stack1 = xgb.train(xgb_params, dtrain_stack1, nrounds=as.integer(best_nrounds))

prediction1 = exp(predict(gbdt_stack1,dtest_stack2)) - SHIFT

submission_stack1 = submission[1:(ntrain/2),]

submission_stack1$loss = prediction1

write.csv(submission_stack1,'stack_train2output_xgboost-v2.csv',row.names = FALSE)

# train 2

submission = fread(SUBMISSION_FILE, showProgress = TRUE)

gbdt_stack2 = xgb.train(xgb_params, dtrain_stack2, nrounds=as.integer(best_nrounds))
prediction2 = exp(predict(gbdt_stack2,dtest_stack1)) - SHIFT

submission_stack2 = submission[((ntrain/2)+1):ntrain,]
submission_stack2$loss = prediction2

write.csv(submission_stack2,'stack_train1output_xgboost-v2.csv',row.names = FALSE)

# Running same model on test set
submission_stack_test = fread(SUBMISSION_FILE, showProgress = TRUE)
dtrain_stack_test= xgb.DMatrix(as.matrix(x_train), label=y_train)
dtest_stack_test = xgb.DMatrix(as.matrix(x_test))

gbdt_stack3_test = xgb.train(xgb_params, dtrain_stack_test, nrounds=as.integer(best_nrounds))
prediction3 = exp(predict(gbdt_stack3_test,dtest_stack_test)) - SHIFT
submission_stack_test$loss = prediction3
write.csv(submission_stack_test,'stack_whole_test_output-v2.csv',row.names = FALSE)

#Combining predictions 1 & 2 into original training set to feed into level 2 model
prediction1 = as.data.frame(prediction1)
prediction2 = as.data.frame(prediction2)
colnames(prediction1) = 'train_pred'
colnames(prediction2) = 'train_pred'

combined_prediction = data.frame('combined_pred' = numeric(188318))
combined_prediction$combined_pred = rbind(prediction1,prediction2)

#x_train$first_level_preds = combined_prediction$combined_pred
#x_test$first_level_preds = prediction3


# Second layer DNN models
setwd("C:/Users/HP/Desktop/Kaggle/All State Severity Claims")
library(ggplot2) # Data visualization
library(readr) 
library(data.table)

# Load H2O
library(h2o)
kd_h2o<-h2o.init(nthreads = -1, max_mem_size = "4g")


set.seed(12346)
TRAIN_FILE = "C:/Users/HP/Desktop/Kaggle/All State Severity Claims/train.csv"
TEST_FILE = "C:/Users/HP/Desktop/Kaggle/All State Severity Claims/test.csv"
SUBMISSION_FILE = "C:/Users/HP/Desktop/Kaggle/All State Severity Claims/sample_submission.csv"

#Reading Data, old school read.csv. Using fread is faster. 
train = read.csv("C:/Users/HP/Desktop/Kaggle/All State Severity Claims/train.csv")
test = read.csv("C:/Users/HP/Desktop/Kaggle/All State Severity Claims/test.csv")
SUBMISSION_FILE = read.csv("C:/Users/HP/Desktop/Kaggle/All State Severity Claims/sample_submission.csv",colClasses = c("integer", "numeric"))
train_ouput1 = read.csv("C:/Users/HP/Desktop/Kaggle/All State Severity Claims/stack_train1output_xgboost.csv")
train_ouput2 = read.csv("C:/Users/HP/Desktop/Kaggle/All State Severity Claims/stack_train2output_xgboost.csv")
test_ouput_firstlayer = read.csv("C:/Users/HP/Desktop/Kaggle/All State Severity Claims/stack_whole_test_output.csv")

# remove skewness in train
for (f in colnames(train)[colnames(train) %like% "^cont"]) {
  
  tst <- e1071::skewness(train[, eval(as.name(f))])
  if (tst > .25) {
    if (is.na(train[, BoxCoxTrans(eval(as.name(f)))$lambda])) next
      lambda = BoxCoxTrans((train[,eval(as.name(f))]))$lambda
        train[,f] = BoxCox(train[,eval(as.name(f))],lambda)
    }
}


# remove skewness in test
for (f in colnames(test)[colnames(test) %like% "^cont"]) {
  
  tst <- e1071::skewness(test[, eval(as.name(f))])
  if (tst > .25) {
    if (is.na(test[, BoxCoxTrans(eval(as.name(f)))$lambda])) next
    lambda = BoxCoxTrans((test[,eval(as.name(f))]))$lambda
    test[,f] = BoxCox(test[,eval(as.name(f))],lambda)
  }
}


# scale for train

for (f in colnames(train)[colnames(train) %like% "^cont"]){
  train[, f] = scale(train[,eval(as.name(f))])
}


# scale for test

for (f in colnames(test)[colnames(test) %like% "^cont"]){
  test[, f] = scale(test[,eval(as.name(f))])
}

train<-train[,-1]
test_id<-test[,1]
test<-test[,-1]

combined_prediction = rbind(train_ouput1,train_ouput2)
train$first_level_preds = combined_prediction$loss
test$first_level_preds = test_ouput_firstlayer$loss

features_to_drop <- c("cat67","cat21","cat60","cat65", "cat32", "cat30",
                      "cat24", "cat74", "cat85", "cat17", "cat14", "cat18",
                      "cat59", "cat22", "cat63", "cat56", "cat58", "cat55",
                      "cat33", "cat34", "cat46", "cat47", "cat48", "cat68",
                      "cat35", "cat20", "cat69", "cat70", "cat15", "cat62")

`%ni%` <- Negate(`%in%`)
train = subset(train,select = names(train) %ni% features_to_drop)
test = subset(test,select = names(test) %ni% features_to_drop)
train = subset(train, select=c(102,1:101))
test =  subset(test, select=c(101,1:100))

#nrow(train)
#nrow(test)
#which(colnames(train3)=='first_level_preds')


index<-sample(1:(dim(train)[1]), 0.2*dim(train)[1], replace=FALSE)
#nrow(train)
train_frame<-train[-index,]
valid_frame<-train[index,]


valid_predict<-valid_frame[,-ncol(train_frame)]
valid_loss<-valid_frame[,ncol(train_frame)]

#View(valid_predict)

# log transform
train_frame[,ncol(train_frame)]<-log(train_frame[ncol(train_frame)])
valid_frame[,ncol(train_frame)]<-log(valid_frame[,ncol(train_frame)])

#View(train_frame)

# load H2o data frame // validate that H2O flow looses all continous data
train_frame.hex<-as.h2o(train_frame)
valid_frame.hex<-as.h2o(valid_frame)
valid_predict.hex<-as.h2o(valid_predict)
test.hex<-as.h2o(test)

#View(train_frame.hex)
#which(colnames(valid_frame)=='loss')
## DNN Neural net 1 // increase epochs for higher accuracy

start<-proc.time()
dnn_model_1<-h2o.deeplearning(x=1:101, y=102, 
                              training_frame=train_frame.hex, validation_frame=valid_frame.hex,
                              epochs=20, 
                              stopping_rounds=3,
                              overwrite_with_best_model=T,
                              activation="Rectifier",
                              distribution="huber",
                              hidden=c(91,91))

print("DNN_1 runtime:")
print(proc.time()-start)
pred_dnn_1<-as.matrix(predict(dnn_model_1, valid_predict.hex))
score_dnn_1=mean(abs(exp(pred_dnn_1)- valid_loss))
cat("score_dnn_1:",score_dnn_1,"\n")

## DNN Neural net 2 // increase epochs for higher accuracy
start<-proc.time()
dnn_model_2<-h2o.deeplearning(x=1:101, y=102, 
                              training_frame=train_frame.hex, validation_frame=valid_frame.hex,
                              epochs=100, 
                              stopping_rounds=4,
                              overwrite_with_best_model=T,
                              activation="Rectifier",
                              distribution="huber",
                              hidden=c(101,101,101))
print("DNN_2 runtime:")
print(proc.time()-start)
pred_dnn_2<-as.matrix(predict(dnn_model_2, valid_predict.hex))
score_dnn_2=mean(abs(exp(pred_dnn_2)-valid_loss))
cat("score_dnn_2:",score_dnn_2,"\n")


## DNN Neural net 3 // increase epochs for higher accuracy
start<-proc.time()
dnn_model_3<-h2o.deeplearning(x=1:101, y=102, 
                              training_frame=train_frame.hex, validation_frame=valid_frame.hex,
                              epochs=60, 
                              stopping_rounds=2,
                              overwrite_with_best_model=T,
                              activation="Rectifier",
                              distribution="huber",
                              hidden=c(53,53))
print("DNN_3 runtime:")
print(proc.time()-start)
pred_dnn_3<-as.matrix(predict(dnn_model_3, valid_predict.hex))
score_dnn_3=mean(abs(exp(pred_dnn_3)-valid_loss))
cat("score_dnn_3:",score_dnn_3,"\n")


## DNN Neural net 4 // increase epochs for higher accuracy
start<-proc.time()
dnn_model_4<-h2o.deeplearning(x=1:101, y=102, 
                              training_frame=train_frame.hex, validation_frame=valid_frame.hex,
                              epochs=70, 
                              stopping_rounds=3,
                              overwrite_with_best_model=T,
                              activation="Rectifier",
                              distribution="huber",
                              hidden=c(101, 101))
print("DNN_4 runtime:")
print(proc.time()-start)
pred_dnn_4<-as.matrix(predict(dnn_model_4, valid_predict.hex))
score_dnn_4=mean(abs(exp(pred_dnn_4)-valid_loss))
cat("score_dnn_4:",score_dnn_4,"\n")



## DNN Neural net 5 // increase epochs for higher accuracy
start<-proc.time()
dnn_model_5<-h2o.deeplearning(x=1:101, y=102, 
                              training_frame=train_frame.hex, validation_frame=valid_frame.hex,
                              epochs=60, 
                              stopping_rounds=5,
                              overwrite_with_best_model=T,
                              activation="Rectifier",
                              distribution="huber",
                              hidden=c(91,91,91))
print("DNN_5 runtime:")
print(proc.time()-start)
pred_dnn_5<-as.matrix(predict(dnn_model_5, valid_predict.hex))
score_dnn_5=mean(abs(exp(pred_dnn_5)-valid_loss))
cat("score_dnn_5:",score_dnn_5,"\n")
# Average everything
pred_ensemble=(pred_dnn_1+pred_dnn_2+pred_dnn_3+pred_dnn_4+pred_dnn_5)/5
score_ensemble=mean(abs(exp(pred_ensemble)-valid_loss))

# predict results
pred_dnn_1<-(as.matrix(predict(dnn_model_1, test.hex)))
pred_dnn_2<-(as.matrix(predict(dnn_model_2, test.hex)))
pred_dnn_3<-(as.matrix(predict(dnn_model_3, test.hex)))
pred_dnn_4<-(as.matrix(predict(dnn_model_4, test.hex)))
pred_dnn_5<-(as.matrix(predict(dnn_model_5, test.hex)))


# Final test prediction
pred_all<-exp((pred_dnn_1+pred_dnn_2+pred_dnn_3+pred_dnn_4+pred_dnn_5)/5)
# Write submissions
SUBMISSION_FILE$loss = pred_all
write.csv(SUBMISSION_FILE, 'h2o_dnn_stacking_secondlevel_output-v2.csv', row.names=FALSE)










