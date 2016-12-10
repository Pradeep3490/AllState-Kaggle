################## Kaggle AllState Competition November 2016 ###########################

setwd("C:/Users/HP/Desktop/Kaggle/All State Severity Claims")
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

ID = 'id'
TARGET = 'loss'
SEED = 0
SHIFT = 200


TRAIN_FILE = "C:/Users/HP/Desktop/Kaggle/All State Severity Claims/train.csv"
TEST_FILE = "C:/Users/HP/Desktop/Kaggle/All State Severity Claims/test.csv"
SUBMISSION_FILE = "C:/Users/HP/Desktop/Kaggle/All State Severity Claims/sample_submission.csv"


train = fread(TRAIN_FILE, showProgress = TRUE)
test = fread(TEST_FILE, showProgress = TRUE)
#test$loss = 'NA'
submission = fread(SUBMISSION_FILE, showProgress = TRUE)



y_train = log(train[,TARGET, with = FALSE] + SHIFT)[[TARGET]]
#y_train = as.data.frame(y_train)
train[, c(ID, TARGET) := NULL]
test[, c(ID) := NULL]


ntrain = nrow(train)

train_test = rbind(train,test,fill=TRUE)



# remove skewness

for (f in colnames(train_test)[colnames(train_test) %like% "^cont"]) {
  
  tst <- e1071::skewness(train_test[, eval(as.name(f))])
  if (tst > .25) {
    if (is.na(train_test[, BoxCoxTrans(eval(as.name(f)))$lambda])) next
    lambda = BoxCoxTrans(eval(as.name(f)))$lambda
      train_test[,f] = BoxCox(train_test[,eval(as.name(f))],lambda)
      }
}

# scale

for (f in colnames(train_test)[colnames(train_test) %like% "^cont"]) {
  train_test[, f] = scale(eval(as.name(f)))
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



dtrain = xgb.DMatrix(as.matrix(x_train), label=y_train)
dtest = xgb.DMatrix(as.matrix(x_test))


xgb_params = list(
  colsample_bytree = 0.4,
  subsample = 0.8,
  eta = 0.03, # replace this with 0.01 for local run to achieve 1113.93
  objective = 'reg:linear',
  max_depth = 12,
  alpha = 1,
  gamma = 2,
  min_child_weight = 1,
  booster='gbtree'
)


xg_eval_mae <- function (yhat, dtrain) {
  y = getinfo(dtrain, "label")
  err= mae(exp(y),exp(yhat) )
  return (list(metric = "error", value = err))
}


best_nrounds = 545


# can use depending on computing power
#res = xgb.cv(xgb_params,
 #            dtrain,
  #           nrounds=500,
   #          nfold=3,
    #         early_stopping_rounds=15,
     #        print_every_n = 10,
      #       verbose= 1,
       #      feval=xg_eval_mae,
        #     maximize=FALSE)

#cv_mean = res$test_error_mean[best_nrounds]


gbdt = xgb.train(xgb_params, dtrain, nrounds=as.integer(best_nrounds/0.8))

prediction = exp(predict(gbdt,dtest)) - SHIFT

submission$loss = prediction

write.csv(submission,'xgb_dec10v2.csv',row.names = FALSE)



# DNN Models

library(ggplot2) # Data visualization
library(readr) 
library(data.table)


setwd("C:/Users/HP/Desktop/Kaggle/All State Severity Claims")

# Load H2O
library(h2o)
kd_h2o<-h2o.init(nthreads = -1, max_mem_size = "4g")

#Reading Data, old school read.csv. Using fread is faster. 
set.seed(12345)

train = read.csv("C:/Users/HP/Desktop/Kaggle/All State Severity Claims/train.csv")
test = read.csv("C:/Users/HP/Desktop/Kaggle/All State Severity Claims/test.csv")
SUBMISSION_FILE = read.csv("C:/Users/HP/Desktop/Kaggle/All State Severity Claims/sample_submission.csv",colClasses = c("integer", "numeric"))

#View(train)

train<-train[,-1]
test_label<-test[,1]
test<-test[,-1]

index<-sample(1:(dim(train)[1]), 0.2*dim(train)[1], replace=FALSE)
#nrow(train)
train_frame<-train[-index,]
valid_frame<-train[index,]


valid_predict<-valid_frame[,-ncol(valid_frame)]
valid_loss<-valid_frame[,ncol(valid_frame)]

# log transform
train_frame[,ncol(train_frame)]<-log(train_frame[,ncol(train_frame)])
valid_frame[,ncol(train_frame)]<-log(valid_frame[,ncol(valid_frame)])

# load H2o data frame // validate that H2O flow looses all continous data
train_frame.hex<-as.h2o(train_frame)
valid_frame.hex<-as.h2o(valid_frame)
valid_predict.hex<-as.h2o(valid_predict)
test.hex<-as.h2o(test)


## DNN Neural net 1 // increase epochs for higher accuracy
start<-proc.time()
dnn_model_1<-h2o.deeplearning(x=1:(ncol(train_frame.hex)-1), y=ncol(train_frame.hex), 
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
score_dnn_1=mean(abs(exp(pred_dnn_1)-valid_loss))
cat("score_dnn_1:",score_dnn_1,"\n")


## DNN Neural net 2 // increase epochs for higher accuracy

start<-proc.time()
dnn_model_2<-h2o.deeplearning(x=1:(ncol(train_frame.hex)-1), y=ncol(train_frame.hex), 
                              training_frame=train_frame.hex, validation_frame=valid_frame.hex,
                              epochs=80, 
                              stopping_rounds=3,
                              overwrite_with_best_model=T,
                              activation="Rectifier",
                              distribution="huber",
                              hidden=c(71,71,71))
print("DNN_2 runtime:")
print(proc.time()-start)
pred_dnn_2<-as.matrix(predict(dnn_model_2, valid_predict.hex))
score_dnn_2=mean(abs(exp(pred_dnn_2)-valid_loss))
cat("score_dnn_2:",score_dnn_2,"\n")


## DNN Neural net 3 // increase epochs for higher accuracy
start<-proc.time()
dnn_model_3<-h2o.deeplearning(x=1:(ncol(train_frame.hex)-1), y=ncol(train_frame.hex), 
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
dnn_model_4<-h2o.deeplearning(x=1:(ncol(train_frame.hex)-1), y=ncol(train_frame.hex), 
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
dnn_model_5<-h2o.deeplearning(x=1:(ncol(train_frame.hex)-1), y=ncol(train_frame.hex), 
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
write.csv(SUBMISSION_FILE, 'h2o_blend-DEC9.csv', row.names=FALSE)

 #Ensembling with DNN

dnn_output = read.csv("C:/Users/HP/Desktop/Kaggle/All State Severity Claims/Ensemble/h2o_blend-DEC9.csv")
XGB_output = read.csv("C:/Users/HP/Desktop/Kaggle/All State Severity Claims/Ensemble/xgb_dec10v2.csv")
SUBMISSION_FILE = "C:/Users/HP/Desktop/Kaggle/All State Severity Claims/sample_submission.csv"
submission = fread(SUBMISSION_FILE, showProgress = TRUE)
#View(submission)

submission$loss = (dnn_output$loss+XGB_output$loss)/2


#View(submission)

write.csv(submission,'xgb_dnn_ensemble_dec10.v3.csv',row.names = FALSE)

# PL score of 1117 with rank of 1265/3300 top 40%


