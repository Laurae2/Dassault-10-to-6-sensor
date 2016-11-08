# LOAD PACKAGES
library(data.table)
library(Metrics)
library(xgboost)
library(R.utils)

# LOAD DATA
remove(list = ls())
setwd("D:/Data Science/Challengedata/Dassault")
train <- as.data.frame(fread("input_training.csv", header = TRUE))
test <- as.data.frame(fread("input_testing.csv", header = TRUE))
sample_submission <- as.data.frame(fread("challenge_output_data_training_file_machine_learning_for_sensors_reduction_in_body_posture_tracking.csv", header = TRUE))

# CORRECT ALL MISTAKES IN DATA
colnames(train) <- c("Time", "Torso_X", "Torso_Y", "Torso_Z", "Torso_W", "Head_X", "Head_Y", "Head_Z", "Head_W", "LeftUpperLeg_X", "LeftUpperLeg_Y", "LeftUpperLeg_Z", "LeftUpperLeg_W", "RightUpperArm_X", "RightUpperArm_Y", "RightUpperArm_Z", "RightUpperArm_W", "RightLowerLeg_X", "RightLowerLeg_Y", "RightLowerLeg_Z", "RightLowerLeg_W", "LeftLowerArm_X", "LeftLowerArm_Y", "LeftLowerArm_Z", "LeftLowerArm_W")
colnames(test) <- c("Time", "Torso_X", "Torso_Y", "Torso_Z", "Torso_W", "Head_X", "Head_Y", "Head_Z", "Head_W", "LeftUpperLeg_X", "LeftUpperLeg_Y", "LeftUpperLeg_Z", "LeftUpperLeg_W", "RightUpperArm_X", "RightUpperArm_Y", "RightUpperArm_Z", "RightUpperArm_W", "RightLowerLeg_X", "RightLowerLeg_Y", "RightLowerLeg_Z", "RightLowerLeg_W", "LeftLowerArm_X", "LeftLowerArm_Y", "LeftLowerArm_Z", "LeftLowerArm_W")
colnames(sample_submission) <- c("Time", "RightUpperLeg_X", "RightUpperLeg_Y", "RightUpperLeg_Z", "RightUpperLeg_W", "LeftUpperArm_X", "LeftUpperArm_Y", "LeftUpperArm_Z", "LeftUpperArm_W", "LeftLowerLeg_X", "LeftLowerLeg_Y", "LeftLowerLeg_Z", "LeftLowerLeg_W", "RightLowerArm_X", "RightLowerArm_Y", "RightLowerArm_Z", "RightLowerArm_W")

write.csv(cbind(train, sample_submission[, -1]), file = "proc_train.csv", row.names = FALSE)
write.csv(test, file = "proc_test.csv", row.names = FALSE)

# Create Custom Objective Function
pearson_objective <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  err <- cor(preds, labels)
  err <- ifelse(is.na(err), 0, err*err)
  return(list(metric = "Rsquared", value = err))
}

# Prepare Validation to Test
submission <- data.frame(matrix(ncol = 17, nrow = 83199))
colnames(submission) <- c("Time", "RightUpperLeg_X", "RightUpperLeg_Y", "RightUpperLeg_Z", "RightUpperLeg_W", "LeftUpperArm_X", "LeftUpperArm_Y", "LeftUpperArm_Z", "LeftUpperArm_W", "LeftLowerLeg_X", "LeftLowerLeg_Y", "LeftLowerLeg_Z", "LeftLowerLeg_W", "RightLowerArm_X", "RightLowerArm_Y", "RightLowerArm_Z", "RightLowerArm_W")
submission[, "Time"] <- test$Time

predicting <- c("RightUpperLeg_X", "RightUpperLeg_Y", "RightUpperLeg_Z", "RightUpperLeg_W", "LeftUpperArm_X", "LeftUpperArm_Y", "LeftUpperArm_Z", "LeftUpperArm_W", "LeftLowerLeg_X", "LeftLowerLeg_Y", "LeftLowerLeg_Z", "LeftLowerLeg_W", "RightLowerArm_X", "RightLowerArm_Y", "RightLowerArm_Z", "RightLowerArm_W")
predicting <- predicting[1]

# Convert data to an appropriate form for Extreme Gradient Boosting
train_xgb <- xgb.DMatrix(data = data.matrix(train[, -1]), label = sample_submission[, predicting])
test_xgb <- xgb.DMatrix(data = data.matrix(test[, -1]))

# Create self-made folds for model cross-validation
folds <- list()
tempInt <- which(sample_submission[, "Time"] == 0)
folds$Fold01 <- 1:3753
folds$Fold02 <- 3754:5633
folds$Fold03 <- 5634:7380
for (i in 2:40) {
  folds$Fold01 <- c(folds$Fold01, tempInt[i*3-2]:(tempInt[i*3-1]-1))
  folds$Fold02 <- c(folds$Fold02, tempInt[i*3-1]:(tempInt[i*3]-1))
  folds$Fold03 <- c(folds$Fold03, tempInt[i*3]:(tempInt[i*3+1]-1))
}
folds$Fold01 <- c(folds$Fold01, 321115:322931)
folds$Fold02 <- c(folds$Fold02, 322932:324845)
folds$Fold03 <- c(folds$Fold03, 324846:328575)

# Train a cross-validated model to assess performance
gc(verbose = FALSE)
set.seed(11111)
modelization <- xgb.cv(data = train_xgb,
                       folds = folds,
                       nthread = 2,
                       max.depth = 7,
                       eta = 0.2,
                       nrounds = 1000000,
                       subsample = 0.95,
                       colsample_bytree = 1.00,
                       verbose = TRUE,
                       eval_metric = pearson_objective,
                       objective = "reg:linear",
                       early.stop.round = 30,
                       maximize = TRUE,
                       prediction = TRUE)

# Train a single model on all data
gc(verbose = FALSE)
set.seed(11111)
modelization <- xgb.train(data = train_xgb,
                          nthread = 2,
                          max.depth = 10,
                          eta = 1,
                          nrounds = 350,
                          subsample = 1.00,
                          colsample_bytree = 1.00,
                          verbose = TRUE,
                          objective = "reg:linear",
                          eval_metric = pearson_objective,
                          early.stop.round = 10,
                          maximize = TRUE,
                          watchlist = list(train = train_xgb))

# Create validation submission for testing, but shrinked
submission[, predicting] <- predict(modelization, test_xgb, ntreelimit = 304)
summary(submission[, predicting])
submission[submission[, predicting] > 1, predicting] <- 1
submission[submission[, predicting] < -1, predicting] <- -1
summary(submission[, predicting])
summary(sample_submission[, predicting])
write.csv(submission, file = "submission.csv", row.names = FALSE)





# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Using an Auto-Learner per Fold if you need to use all folds separately

submission <- data.frame(matrix(ncol = 17, nrow = 83199))
colnames(submission) <- c("Time", "RightUpperLeg_X", "RightUpperLeg_Y", "RightUpperLeg_Z", "RightUpperLeg_W", "LeftUpperArm_X", "LeftUpperArm_Y", "LeftUpperArm_Z", "LeftUpperArm_W", "LeftLowerLeg_X", "LeftLowerLeg_Y", "LeftLowerLeg_Z", "LeftLowerLeg_W", "RightLowerArm_X", "RightLowerArm_Y", "RightLowerArm_Z", "RightLowerArm_W")
submission[, "Time"] <- test$Time

validation <- data.frame(matrix(ncol = 17, nrow = 328575))
colnames(validation) <- c("Time", "RightUpperLeg_X", "RightUpperLeg_Y", "RightUpperLeg_Z", "RightUpperLeg_W", "LeftUpperArm_X", "LeftUpperArm_Y", "LeftUpperArm_Z", "LeftUpperArm_W", "LeftLowerLeg_X", "LeftLowerLeg_Y", "LeftLowerLeg_Z", "LeftLowerLeg_W", "RightLowerArm_X", "RightLowerArm_Y", "RightLowerArm_Z", "RightLowerArm_W")
validation[, "Time"] <- train$Time


submission <- as.data.frame(fread("submission - Copy.csv", header = TRUE))
validation <- as.data.frame(fread("validation - Copy.csv", header = TRUE))
train <- cbind(train, validation[, -1])
test <- cbind(test, submission[, -1])

submission <- as.data.frame(fread("submission.csv", header = TRUE))
validation <- as.data.frame(fread("validation.csv", header = TRUE))

StartingFrom <- 16 - sum(is.na(submission[1, -1])) + 1

folds <- list()
tempInt <- which(sample_submission[, "Time"] == 0)
folds$Fold01 <- 1:3753
folds$Fold02 <- 3754:5633
folds$Fold03 <- 5634:7380
for (i in 2:40) {
  folds$Fold01 <- c(folds$Fold01, tempInt[i*3-2]:(tempInt[i*3-1]-1))
  folds$Fold02 <- c(folds$Fold02, tempInt[i*3-1]:(tempInt[i*3]-1))
  folds$Fold03 <- c(folds$Fold03, tempInt[i*3]:(tempInt[i*3+1]-1))
}
folds$Fold01 <- c(folds$Fold01, 321115:322931)
folds$Fold02 <- c(folds$Fold02, 322932:324845)
folds$Fold03 <- c(folds$Fold03, 324846:328575)

Booster <- "gblinear"
Counter <- 0
StartTime <- System$currentTimeMillis()
MaxCounter <- 3*(16 - StartingFrom + 1)
Paster <- paste("%0", nchar(MaxCounter, "width"), "d", sep = "")

for (i in c("RightUpperLeg_X", "RightUpperLeg_Y", "RightUpperLeg_Z", "RightUpperLeg_W", "LeftUpperArm_X", "LeftUpperArm_Y", "LeftUpperArm_Z", "LeftUpperArm_W", "LeftLowerLeg_X", "LeftLowerLeg_Y", "LeftLowerLeg_Z", "LeftLowerLeg_W", "RightLowerArm_X", "RightLowerArm_Y", "RightLowerArm_Z", "RightLowerArm_W")[StartingFrom:16]) {
  
  submission_temp <- data.frame(matrix(nrow = 83199, ncol = 3))
  submission_temp2 <- rep(0, 328575)
  
  for (j in c("Fold01", "Fold02", "Fold03")) {
    
    tempInt <- which(j == c("Fold01", "Fold02", "Fold03"))
    Counter <- Counter + 1
    CurrentTime <- System$currentTimeMillis()
    SpentTime <- (CurrentTime - StartTime) / 1000
    cat("\r[Task ", sprintf(Paster, Counter) , "/", MaxCounter, " | CPU = ", round(SpentTime, digits = 2), "s | ETA = ", round((MaxCounter - Counter) * SpentTime / Counter, 2), "s]: Predicting ", i, ", fold ", tempInt, "/3: ", sep = "")
    
    train_xgb <- xgb.DMatrix(data = data.matrix(train[-folds[[j]], -1]), label = sample_submission[-folds[[j]], i])
    valid_xgb <- xgb.DMatrix(data = data.matrix(train[folds[[j]], -1]), label = sample_submission[folds[[j]], i])
    test_xgb <- xgb.DMatrix(data = data.matrix(test[, -1]))
    
    gc(verbose = FALSE)
    set.seed(11111)
    sink(file = "junktext.txt", append = TRUE, split = FALSE)
    if (Booster == "gbtree") {
      
      modelization <- xgb.train(data = train_xgb,
                                nthread = 2,
                                max.depth = 7,
                                eta = 0.1,
                                nrounds = 1000000,
                                subsample = 0.95,
                                colsample_bytree = 1.00,
                                verbose = FALSE,
                                objective = "reg:linear",
                                eval_metric = pearson_objective,
                                early.stop.round = 50,
                                maximize = TRUE,
                                watchlist = list(test = valid_xgb, train = train_xgb))
      
    } else {
      
      modelization <- xgb.train(data = train_xgb,
                                nthread = 2,
                                max.depth = 7,
                                eta = 0.3,
                                nrounds = 1000000,
                                subsample = 0.95,
                                colsample_bytree = 1.00,
                                verbose = FALSE,
                                objective = "reg:linear",
                                booster = "gblinear",
                                eval_metric = pearson_objective,
                                early.stop.round = 30,
                                maximize = TRUE,
                                watchlist = list(test = valid_xgb, train = train_xgb))
      
      gc(verbose = FALSE)
      
      modelization <- xgb.train(data = train_xgb,
                                nthread = 2,
                                max.depth = 7,
                                eta = 0.3,
                                nrounds = modelization$bestInd,
                                subsample = 0.95,
                                colsample_bytree = 1.00,
                                objective = "reg:linear",
                                booster = "gblinear",
                                eval_metric = pearson_objective,
                                early.stop.round = modelization$bestInd,
                                maximize = TRUE,
                                watchlist = list(test = valid_xgb))
      
    }
    
    sink()
    
    gc(verbose = FALSE)
    
    CurrentTime <- System$currentTimeMillis()
    SpentTime <- (CurrentTime - StartTime) / 1000
    cat("\r[Task ", sprintf(Paster, Counter) , "/", MaxCounter, " | CPU = ", round(SpentTime, digits = 2), "s | ETA = ", round((MaxCounter - Counter) * SpentTime / Counter, 2), "s]: Predicting ", i, ", fold ", tempInt, "/3: ", modelization$bestInd, " rounds (", sprintf("%06f", modelization$bestScore), ").\n", sep = "")
    
    if (Booster == "gbtree") {
      submission_temp[, tempInt] <- predict(modelization, test_xgb, ntreelimit = modelization$bestInd)
    } else {
      submission_temp[, tempInt] <- predict(modelization, test_xgb)
    }
    gc(verbose = FALSE)
    submission_temp[submission_temp[, tempInt] > 1, tempInt] <- 1
    submission_temp[submission_temp[, tempInt] < -1, tempInt] <- -1
    
    if (Booster == "gbtree") {
      submission_temp2[folds[[j]]] <- predict(modelization, valid_xgb, ntreelimit = modelization$bestInd)
    } else {
      submission_temp2[folds[[j]]] <- predict(modelization, valid_xgb)
    }
    gc(verbose = FALSE)
    submission_temp2[submission_temp2 > 1] <- 1
    submission_temp2[submission_temp2 < -1] <- -1
    
  }
  
  submission[, i] <- rowMeans(submission_temp)
  validation[, i] <- submission_temp2
  cat("-- Validation of ", i, ": ", sprintf("%06f", cor(validation[, i], sample_submission[, i])^2), ", Mean: ", sprintf("%06f", mean(validation[, i])), " [", sprintf("%06f", min(validation[, i])), ", ", sprintf("%06f", max(validation[, i])), "].\n", sep = "")
  cat("-- Prediction of ", i, ": Mean: ", sprintf("%06f", mean(submission[, i])), " [", sprintf("%06f", min(submission[, i])), ", ", sprintf("%06f", max(submission[, i])), "].\n", sep = "")
  
  write.table(submission, file = "submission.csv", row.names = FALSE, sep = ";")
  write.csv(validation, file = "validation.csv", row.names = FALSE)
  cat("---- Successfully saved files!\n\n", sep = "")
  
}
