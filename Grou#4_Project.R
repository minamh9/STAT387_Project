
#################################### Libraries #################################
################################################################################

library(tidyverse)
library(MASS)        # LDA, QDA, OLS, Ridge Regression, Box-Cox, stepAIC, etc...
library(ROCR)        # Precision/Recall/Sens./Spec./ performance plot
library(class)       # KNN, SOM, LVQ
library(e1071)       # Naive Bayesian Classifier, SVM, GKNN, ICA, LCA
library(boot)        # LOOCV, Bootstrap
library(caret)       # Classification/Regression Training
library(dplyr)       # For piping and mutation methods
library(ggplot2)     # For plots
library(randomForest)# for RFE
library(naivebayes)  # for NB
library(ROSE)        # for SMOTE
library(fastDummies) # For Get Dummies
library(pROC)




################################### LDA ########################################
################################################################################

german <- read.csv("germancredit.csv", header=T)
german[, sapply(german, is.character)] <- lapply(german[, sapply(german, is.character)], as.factor)
german$Default = as.factor(german$Default)

train_index = sample(nrow(german), nrow(german) * 0.3) # 50% for training

set.seed(123)
train_data =german[]
test_data = german[-train_index, ]
test_Y = test_data$Default

results.matrix = matrix(0, nrow = 5, ncol = 4)

set.seed(123)
lda.fit = lda(Default~. , data = train_data)
lda.pred = predict(lda.fit, test_data)
names(lda.pred)
lda.class = lda.pred$class

table(lda.class, test_Y)

sum(lda.pred$posterior[,1] >= 0.5) 
sum(lda.pred$posterior[,1] < 0.5)  

lda.tn = sum((test_Y == unique(test_Y)[1])&(lda.class == unique(test_Y)[1]))
lda.tp = sum((test_Y == unique(test_Y)[2])&(lda.class == unique(test_Y)[2]))

lda.fp = sum((test_Y == unique(test_Y)[1])&(lda.class == unique(test_Y)[2]))
lda.fn = sum((test_Y == unique(test_Y)[2])&(lda.class == unique(test_Y)[1]))

lda.n = lda.tn + lda.fp
lda.p = lda.fn + lda.tp

# Specificity, Sensitivity, Overall Error/ Probability of Misclassification #
spec.lda  = 1 - (lda.fp/lda.n)
sen.lda   = lda.tp/lda.p
oer.lda1  = (lda.fn + lda.fp)/(lda.n + lda.p)

# ROCR #
lda.pred <- prediction(lda.pred$posterior[,2], test_data$Default) 
lda.perf <- performance(lda.pred,"tpr","fpr")
plot(lda.perf,colorize=TRUE, lwd = 2, main = "LDA ROC Curve")
abline(a = 0, b = 1) 

lda.auc = performance(lda.pred, measure = "auc")

colnames(results.matrix) = c("SPEC", "SENS", "OER", "AUC")
rownames(results.matrix) = c("LDA", "QDA", "NAIVE-B", "KNN", "LOG")
results.matrix[1,] = as.numeric( c(spec.lda, sen.lda, oer.lda1, lda.auc@y.values))
results.matrix


################################## QDA #########################################
################################################################################

set.seed(123)
qda.fit = qda(Default~. , data = train_data)

qda.pred  = predict(qda.fit,  test_data)
qda.class = qda.pred$class

table(qda.class, test_Y)

qda.tn = sum((test_Y == unique(test_Y)[1])&(qda.class == unique(test_Y)[1]))
qda.tp = sum((test_Y == unique(test_Y)[2])&(qda.class == unique(test_Y)[2]))

qda.fp = sum((test_Y == unique(test_Y)[1])&(qda.class == unique(test_Y)[2]))
qda.fn = sum((test_Y == unique(test_Y)[2])&(qda.class == unique(test_Y)[1]))

qda.n = qda.tn + qda.fp
qda.p = qda.fn + qda.tp

# Specificity, Sensitivity, Overall Error/ Probability of Misclassification #
spec.qda  = 1 - (qda.fp/qda.n)
sen.qda   = qda.tp/qda.p
oer.qda1  = (qda.fn + qda.fp)/(qda.n + qda.p)

# ROCR #
qda.pred <- prediction(qda.pred$posterior[,2], test_data$Default) 
qda.perf <- performance(qda.pred,"tpr","fpr")
plot(qda.perf,colorize=TRUE, lwd = 2, main = "QDA ROC Curve")
abline(a = 0, b = 1) 


qda.auc = performance(qda.pred, measure = "auc")
print(qda.auc@y.values)


results.matrix[2,] = as.numeric( c(spec.qda, sen.qda, oer.qda1, qda.auc@y.values))
results.matrix



################################## Naive Bayes #################################
################################################################################

nb.fit = naiveBayes(Default~. , data = train_data)

nb.class = predict(nb.fit , test_data)
table(nb.class, test_Y)

mean(nb.class == test_Y)

nb.tn = sum((test_Y == unique(test_Y)[1])&(nb.class == unique(test_Y)[1]))
nb.tp = sum((test_Y == unique(test_Y)[2])&(nb.class == unique(test_Y)[2]))

nb.fp = sum((test_Y == unique(test_Y)[1])&(nb.class == unique(test_Y)[2]))
nb.fn = sum((test_Y == unique(test_Y)[2])&(nb.class == unique(test_Y)[1]))

nb.n = nb.tn + nb.fp
nb.p = nb.fn + nb.tp

# Specificity, Sensitivity, Overall Error/ Probability of Misclassification #
spec.nb  = 1 - (nb.fp/nb.n)
sen.nb   = nb.tp/nb.p
oer.nb1  = (nb.fn + nb.fp)/(nb.n + nb.p)


# ROCR #
nb.pred = predict(nb.fit , test_data, type = "raw")
nb.pred <- prediction(nb.pred[,2], test_data$Default) 
nb.perf <- performance(nb.pred,"tpr","fpr")
plot(nb.perf,colorize=TRUE, lwd = 2, main = "NAIVE-B ROC Curve")
abline(a = 0, b = 1) 

nb.auc = performance(nb.pred, measure = "auc")

results.matrix[3,] = as.numeric( c(spec.nb, sen.nb, oer.nb1, nb.auc@y.values))


########################### Logistic Regression ################################
################################################################################


set.seed(123)
log.reg = glm(Default~. , data = train_data, family = binomial)

log.pred = predict(log.reg, test_data, type = "response")

log.pred <- ifelse(log.pred >= 0.5, 1, 0)

table(log.pred, test_Y)

log.tn = sum((test_Y == unique(test_Y)[1])&(log.pred == unique(test_Y)[1]))
log.tp = sum((test_Y == unique(test_Y)[2])&(log.pred == unique(test_Y)[2]))

log.fp = sum((test_Y == unique(test_Y)[1])&(log.pred == unique(test_Y)[2]))
log.fn = sum((test_Y == unique(test_Y)[2])&(log.pred == unique(test_Y)[1]))

log.n = log.tn + log.fp
log.p = log.fn + log.tp


# Specificity, Sensitivity, Overall Error/ Probability of Misclassification #
spec.log  = 1 - (log.fp/log.n)
sen.log   = log.tp/log.p
oer.log1  = (log.fn + log.fp)/(log.n + log.p)

# ROCR #
log.pred <- prediction(log.pred, test_data$Default) 
log.perf <- performance(log.pred,"tpr","fpr")
plot(log.perf,colorize=TRUE, lwd = 2, main = "LOG ROC Curve")
abline(a = 0, b = 1) 


log.auc = performance(log.pred, measure = "auc")

results.matrix[5,] = as.numeric( c(spec.log, sen.log, oer.log1, log.auc@y.values))
results.matrix


###################### KNN Using Sampling method ###############################
################################################################################

credit = read.csv("germancredit.csv", header=T)
credit[, sapply(credit, is.character)] <- lapply(credit[, sapply(credit, is.character)], as.factor)

credit$Default = as.factor(credit$Default)

set.seed(123)
test_index <- sample(1:nrow(credit), round(nrow(credit) * 0.3), replace = FALSE)


k_values <- seq(1, 100, by = 1)
test_error <- rep(0, length(k_values))

for (i in 1:length(k_values)) {
  knn.pred = knn(data.matrix(credit[,-1]), data.matrix(credit[test_index,-1]), 
                 credit[,1], k=i, prob = TRUE)
  test_error[i] <- mean(knn.pred!= credit[test_index,1])
}

plot(k_values, test_error, type = "b", xlab = "K", ylab = "Test Error Rate")
optimal_K <- k_values[which.min(test_error)]

knn.optimal = knn(data.matrix(credit[,-1]), data.matrix(credit[test_index,-1]), 
                  credit[,1], k=optimal_K, prob = TRUE )
table(knn.optimal , credit[test_index,1])


knn.tn = sum((credit[test_index,1] == unique(credit[test_index,1])[1])&
               (knn.optimal == unique(credit[test_index,1])[1]))
knn.tp = sum((credit[test_index,1]== unique(credit[test_index,1])[2])&
               (knn.optimal == unique(credit[test_index,1])[2]))

knn.fp = sum((credit[test_index,1] == unique(credit[test_index,1])[1])&
               (knn.optimal == unique(credit[test_index,1])[2]))
knn.fn = sum((credit[test_index,1]  == unique(credit[test_index,1])[2])&
               (knn.optimal == unique(credit[test_index,1])[1]))

knn.n = knn.tn + knn.fp
knn.p = knn.fn + knn.tp


# Specificity, Sensitivity, Overall Error/ Probability of Misclassification #
spec.knn  = 1 - (knn.fp/knn.n)
sen.knn   = knn.tp/knn.p
oer.knn1  = (knn.fn + knn.fp)/(knn.n + knn.p)
#oer.knn2  = 1 - mean(knn.pred == Direction.2005)


test_y <- credit[test_index,]


# ROCR #
knn.pred = predict(knn.optimal , test_data)

knn.p <- prediction(attributes(knn.optimal)$prob,test_y$Default) 
knn.perf <- performance(knn.p, "tpr", "fpr")
plot(knn.perf, colorize = TRUE, lwd = 2, main = "ROC Curve")
abline(a = 0, b = 1) 


knn.auc = performance(knn.p, measure = "auc")

results.matrix[4,] = as.numeric( c(spec.knn, sen.knn, oer.knn1, knn.auc@y.values))
results.matrix


########################## kNN using 10_fold CV ################################
################################################################################

# Set the number of folds
k <- 10
# Randomly assign each row in the data to a fold
set.seed(1234) # for reproducibility
fold_indices <- sample(rep(1:k, length.out = nrow(credit)))

# Initialize an empty list to store the folds
folds <- vector("list", k)

# Assign each row to a fold
for (i in 1:k) {
  folds[[i]] <- which(fold_indices == i)
}

#To store the error rate of each fold
error_rate <- numeric(k)
kappa <- numeric(k)
confusion_matrices <- vector("list", k)

# Loop through each fold
for (i in 1:k) {
  # Extract the i-th fold as the testing set
  test_indices <- unlist(folds[[i]])
  
  test <- credit[test_indices, ]
  train <- credit[]
}

set.seed(1234)
# Define the training control object for 10-fold cross-validation
train_control <- trainControl(method = "cv", number = 10)

# tuneLength argument to specify the range of values of K to be considered for tuning
set.seed(1234)
knn_model <- train(Default ~ ., 
                   data = train, 
                   method = "knn", 
                   trControl = train_control,
                   tuneGrid = data.frame(k = 1:10))

# Make predictions on the test data using the trained model and calculate the test error rate
knn.predictions <- predict(knn_model, newdata = test)

confusionMatrix(knn.predictions, test$Default)

# Convert predictions to a numeric vector
knn.predictions <- as.numeric(knn.predictions)

# Calculate the AUC using the performance() and auc() functions:
pred_obj <- prediction(knn.predictions, test$Default)
auc_val <- performance(pred_obj, "auc")@y.values[[1]]
auc_val

# Performance plot for TP and FP
roc_obj <- performance(pred_obj, "tpr", "fpr")
plot(roc_obj, colorize = TRUE, lwd = 2,
     xlab = "False Positive Rate", 
     ylab = "True Positive Rate",
     main = "KNN ROC Curves")
abline(a = 0, b = 1)
x_values <- as.numeric(unlist(roc_obj@x.values))
y_values <- as.numeric(unlist(roc_obj@y.values))



#===============================================================================
#======================== One-Hot Encoding (Dummies) ===========================
#===============================================================================

# Below I am making dummy variables (or one-hot encodings) of the categorical
# variables. This will be used for many circumstances where qualitative data
# isn't allowed. For instance, performing PCA, or SMOTE. In James Gareth's
# book "An Introduction To Statistical Learning" (ISLR), the authors discuss
# one-hot encoding in Chapter 6, specifically in the context of linear
# regression models. In Section 6.2.2, the authors explain that when one-hot
# encoding a categorical variable with k levels, one should create k-1 binary
# variables. The reason for this is to avoid perfect multicollinearity in the
# model matrix, which can cause problems when fitting regression type models.

GetDummies <- function(data, dropFirst=FALSE) {
  
  # Instantiate an empty data.frame in R because R is horrible! :)
  dummy.data <- data.frame(
    Default=as.Date(character()),
    stringsAsFactors=FALSE
  ) 
  
  # Optional argument
  if (dropFirst==TRUE) {
    # Select the qualitative columns in the dataset
    qual_cols <- sapply(data, is.character)
    
    # Create dummy variables for the qualitative columns
    data_dummy <- dummy_cols(data[, qual_cols], remove_first_dummy = TRUE)
    
    # Combine the dummy variables with the original dataset
    data_final <- cbind(data[, !qual_cols], data_dummy)
    
    # Use select_if to drop non-integer columns
    dummy.data <- select_if(data_final, is.integer)
  } else {
    # One-Hot Encoding Without Removing the First Column
    dummy <- dummyVars(" ~ .", data=data)
    dummy.data <- data.frame(predict(dummy, newdata = data)) 
  }
  
  return(dummy.data)
}

StandardizeData <- function(dummy.data) {
  
  # Check if all columns are numeric
  all_numeric <- sapply(dummy.data, is.numeric)
  if (any(!all_numeric)) {
    print(cat("The following columns are not numeric:", names(dummy.data)[!all_numeric], "\n"))
    return(NULL)
  }
  
  # Standardize the data
  dummy.data_std <- scale(dummy.data)
  
  # Remove the unwanted standardized target column
  dummy.data_std <- subset(dummy.data_std, select = -Default)
  
  # Add the new column to the beginning of the dataset
  dummy.data_std <- cbind(dummy.data$Default, dummy.data_std)
  
  # rename the target column (I hate R so much)
  colnames(dummy.data_std)[1] <- "Default"
  
  return(dummy.data_std)
}

#===============================================================================
#========================= Visualizations and plots ============================
#===============================================================================

Balance.Plot <- function(data) {
  class_counts <- data.frame(table(data$Default))
  class_counts <- as_tibble(class_counts)
  class_counts$class <- c("Default 0", "Default 1")
  ggplot(class_counts, aes(x = class, y = Freq, fill=as.factor(class))) +
    geom_bar(stat = "identity") +
    labs(x = "Class", y = "Count", title = "Class Counts") +
    scale_fill_manual(name = "Default",
                      labels = c("Good", "Bad"),
                      values=c("#00BFC4", "#F8766D")) +
    theme_minimal()
}

Count.Plot <- function(data, column.name) {  
  german.0.default <- data[data$Default == 0, ]
  german.1.default <- data[data$Default == 1, ]
  
  checkingstatus.counts.0 <- table(german.0.default[column.name])
  checkingstatus.counts.1 <- table(german.1.default[column.name])
  
  counts.df.0 <- as.data.frame(checkingstatus.counts.0)
  counts.df.1 <- as.data.frame(checkingstatus.counts.1)
  
  colnames(counts.df.0) <- c(column.name, "count")
  colnames(counts.df.1) <- c(column.name, "count")
  
  counts.df.0 <- counts.df.0 %>% mutate(class = 0)
  counts.df.1 <- counts.df.1 %>% mutate(class = 1)
  
  # Combine the two data frames
  combined_df <- rbind(counts.df.0, counts.df.1)
  
  # Create the plot
  ggplot(combined_df, aes(x=combined_df[,column.name], y=count, 
                          fill=as.factor(class))) +
    geom_bar(stat="identity", position="dodge") +
    labs(x = column.name, y="Count", fill="Class") +
    scale_fill_manual(labels = c("Good", "Bad"),
                      values=c("#00BFC4", "#F8766D")) +
    theme_minimal()
}

Density.Plot <- function(data, column.name) {
  
  # Create subsets of the dataframe based on the binary class
  df_default0 <- data[data[["Default"]] == 0,]
  df_default1 <- data[data[["Default"]] == 1,]
  
  # Plot the two density plots on the same plot
  ggplot() +
    geom_density(data = df_default0, aes(x = df_default0[,column.name], 
                                         fill = "Default 0"), alpha = 0.5) +
    geom_density(data = df_default1, aes(x = df_default1[,column.name], 
                                         fill = "Default 1"), alpha = 0.5) +
    labs(title = paste("Distribution of", column.name, "by Default"),
         x = column.name,
         y = "Density") +
    scale_fill_manual(labels = c("Good", "Bad"),
                      values = c("#00BFC4", "#F8766D"), name = "Default") +
    theme_minimal()
}

#===============================================================================
#====================================== RFE function ===========================
#===============================================================================


RFE <- function(data, num.features=4) {
  
  # Define the predictor and response variables
  train.X <- data[, !(names(data) %in% c("Default"))]
  train.Y <- as.factor(data[, "Default"])
  
  # Define the control parameters for feature selection
  ctrl <- rfeControl(functions = rfFuncs,
                     method = "cv",
                     number = 10)
  
  # Perform recursive feature elimination using the random forest algorithm
  rf_rfe <- rfe(train.X, train.Y, sizes = c(1:num.features), rfeControl = ctrl)
  
  # Print the results
  print(rf_rfe)
  
  # Plot the results
  p <- plot(rf_rfe, type = c("g", "o"))
  print(p)
  
  # Get the features
  features <- row.names(varImp(rf_rfe))[1:num.features]
  
  varimp_data <- data.frame(feature = features,
                            importance = varImp(rf_rfe)[1:num.features, 1])
  
  # Plots the variable importances
  gg <- ggplot(data = varimp_data, 
               aes(x = reorder(feature, -importance), y = importance, fill = feature)) +
    geom_bar(stat="identity") + 
    labs(x = "Features", y = "Variable Importance") + 
    geom_text(aes(label = round(importance, 2)), vjust=1.6, color="white", size=4) + 
    theme_bw() + 
    theme(legend.position = "none") + 
    theme(axis.text.x = element_text(angle = 90, vjust = 0.5))
  
  print(gg)
  
  target.features <- c("Default", features)
  
  return(subset(data, select=target.features))
}

#===============================================================================
#================================= PCA function=================================
#===============================================================================

# This function gets the first number of Principle Components.
# It requires a pca object and the number of PC's desired.


Get.First.n.PCs <- function(data, n=4) {
  # Extract the first four principal components
  first_n_pcs <- data$x[, 1:n]
  
  # Create a new PCA results object with only the first four principal components
  german.pca.first.n <- data
  german.pca.first.n$x <- first_n_pcs
  german.pca.first.n$rotation <- data$rotation[, 1:n, drop = FALSE]
  
  return(german.pca.first.n)
}



#===============================================================================
#================================== LDA function ===============================
#===============================================================================

LDA.Model <- function(data) {
  # Split the data into training and testing sets
  set.seed(123) # for reproducibility
  train_index <- sample(nrow(data), nrow(data) * 0.5) # 70% for training
  train_data <- data[]
  train_label <- as.factor(data[train_index,"Default"])
  test_data <- data[-train_index, ]
  test_label <- as.factor(data[-train_index,"Default"])
  
  # Run LDA and predictions
  germanlda_short <- lda(Default ~ . , data.frame(train_data))
  predictions_short <- predict(germanlda_short, newdata = data.frame(test_data))
  
  # Printing the confusion matrix
  cm <- confusionMatrix(predictions_short$class, test_label)
  table_cm <- as.table(cm)
  acc <- sum(diag(table_cm)) / sum(table_cm)
  oer = 1 - acc
  fpr <- table_cm[1,2]/(table_cm[1,2]+table_cm[1,1])
  
  
  # Printing the ROC curve
  roc.curve <- roc(test_label, predictions_short$posterior[,2])
  print(plot(roc.curve))
  print(roc.curve$auc)
  
  ldaPredictiction <- prediction(
    as.numeric(predictions_short$class),
    as.numeric(test_label)
  )
  
  ldaPerf <- performance(ldaPredictiction, measure = "tpr", x.measure = "fpr")
  ldaAUC <- performance(ldaPredictiction, measure = "auc")
  
  print(plot(ldaPerf))
  
  # Report the model performance metrics for the optimal K
  # Extract performance metrics
  specificity <- slot(ldaPerf, "y.values")[[1]]
  sensitivity <- 1 - slot(ldaPerf, "x.values")[[1]]
  auc <- slot(ldaAUC, "y.values")
  
  
  ldaError <- mean(as.numeric(predictions_short$class) !=as.numeric(test_label))
  
  # Print performance metrics
  print(cm)
  print(paste0("AUC: ", auc))
  print(paste0("Error rate:", ldaError))
  print(paste0("specificity: ",specificity))
  print(paste0("sensitivity: ",sensitivity))
  
  # Calculate false positives
  false_positives <- sum(as.numeric(predictions_short$class) == 2 & as.numeric(test_label) == 1)
  
  # Calculate false positives as a percentage
  total_negatives <- sum(as.numeric(test_label) == 1)
  false_positives_percent <- false_positives / total_negatives * 100
  
  
  # Print the false positives percentage
  print(paste0("False positives percentage: ", round(false_positives_percent,3), "%"))
  
  return(list(round(fpr,3),oer))
}


#===============================================================================
#================================= QDA function ================================
#===============================================================================

QDA.Model <- function(data) {
  # Split the data into training and testing sets
  set.seed(123) # for reproducibility
  train_index <- sample(nrow(data), nrow(data) * 0.5) # 70% for training
  train_data <- data[]
  train_label <- as.factor(data[train_index,"Default"])
  test_data <- data[-train_index, ]
  test_label <- as.factor(data[-train_index,"Default"])
  
  
  germanQda <- qda(Default ~ . , data.frame(train_data))
  predictions <- predict(germanQda, newdata =  data.frame(test_data))
  train_data$Default <- factor(train_data$Default, levels = levels(predictions$class))
  
  # Printing the confusion matrix
  cm <- confusionMatrix(predictions$class, test_label)
  table_cm <- as.table(cm)
  acc <- sum(diag(table_cm)) / sum(table_cm)
  fpr <- table_cm[1,2]/(table_cm[1,2]+table_cm[1,1])
  oer = 1 - acc
  
  
  # Printing the ROC curve
  roc.curve <- roc(test_label, predictions$posterior[,2])
  print(plot(roc.curve))
  print(roc.curve$auc)
  
  qdaPredictiction <- prediction(
    as.numeric(predictions$class),
    as.numeric(test_label)
  )
  
  qdaPerf <- performance(qdaPredictiction, measure = "tpr", x.measure = "fpr")
  qdaAUC <- performance(qdaPredictiction, measure = "auc")
  
  print(plot(qdaPerf))
  
  # Report the model performance metrics for the optimal K
  # Extract performance metrics
  specificity <- slot(qdaPerf, "y.values")[[1]]
  sensitivity <- 1 - slot(qdaPerf, "x.values")[[1]]
  auc <- slot(qdaAUC, "y.values")
  
  qdaError <- mean(as.numeric(predictions$class) !=as.numeric(test_label))
  
  # Print performance metrics
  print(cm)
  print(paste0("AUC: ", auc))
  print(paste0("Error rate:", qdaError))
  print(paste0("specificity: ",specificity))
  print(paste0("sensitivity: ",sensitivity))
  
  # Calculate false positives
  false_positives <- sum(as.numeric(predictions$class) == 2 & as.numeric(test_label) == 1)
  
  # Calculate false positives as a percentage
  total_negatives <- sum(as.numeric(test_label) == 1)
  false_positives_percent <- false_positives / total_negatives * 100
  
  # Print the false positives percentage
  print(paste0("False positives percentage: ", round(false_positives_percent,3), "%"))
  
  return(list(round(fpr,3),oer))
}



#===============================================================================
#=================================== GLM function ==============================
#===============================================================================
glm_func <- function(data) {
  
  # Split the data into training and testing sets on PCA
  set.seed(123) # for reproducibility
  train_index <- sample(nrow(data), nrow(data) * 0.5) # 70% for training
  train_data <- data[]
  test_data <- data[-train_index, ]
  test_label <- data[-train_index,"Default"]
  
  
  # Train the GLM model using the training data
  glm_model <- glm(Default ~ ., data = train_data, family = binomial(link = "logit"))
  
  # Evaluate the model performance on the test set
  glm_pred <- predict(glm_model, newdata = test_data, type = "response")
  glm_pred <- ifelse(glm_pred >= 0.5, 1, 0)
  
  glmConf <- confusionMatrix(factor(glm_pred, levels = c(0,1)),
                             factor(test_data$Default, levels = c(0,1)))
  table_cm <- as.table(glmConf)
  acc <- sum(diag(table_cm)) / sum(table_cm)
  fpr <- table_cm[1,2]/(table_cm[1,2]+table_cm[1,1])
  oer = 1 - acc
  
  # Get predictions
  pred_obj <- prediction(glm_pred,test_label)
  
  # Plot ROC Curve
  perf <- performance(pred_obj, measure = "tpr", x.measure = "fpr")
  plot(perf, main = "ROC Curve", col = "blue", lwd = 2, legacy.axes = TRUE)
  abline(a = 0, b = 1, lty = 2, col = "red")
  
  glmPredictiction <- prediction(as.numeric(glm_pred), as.numeric(test_label))
  glmPerf <- performance(glmPredictiction, measure = "tpr", x.measure = "fpr")
  glmAUC <- performance(glmPredictiction, measure = "auc")
  
  print(plot(glmPerf))
  
  # Extract performance metrics
  sensitivity <- slot(glmPerf, "y.values")[[1]]
  specificity <- 1 - slot(glmPerf, "x.values")[[1]]
  auc <- slot(glmAUC, "y.values")
  glmError <- mean(as.numeric(glm_pred) !=as.numeric(test_label))
  
  # Print performance metrics
  print(glmConf)
  print(paste0("Sensitivity: ", sensitivity))
  print(paste0("Specificity: ", specificity))
  print(paste0("AUC: ", auc))
  print(paste0("Error rate:", glmError))
  
  # Calculate false positives
  false_positives <- sum(as.numeric(glm_pred) == 1 & as.numeric(test_label) == 0)
  
  # Calculate false positives as a percentage
  total_negatives <- sum(as.numeric(test_label) == 0)
  false_positives_percent <- false_positives / total_negatives * 100
  
  # Print the false positives percentage
  print(paste0("False positives percentage: ", round(false_positives_percent,3), "%"))
  
  return(list(round(fpr,3), oer))
}


#===============================================================================
#=================================== KNN function===============================
#===============================================================================

KNN.Model <- function(data) {
  
  # Split the data into training and testing sets on PCA
  set.seed(123) # for reproducibility
  train_index <- sample(nrow(data), nrow(data) * 0.5) # 70% for training
  train_data <- data[]
  train_label <- as.factor(data[,"Default"])
  test_data <- data[-train_index, ]
  test_label <- as.factor(data[-train_index,"Default"])
  
  # Train the KNN classifier using the training data
  knn_model <- train(
    x = train_data,
    y = train_label,
    method = "knn",
    trControl = trainControl(method = "cv", number = 10),
    tuneGrid = data.frame(k = 1:30)
  )
  
  # Get predictions
  pred <- predict(knn_model, newdata = test_data, type = "prob")
  pred_obj <- prediction(pred[,2], test_label)
  
  # Evaluate the model performance on the test set for each value of K
  knnPred <- predict(knn_model, newdata=test_data)
  knnConf <- confusionMatrix(knnPred, test_label)
  table_cm <- as.table(knnConf)
  acc <- sum(diag(table_cm)) / sum(table_cm)
  fpr <- table_cm[1,2]/(table_cm[1,2]+table_cm[1,1])
  oer = 1 - acc
  
  # Plot ROC Curve
  perf <- performance(pred_obj, measure = "tpr", x.measure = "fpr")
  plot(perf, main = "ROC Curve", col = "blue", lwd = 2, legacy.axes = TRUE)
  abline(a = 0, b = 1, lty = 2, col = "red")
  
  # Choose the K that gives the lowest test error rate
  kOpt <- knn_model$bestTune$k
  
  # Plot the tuning parameter performance
  gg <- ggplot(knn_model$results, aes(x=k, y=Accuracy)) +
    geom_line() +
    geom_point(size = 3) +
    geom_vline(xintercept=kOpt, color="red", linetype="dashed") +
    labs(title="Tuning Parameter Performance",
         x="K",
         y="Accuracy") +
    theme_minimal()
  
  print(gg)
  
  knnPredictiction <- prediction(as.numeric(knnPred), as.numeric(test_label))
  knnPerf <- performance(knnPredictiction, measure = "tpr", x.measure = "fpr")
  knnAUC <- performance(knnPredictiction, measure = "auc")
  
  print(plot(knnPerf))
  
  # Report the model performance metrics for the optimal K
  # Extract performance metrics
  sensitivity <- slot(knnPerf, "y.values")[[1]]
  specificity <- 1 - slot(knnPerf, "x.values")[[1]]
  auc <- slot(knnAUC, "y.values")
  knnError <- mean(as.numeric(knnPred) !=as.numeric(test_label))
  
  # Print performance metrics
  print(knnConf)
  print(paste0("Sensitivity: ", sensitivity))
  print(paste0("Specificity: ", specificity))
  print(paste0("AUC: ", auc))
  print(paste0("Optimal K:", kOpt))
  print(paste0("Error rate:", knnError))
  
  # Calculate false positives
  false_positives <- sum(as.numeric(knnPred) == 2 & as.numeric(test_label) == 1)
  
  # Calculate false positives as a percentage
  total_negatives <- sum(as.numeric(test_label) == 1)
  false_positives_percent <- false_positives / total_negatives * 100
  
  # Print the false positives percentage
  print(paste0("False positives percentage: ", round(false_positives_percent,3), "%"))
  
  return(list(round(fpr,3), oer))
}


#===============================================================================
#============================== Naive Bayes function ===========================
#===============================================================================


NB.Model <- function(data) {
  
  # Split the data into training and testing sets on PCA
  set.seed(123) # for reproducibility
  train_index <- sample(nrow(data), nrow(data) * 0.5) # 70% for training
  train_data <- data[]
  train_label <- as.factor(data[,"Default"])
  test_data <- data[-train_index, ]
  test_label <- as.factor(data[-train_index,"Default"])
  
  # Train the Naïve Bayes classifier using the training data
  nb_model <- naive_bayes(train_data, train_label)
  
  # # Evaluate the model performance on the test set
  nb_pred <- predict(nb_model, newdata=test_data)
  nbConf <- confusionMatrix(nb_pred, test_label)
  
  # Get predictions
  pred <- predict(nb_model, newdata = test_data, type = "prob")
  pred_obj <- prediction(pred[,2], test_label)
  
  # Evaluate the model performance on the test set for each value of K
  nbPred <- predict(nb_model, newdata=test_data)
  nbConf <- confusionMatrix(nbPred, test_label)
  table_cm <- as.table(nbConf)
  acc <- sum(diag(table_cm)) / sum(table_cm)
  fpr <- table_cm[1,2]/(table_cm[1,2]+table_cm[1,1])
  oer = 1 - acc
  
  # Plot ROC Curve
  perf <- performance(pred_obj, measure = "tpr", x.measure = "fpr")
  plot(perf, main = "ROC Curve", col = "blue", lwd = 2, legacy.axes = TRUE)
  abline(a = 0, b = 1, lty = 2, col = "red")
  
  nbPredictiction <- prediction(as.numeric(nb_pred), as.numeric(test_label))
  nbPerf <- performance(nbPredictiction, measure = "tpr", x.measure = "fpr")
  nbAUC <- performance(nbPredictiction, measure = "auc")
  
  print(plot(nbPerf))
  
  # Extract performance metrics
  sensitivity <- slot(nbPerf, "y.values")[[1]]
  specificity <- 1 - slot(nbPerf, "x.values")[[1]]
  auc <- slot(nbAUC, "y.values")
  nbError <- mean(as.numeric(nb_pred) !=as.numeric(test_label))
  
  # Print performance metrics
  print(nbConf)
  print(paste0("Sensitivity: ", sensitivity))
  print(paste0("Specificity: ", specificity))
  print(paste0("AUC: ", auc))
  print(paste0("Error rate:", nbError))
  
  # Calculate false positives
  false_positives <- sum(as.numeric(nb_pred) == 2 & as.numeric(test_label) == 1)
  
  # Calculate false positives as a percentage
  total_negatives <- sum(as.numeric(test_label) == 1)
  false_positives_percent <- false_positives / total_negatives * 100
  
  # Print the false positives percentage
  print(paste0("False positives percentage: ", round(false_positives_percent,3), "%"))
  
  return(list(round(fpr,3),oer))
}

#===============================================================================
#===================== Using Above function for modeling  ======================
#===============================================================================


dummy <- dummyVars(" ~ .", data=german, fullRank = TRUE)
newgerman <- data.frame(predict(dummy, newdata = german))


RFE.4.German <- RFE(newgerman)

pca.four <- Get.First.n.PCs(newgerman)

RFE.10.German <- RFE(newgerman, num.features=10)

pca.ten <- Get.First.n.PCs(newgerman,10)

balanced.dummy.german <- ROSE(Default ~ ., data = newgerman, seed = 123)$data

RFE.4.German.balance <- RFE(balanced.dummy.german)

pca.four02 <- Get.First.n.PCs(balanced.dummy.german,4)

RFE.10.German.balance <- RFE(balanced.dummy.german, num.features=10)

pca.ten02 <- Get.First.n.PCs(balanced.dummy.german,10)

simpleLDA <- LDA.Model(newgerman)

result <- LDA.Model(RFE.4.German)

pca.result <- LDA.Model(pca.four)

result2 <- LDA.Model(RFE.10.German)

pca.result2 <- LDA.Model(pca.ten)

result3 <- LDA.Model(RFE.4.German.balance)

pca.result3 <- LDA.Model(pca.four02)

result4 <- LDA.Model(RFE.10.German.balance)

pca.result4 <- LDA.Model(pca.ten02)


simpleQDA <- QDA.Model(newgerman)

qda.result <- QDA.Model(RFE.4.German)

qda.pca.result <- QDA.Model(pca.four)

qda.result2 <- QDA.Model(RFE.10.German)

qda.pca.result2 <- QDA.Model(pca.ten)


qda.result3 <- QDA.Model(RFE.4.German.balance)

qda.pca.result3 <- QDA.Model(pca.four02)

qda.result4 <- QDA.Model(RFE.10.German.balance)

qda.pca.result4 <- QDA.Model(pca.ten02)


simpleGLM <- glm_func(newgerman)

glm.result <- glm_func(RFE.4.German)

glm.pca.result <- glm_func(pca.four)

glm.result2 <- glm_func(RFE.10.German)

glm.pca.result2 <- glm_func(pca.ten)

glm.result3 <- glm_func(RFE.4.German.balance)

glm.pca.result3 <- glm_func(pca.four02)

glm.result4 <- glm_func(RFE.10.German.balance)

glm.pca.result4 <- glm_func(pca.ten02)

simpleKNN <- KNN.Model(newgerman)

knn.result <- KNN.Model(RFE.4.German)

knn.pca.result <- KNN.Model(pca.four)

knn.result2 <- KNN.Model(RFE.10.German)

knn.pca.result2 <- KNN.Model(pca.ten)

knn.result3 <- KNN.Model(RFE.4.German.balance)

knn.pca.result3 <- KNN.Model(pca.four02)

knn.result4 <- KNN.Model(RFE.10.German.balance)

knn.pca.result4 <- KNN.Model(pca.ten02)


simpleNB <- NB.Model(newgerman)

nb.result <- NB.Model(RFE.4.German)

nb.pca.result <- NB.Model(pca.four)

nb.result2 <- NB.Model(RFE.10.German)

nb.pca.result2 <- NB.Model(pca.ten)

nb.result3 <- NB.Model(RFE.4.German.balance)

nb.pca.result3 <- NB.Model(pca.four02)

nb.result4 <- NB.Model(RFE.10.German.balance)

nb.pca.result4 <- NB.Model(pca.ten02)


comb_result1 <- cbind(RFE_4_feature = result[2],RFE_10_feature= result2[2],
                      PC_5 = pca.result[2], PC_10 = pca.result2[2], Base = simpleLDA[2])

qda_comb_result1 <- cbind(RFE_4_feature = qda.result[2],RFE_10_feature= qda.result2[2],
                          PC_5 = qda.pca.result[2], PC_10 = qda.pca.result2[2],Base = simpleQDA[2])

glm_comb_result1 <- cbind(RFE_4_feature = glm.result[2],RFE_10_feature= glm.result2[2],
                          PC_5 = glm.pca.result[2], PC_10 = glm.pca.result2[2], Base = simpleGLM[2])

knn_comb_result1 <- cbind(RFE_4_feature = knn.result[2],RFE_10_feature= knn.result2[2],
                          PC_5 = knn.pca.result[2], PC_10 = knn.pca.result2[2], Base = simpleKNN[2])

nb_comb_result1 <- cbind(RFE_4_feature = nb.result[2],RFE_10_feature= nb.result2[2],
                         PC_5 = nb.pca.result[2], PC_10 = nb.pca.result2[2], Base = simpleNB[2])

comb_result2 <- cbind(RFE_4_feature =result3[2],RFE_10_feature=result4[2],
                      PC_5 = pca.result3[2], PC_10 = pca.result4[2] ,Base = simpleLDA[2])

qda_comb_result2 <- cbind(RFE_4_feature =qda.result3[2],RFE_10_feature=qda.result4[2],
                          PC_5 = qda.pca.result3[2], PC_10 = qda.pca.result4[2],Base = simpleQDA[2])

glm_comb_result2 <- cbind(RFE_4_feature =glm.result3[2],RFE_10_feature=glm.result4[2],
                          PC_5 = glm.pca.result3[2], PC_10 = glm.pca.result4[2], Base = simpleGLM[2])

knn_comb_result2 <- cbind(RFE_4_feature =knn.result3[2],RFE_10_feature=knn.result4[2],
                          PC_5 = knn.pca.result3[2], PC_10 = knn.pca.result4[2], Base = simpleKNN[2])

nb_comb_result2 <- cbind(RFE_4_feature =nb.result3[2],RFE_10_feature=nb.result4[2],
                         PC_5 = nb.pca.result3[2], PC_10 = nb.pca.result4[2], Base = simpleNB[2])

comb_result <- rbind(comb_result1,qda_comb_result1,glm_comb_result1,
                     knn_comb_result1, nb_comb_result1, comb_result2, 
                     qda_comb_result2, glm_comb_result2, knn_comb_result2,
                     nb_comb_result2)

#table to compare all models with different dimensional reduction methid and different models

rownames(comb_result) <- c("LDA(std)","LDA(balance)","QDA(std)","QDA(balance)",
                           "GLM(std)","GLM(balance)", "KNN(std)","KNN(balance)",
                           "NB(std)","NB(balance)")

knitr::kable(comb_result, "simple")


#===============================================================================
#================================== Data Analysis ==============================
#===============================================================================

german <- read.csv("germancredit.csv", header=T)


# Examine Dimensions

# Counting Just the rows that did not Default (Good!)
default.0 <- dim(german[german$Default == 0, ])[1]
default.1 <- dim(german[german$Default == 1, ])[1]
paste("The number of those that did not default are", default.0, 
      "and those that did default are", default.1)

# Visualizing Dataset Class Imbalance
Balance.Plot(german)

# SMOTE Requires dummy variables
dummy.german <- GetDummies(german)

# Deal with multicolinearity for one-hot-encoded dataset
dummy.df = GetDummies(german, dropFirst = T)

# Apply SMOTE to balance the dataset
balanced.dummy.german <- ROSE(Default ~ ., data = dummy.german, seed = 123)$data


Balance.Plot(balanced.dummy.german)

corrplot(cor(dummy.german), method="square")
corrplot(cor(dummy.german), method="number")

# Value Counts For Each Class Type
Count.Plot(german, "checkingstatus1")

Count.Plot(german, "history")

Count.Plot(german, "purpose")

Count.Plot(german, "savings")

Count.Plot(german, "employ")


# Plotting Numerical Variable Distributions

Density.Plot(german, "duration")

Density.Plot(german, "amount")

Density.Plot(german, "installment")

Density.Plot(german, "residence")

Density.Plot(german, "age")

Density.Plot(german, "cards")

Density.Plot(german, "liable")

Density.Plot(balanced.dummy.german, "liable")
