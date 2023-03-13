

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
    scale_fill_manual(values=c("#F8766D", "#00BFC4")) +
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
  ggplot(combined_df, aes(x=combined_df[,column.name], y=count, fill=as.factor(class))) +
    geom_bar(stat="identity", position="dodge") +
    labs(x = column.name, y="Count", fill="Class") +
    scale_fill_manual(values=c("#F8766D", "#00BFC4")) +
    theme_minimal()
}

Density.Plot <- function(data, column.name) {
  
  # Create subsets of the dataframe based on the binary class
  df_default0 <- data[data[["Default"]] == 0,]
  df_default1 <- data[data[["Default"]] == 1,]
  
  # Plot the two density plots on the same plot
  ggplot() +
    geom_density(data = df_default0, aes(x = df_default0[,column.name], fill = "Default 0"), alpha = 0.5) +
    geom_density(data = df_default1, aes(x = df_default1[,column.name], fill = "Default 1"), alpha = 0.5) +
    labs(title = paste("Distribution of", column.name, "by Default"),
         x = column.name,
         y = "Density") +
    scale_fill_manual(values = c("#F8766D", "#00BFC4"), name = "Default") +
    theme_minimal()
}

#===============================================================================
#====================================== RFE ====================================
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
    geom_bar(stat="identity") + labs(x = "Features", y = "Variable Importance") + 
    geom_text(aes(label = round(importance, 2)), vjust=1.6, color="white", size=4) + 
    theme_bw() + theme(legend.position = "none") + theme(axis.text.x = element_text(angle = 90, vjust = 0.5))
  
  print(gg)
  
  target.features <- c("Default", features)
  
  return(subset(data, select=target.features))
}

#===============================================================================
#====================================== PCA ====================================
#===============================================================================

# This function gets the first number of Principle Components.
# It requires a pca object and the number of PC's desired.

Get.First.n.PCs <- function(pca.data, n=4) {
  # Extract the first four principal components
  first_n_pcs <- pca.data$x[, 1:n]
  
  # Create a new PCA results object with only the first four principal components
  german.pca.first.n <- pca.data
  german.pca.first.n$x <- first_n_pcs
  german.pca.first.n$rotation <- pca.data$rotation[, 1:n, drop = FALSE]
  
  return(german.pca.first.n)
}

#===============================================================================
#================================== Naive Bayes ================================
#===============================================================================

NB.Model <- function(data) {
  
  # Split the data into training and testing sets on PCA
  set.seed(123) # for reproducibility
  train_index <- sample(nrow(data), nrow(data) * 0.7) # 70% for training
  train_data <- data[train_index, ]
  train_label <- as.factor(data[train_index,"Default"])
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
  knnPred <- predict(knn_model, newdata=test_data)
  knnConf <- confusionMatrix(knnPred, test_label)
  
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
  
  return(false_positives_percent)
}

#===============================================================================
#======================================= KNN ===================================
#===============================================================================

# This function takes in non PCA like data.
KNN.Model <- function(data) {
  
  # One-Hot Encoding Without Removing the First Column
  dummy <- dummyVars(" ~ .", data=data)
  dummy.data <- data.frame(predict(dummy, newdata = data)) 
  
  # Standardize the data
  dummy.data_std <- scale(dummy.data)
  
  # Split the data into training and testing sets
  set.seed(123) # for reproducibility
  train_index <- sample(nrow(dummy.data_std), nrow(dummy.data_std) * 0.7) # 70% for training
  train_data <- dummy.data_std[train_index, ]
  train_label <- as.factor(dummy.data_std[train_index,"Default"])
  test_data <- dummy.data_std[-train_index, ]
  test_label <- as.factor(dummy.data_std[-train_index,"Default"])
  
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
  
  return(false_positives_percent)
}  

#===============================================================================
#======================================= LDA ===================================
#===============================================================================

LDA.Model <- function(data) {
  
  # Select the qualitative columns in the dataset
  qual_cols <- sapply(data, is.character)
  
  # Create dummy variables for the qualitative columns
  data_dummy <- dummy_cols(data[, qual_cols], remove_first_dummy = TRUE)
  
  # Combine the dummy variables with the original dataset
  data_final <- cbind(data[, !qual_cols], data_dummy)
  
  # Use select_if to drop non-integer columns
  dummy.data <- select_if(data_final, is.integer)
  
  # Standardize the data
  dummy.data_std <- scale(dummy.data)
  
  # Remove the unwanted standardized target column
  dummy.data_std <- subset(dummy.data_std, select = -Default)
  
  # Add the new column to the beginning of the dataset
  dummy.data_std <- cbind(data$Default, dummy.data_std)
  
  # rename the target column (I hate R so much)
  colnames(dummy.data_std)[1] <- "Default"
  
  # Performing AIC Backward
  german_glm <- glm(Default~., family=binomial, data=data.frame(dummy.data_std))
  german_glm_back <- step(german_glm, trace=FALSE)
  
  # Collect the names of only the significant predictors
  sig.p.values <- summary(german_glm_back)$coefficients[,4] <= 0.05
  sig.predictor.names <- c("Default", row.names(data.frame(summary(german_glm_back)$coefficients[sig.p.values,])))
  
  # Check if the string "(Intercept)"exists in the vector
  if ("(Intercept)" %in% sig.predictor.names) {
    # Remove the string "(Intercept)" from the vector
    sig.predictor.names <- sig.predictor.names[-which(sig.predictor.names == "(Intercept)")]
  }
  short_germandata <- data.frame(data.frame(dummy.data_std)[,sig.predictor.names])
  
  # Split the data into training and testing sets
  set.seed(123) # for reproducibility
  train_index <- sample(nrow(short_germandata), nrow(short_germandata) * 0.7) # 70% for training
  train_data <- dummy.data_std[train_index, ]
  train_label <- as.factor(short_germandata[train_index,"Default"])
  test_data <- dummy.data_std[-train_index, ]
  test_label <- as.factor(short_germandata[-train_index,"Default"])
  
  # Run LDA and predictions
  germanlda_short <- lda(Default ~ . , data.frame(train_data))
  predictions_short <- predict(germanlda_short, newdata = data.frame(test_data))
  
  # Printing the confusion matrix
  cm <- confusionMatrix(predictions_short$class, test_label)
  
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
  sensitivity <- slot(ldaPerf, "y.values")[[1]]
  specificity <- 1 - slot(ldaPerf, "x.values")[[1]]
  auc <- slot(ldaAUC, "y.values")
  
  ldaError <- mean(as.numeric(predictions_short$class) !=as.numeric(test_label))
  
  # Print performance metrics
  print(cm)
  print(paste0("AUC: ", auc))
  print(paste0("Error rate:", ldaError))
  
  # Calculate false positives
  false_positives <- sum(as.numeric(predictions_short$class) == 2 & as.numeric(test_label) == 1)
  
  # Calculate false positives as a percentage
  total_negatives <- sum(as.numeric(test_label) == 1)
  false_positives_percent <- false_positives / total_negatives * 100
  
  # Print the false positives percentage
  print(paste0("False positives percentage: ", round(false_positives_percent,3), "%"))
  
  return(false_positives_percent)
}

#===============================================================================
#======================================= QDA ===================================
#===============================================================================
 
QDA.Model <- function(data) {
  
  # Select the qualitative columns in the dataset
  qual_cols <- sapply(data, is.character)
  
  # Create dummy variables for the qualitative columns
  data_dummy <- dummy_cols(data[, qual_cols], remove_first_dummy = TRUE)
  
  # Combine the dummy variables with the original dataset
  data_final <- cbind(data[, !qual_cols], data_dummy)
  
  # Use select_if to drop non-integer columns
  dummy.data <- select_if(data_final, is.integer)
  
  # Standardize the data
  dummy.data_std <- scale(dummy.data)
  
  # Remove the unwanted standardized target column
  dummy.data_std <- subset(dummy.data_std, select = -Default)
  
  # Add the new column to the beginning of the dataset
  dummy.data_std <- cbind(data$Default, dummy.data_std)
  
  # rename the target column (I hate R so much)
  colnames(dummy.data_std)[1] <- "Default"
  
  # Performing AIC Backward
  german_glm <- glm(Default~., family=binomial, data=data.frame(dummy.data_std))
  german_glm_back <- step(german_glm, trace=FALSE)
  
  # Collect the names of only the significant predictors
  sig.p.values <- summary(german_glm_back)$coefficients[,4] <= 0.05
  sig.predictor.names <- c("Default", row.names(data.frame(summary(german_glm_back)$coefficients[sig.p.values,])))
  
  # Check if the string "(Intercept)"exists in the vector
  if ("(Intercept)" %in% sig.predictor.names) {
    # Remove the string "(Intercept)" from the vector
    sig.predictor.names <- sig.predictor.names[-which(sig.predictor.names == "(Intercept)")]
  }
  short_germandata <- data.frame(data.frame(dummy.data_std)[,sig.predictor.names])
  
  # Split the data into training and testing sets
  set.seed(123) # for reproducibility
  train_index <- sample(nrow(short_germandata), nrow(short_germandata) * 0.7) # 70% for training
  train_data <- dummy.data_std[train_index, ]
  train_label <- as.factor(short_germandata[train_index,"Default"])
  test_data <- dummy.data_std[-train_index, ]
  test_label <- as.factor(short_germandata[-train_index,"Default"])
  
  # Run LDA and predictions
  germanqda_short <- qda(Default ~ . , data.frame(train_data))
  predictions_short <- predict(germanqda_short, newdata = data.frame(test_data))
  
  # Printing the confusion matrix
  cm <- confusionMatrix(predictions_short$class, test_label)
  
  # Run QDA and predictions
  germanqda_short <- qda(Default ~ . , data.frame(train_data))
  predictions_short <- predict(germanqda_short, newdata = data.frame(test_data))
  
  # Printing the confusion matrix
  cm <- confusionMatrix(predictions_short$class, test_label)
  
  # Printing the ROC curve
  roc.curve <- roc(test_label, predictions_short$posterior[,2])
  print(plot(roc.curve))
  print(roc.curve$auc)
  
  qdaPredictiction <- prediction(
    as.numeric(predictions_short$class),
    as.numeric(test_label)
  )
  
  qdaPerf <- performance(qdaPredictiction, measure = "tpr", x.measure = "fpr")
  qdaAUC <- performance(qdaPredictiction, measure = "auc")
  
  print(plot(qdaPerf))
  
  # Report the model performance metrics for the optimal K
  # Extract performance metrics
  sensitivity <- slot(qdaPerf, "y.values")[[1]]
  specificity <- 1 - slot(qdaPerf, "x.values")[[1]]
  auc <- slot(qdaAUC, "y.values")
  
  qdaError <- mean(as.numeric(predictions_short$class) !=as.numeric(test_label))
  
  # Print performance metrics
  print(cm)
  print(paste0("AUC: ", auc))
  print(paste0("Error rate:", qdaError))
  
  # Calculate false positives
  false_positives <- sum(as.numeric(predictions_short$class) == 2 & as.numeric(test_label) == 1)
  
  # Calculate false positives as a percentage
  total_negatives <- sum(as.numeric(test_label) == 1)
  false_positives_percent <- false_positives / total_negatives * 100
  
  # Print the false positives percentage
  print(paste0("False positives percentage: ", round(false_positives_percent,3), "%"))
  
  return(false_positives_percent)
}

#===============================================================================
#======================================= DICT ==================================
#===============================================================================

# Example: dict["A12"] -> "0 to 200 DM"
colname.dict = c(
  "A11" = "Less than 0 DM",
   "A12" = "0 to 200 DM",
   "A13" = "More than 200 DM",
   "A14" = "No checking account",
   "A30" = "No credit taken or all credit paid back",
   "A31" = "All credits at this bank paid back",
   "A32" = "Existing credits bac back until now",
   "A33" = "Delay in paying off credit in the past",
   "A34" = "Critical account or other credits existing",
   "A40" = "Car (new)",
   "A41" = "Car (used)",
   "A42" = "Furniture/equipment",
   "A43" = "Radio/television",
   "A44" = "Domestic appliances",
   "A45" = "Repairs",
   "A46" = "Ecuation",
   "A47" = "Vacation",
   "A48" = "Retraining",
   "A49" = "Business",
   "A410" = "Other",
   "A61" = "Less than 100 DM",
   "A62" = "100 to 500 DM",
   "A63" = "500 to 1,000 DM",
   "A64" = "More than 1,000 DM",
   "A65" = "Unknown/no savings account",
   "A71" = "Unemployed",
   "A72" = "Less than 1 year",
   "A73" = "1 to 4 years",
   "A74" = "4 to 7 years",
   "A75" = "More than 7 years",
   "A91" = "Male (divorced/separated)",
   "A92" = "Female (divorced/separated/married)",
   "A93" = "Male (single)",
   "A94" = "Male (married/widowed)",
   "A95" = "Female (single)",
   "A96" = "Male (divorced/separated/married)",
   "A101" = "None",
   "A102" = "Co-applicant",
   "A103" = "Guarantor",
   "A121" = "Real estate",
   "A122" = "Building society savings agreement/life insurance",
   "A123" = "Car or other",
   "A124" = "Unknown/no property",
   "A141" = "Bank",
   "A142" = "Stores",
   "A143" = "None",
   "A151" = "Rent",
   "A152" = "Own",
   "A153" = "Free",
   "A171" = "Unemployed/unskilled (non-resident)",
   "A172" = "Unemployed/unskilled (resident)",
   "A173" = "Skilled employee/official",
   "A174" = "Management/self-employed/highly qualified employee/officer",
   "A191" = "None",
   "A192" = "Yes (registered uner the customer's name)",
   "A201" = "Yes",
   "A202" = "No"
)




#===============================================================================
#======================================= GLM ===================================
#===============================================================================
glm_func <- function(data) {
  
  # Split the data into training and testing sets on PCA
  set.seed(123) # for reproducibility
  train_index <- sample(nrow(data), nrow(data) * 0.7) # 70% for training
  train_data <- data[train_index, ]
  test_data <- data[-train_index, ]
  test_label <- data[-train_index,"Default"]
  
  
  # Train the GLM model using the training data
  glm_model <- glm(Default ~ ., data = train_data, family = binomial(link = "logit"))
  
  # Evaluate the model performance on the test set
  glm_pred <- predict(glm_model, newdata = test_data, type = "response")
  glm_pred <- ifelse(glm_pred >= 0.5, 1, 0)
  
  glmConf <- confusionMatrix(factor(glm_pred, levels = c(0,1)), factor(test_data$Default, levels = c(0,1)))
  
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
  
  return(false_positives_percent)
}








