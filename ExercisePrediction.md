---
title: "PredictionMachineLearning_Project"
author: "Sai Preteek Medi"
date: "Sunday, June 21, 2015"
output: html_document
---
# Prediction and Machine Learning Course Project
### Many types of sensors are out on the market which help exercise enthusiasts find statistics on their activity levels. These regular measurements form a component of "Big Data" that is amenable to analysis and prediction. Typically, these statistics merely quantify the total amount of of repetitions but can also be used to determine how well the activity is being performed. The goal of this analysis is to gather data from accelerometers attached to the belt, forearm, arm, and dumbbell of 6 participants to quantify the quality of the activities they perform.  

## Data Ingestion:


```r
setwd("C:/Users/Student/Desktop/RWorkingDir/MachineLearning/ExercisePrediction")
setInternet2(use=T)
trainingUrl<-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testingUrl<-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

if(!file.exists("training.csv")){
    download.file(trainingUrl, "training.csv");
    training<-read.csv("training.csv")}
if(!file.exists("testing.csv")){
    download.file(testingUrl, "testing.csv");
    testing<-read.csv("testing.csv")}
        
        #misspelled columns in the raw dataset is driving me crazy.... 
colnames(training)[13]<-"kurtosis_pitch_belt"
colnames(training)[19]<-"max_pitch_belt"
colnames(training)[70]<-"kurtosis_pitch_arm"
colnames(training)[76]<-"max_pitch_arm"
colnames(training)[88]<-"kurtosis_pitch_dumbbell"
colnames(training)[94]<-"max_pitch_dumbbell"
```

```
## Error in `colnames<-`(`*tmp*`, value = c("X", "user_name", "raw_timestamp_part_1", : 'names' attribute [94] must be the same length as the vector [93]
```

```r
colnames(training)[126]<-"kurtosis_pitch_forearm"
```

```
## Error in `colnames<-`(`*tmp*`, value = c("X", "user_name", "raw_timestamp_part_1", : 'names' attribute [126] must be the same length as the vector [93]
```

```r
colnames(training)[132]<-"max_pitch_forearm" 
```

```
## Error in `colnames<-`(`*tmp*`, value = c("X", "user_name", "raw_timestamp_part_1", : 'names' attribute [132] must be the same length as the vector [93]
```

```r
colnames(testing)[13]<-"kurtosis_pitch_belt"
colnames(testing)[19]<-"max_pitch_belt"
colnames(testing)[70]<-"kurtosis_pitch_arm"
```

```
## Error in `colnames<-`(`*tmp*`, value = c("user_name", "roll_belt", "pitch_belt", : 'names' attribute [70] must be the same length as the vector [53]
```

```r
colnames(testing)[76]<-"max_pitch_arm"
```

```
## Error in `colnames<-`(`*tmp*`, value = c("user_name", "roll_belt", "pitch_belt", : 'names' attribute [76] must be the same length as the vector [53]
```

```r
colnames(testing)[88]<-"kurtosis_pitch_dumbbell"
```

```
## Error in `colnames<-`(`*tmp*`, value = c("user_name", "roll_belt", "pitch_belt", : 'names' attribute [88] must be the same length as the vector [53]
```

```r
colnames(testing)[94]<-"max_pitch_dumbbell"
```

```
## Error in `colnames<-`(`*tmp*`, value = c("user_name", "roll_belt", "pitch_belt", : 'names' attribute [94] must be the same length as the vector [53]
```

```r
colnames(testing)[126]<-"kurtosis_pitch_forearm"
```

```
## Error in `colnames<-`(`*tmp*`, value = c("user_name", "roll_belt", "pitch_belt", : 'names' attribute [126] must be the same length as the vector [53]
```

```r
colnames(testing)[132]<-"max_pitch_forearm"
```

```
## Error in `colnames<-`(`*tmp*`, value = c("user_name", "roll_belt", "pitch_belt", : 'names' attribute [132] must be the same length as the vector [53]
```

```r
as.numeric(is.na(head(training, 2))) #checking for missing, NA values.
```

```
##   [1] 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
##  [36] 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
##  [71] 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
## [106] 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
## [141] 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
## [176] 0 0 0 0 0 0 0 0 0 0 0
```
## Data Processing
### First, we can filter out variables with missing values, and those representing timestamp information that is not pertinent to the algorithm.

```r
training<-training[,colSums(is.na(training))==0]
testing<-testing[,colSums(is.na(testing))==0]
```

### Because the training set is so large and there are only 20 values in testing set, we will create another testing set as a subset of the training set:

```r
set.seed(12345);library(caret)
inTrain<-createDataPartition(y=training$classe,p=0.6, list=F)
trainingSubset<-training[inTrain,];testingSubset<-training[-inTrain,]
```

### The next step is to transform the variables based on feasibility in Machine Learning Algorithms.

#### Using Near Zero Variance to filter these variables out

```r
NZV<-nearZeroVar(trainingSubset,saveMetrics=T)
trainingSubset2<-trainingSubset[,!NZV$nzv]

##conducting filtering on testingSubset
filt<-colnames(trainingSubset2[,-58])
filt2<-colnames(trainingSubset2)
testing<-testing[filt]
```

```
## Error in `[.data.frame`(testing, filt): undefined columns selected
```

```r
testingSubset<-testingSubset[filt2]
#Removing varibles related to timestamp information
trainingSubset2<-trainingSubset2[,-c(2,3,4,5)]
testingSubset<-testingSubset[,-c(2,3,4,5)]
testing<-testing[,-c(2,3,4,5)]
dim(trainingSubset2);dim(testingSubset);dim(testing)
```

```
## [1] 11776    55
```

```
## [1] 7846   55
```

```
## [1] 20 49
```

## Machine Learning Algorithm 
### Now that we have our data cleaned and partitioned, we will proceed with Random Forest Algorithm to decide which covariates to use as predictors. Cross Validation is done to add robustness to the model and we will use 6 partitions for cross-validation.


```r
    ##removing first id variable
trainingSubset2<-trainingSubset2[c(-1)]
RF<-trainControl(method="cv",6)

trainRF<-train(classe~.,data=trainingSubset2,method="rf",trControl=RF,ntree=200)
trainRF
```

```
## Random Forest 
## 
## 11776 samples
##    53 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (6 fold) 
## 
## Summary of sample sizes: 9813, 9813, 9813, 9813, 9815, 9813, ... 
## 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa      Accuracy SD  Kappa SD   
##    2    0.9921874  0.9901169  0.002875928  0.003639233
##   27    0.9961785  0.9951660  0.002006267  0.002538172
##   53    0.9950745  0.9937702  0.001316457  0.001665080
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 27.
```
### We have a Random Forest model generated with 6 Cross Validations and 200 trees. Now to predict estimates using this mode on the testing subset (not the actual testing data).

```r
predictRF<-predict(trainRF, testingSubset)
confusionMatrix(testingSubset$classe,predictRF)$overall[1]
```

```
## Accuracy 
## 0.997196
```
### The Confusion Matrix above shows statistics on how well the model predicts for the validation set.

```r
predAcc<-postResample(predictRF,testingSubset$classe)
predAcc
```

```
##  Accuracy     Kappa 
## 0.9971960 0.9964533
```

```r
o<-confusionMatrix(testingSubset$classe,predictRF)$overall[1]

outofSampleError<-1-o
outofSampleError
```

```
##    Accuracy 
## 0.002803977
```
### Using the above confusion matrix, the accuracy of the model is 99.23%, and the out of sample eror rate is 0.765%.
