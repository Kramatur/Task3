                      #############################
                      ###    Decision Tree      ###
                      #############################
#Setting the Working Directory
setwd("D:/Study Materials/Internship/Task3/3rdTask")
# Importing the dataset 
Data_iris <- read.csv("Iris.csv", stringsAsFactors = TRUE)
head(Data_iris,5) # View the top five rows of the data in R console
str(Data_iris) # Showing the Structure of the data

#Preparing the data for analysis and Understanding the dataset
IrisDT <- subset(Data_iris, select= -c(Id)) # Removing the 'Id' column from the dataset 
head(IrisDT,5)
str(IrisDT)

#Let's see the first 5 rows of data for each class
subset(IrisDT, Species == "Iris-setosa")[1:5,]
subset(IrisDT, Species == "Iris-versicolor")[1:5,]
subset(IrisDT, Species == "Iris-virginica")[1:5,]
# Get column "Species" for all lines where PetalLengthCm < 2.5
subset(IrisDT, PetalLengthCm < 2.5)[,"Species"]

#Splitting the Data into Training and Testing Sets
# Load the Caret package which allows us to partition the data
library(caret)
# We use the dataset to create a partition (80% training 20% testing)
index <- createDataPartition(IrisDT$Species, p=0.80, list=FALSE)
# select 80% of data to train and test the models
trainset <- IrisDT[index,]
# select 20% of the data for testing
validateset <- IrisDT[-index,]

#####Let's Explore the dataset#####

# Dimensions of the data
dim(trainset)
# Structure of the data (Trainset)
str(trainset)
# Summary of the data (Trainset)
summary(trainset)
# Levels of the prediction column
levels(trainset$Species)

# summarize the class distribution
percentage <- prop.table(table(trainset$Species)) * 100
cbind(freq=table(trainset$Species), percentage=percentage)

#Visualizing the Data
#Histograms Showing frequency distribution 
par(mfrow=c(1,4)) # Dividing the area into 4 columns
hist(IrisDT$SepalLengthCm, col = "red")
hist(IrisDT$SepalWidthCm, col = "green")
hist(IrisDT$PetalLengthCm, col="blue")
hist(IrisDT$PetalWidthCm, col = "yellow")
#To save the graph as PDF
dev.copy(pdf,file="Histograms.pdf") 
dev.off()

## Box plot to understand how the distribution varies by class of flower
par(mfrow=c(1,4))
for(i in 1:4) {
  boxplot(trainset[,i], main=names(trainset)[i]) #using for loop to the all the four results at once
}                                                # we can use this for loop to get the above plots too
dev.copy(pdf,file="boxplot.pdf") 
dev.off()

## Distribution of the values considering each class
irisSet <- subset(IrisDT, Species == "Iris-setosa")
irisVer <- subset(IrisDT, Species == "Iris-versicolor")
irisVir <- subset(IrisDT, Species == "Iris-virginica")
par(mfrow=c(1,3),mar=c(6,3,2,1))
boxplot(irisSet[,1:4], main="Setosa",ylim = c(0,8),las=2)
boxplot(irisVer[,1:4], main="Versicolor",ylim = c(0,8),las=2)
boxplot(irisVir[,1:4], main="Virginica",ylim = c(0,8),las=2)
dev.copy(pdf,file="boxplot according to class.pdf") 
dev.off()

# loading ggplot2 
library(ggplot2)
# Scatter plot with smoothing line
scatterplot <- ggplot(data=trainset, aes(x = PetalLengthCm, y = PetalWidthCm))
scatterplot <-scatterplot + 
  geom_point(aes(color=Species, shape=Species)) +
  xlab("Petal Length") +
  ylab("Petal Width") +
  ggtitle("Petal Length-Width")+
  geom_smooth(method="lm")
print(scatterplot)
dev.copy(pdf,file="ScatterplotP.pdf") 
dev.off()

## Box Plot Specifically
box <- ggplot(data=trainset, aes(x=Species, y=SepalLengthCm)) +
  geom_boxplot(aes(fill=Species)) + 
  ylab("Sepal Length") +
  ggtitle("Iris Boxplot") +
  stat_summary(fun.y=mean, geom="point", shape=5, size=4) 
print(box)
dev.copy(pdf,file="BoxplotS.pdf") 
dev.off()

## Histogram Specifically
library(ggthemes)
histogram <- ggplot(data=IrisDT, aes(x=SepalWidthCm)) +
  geom_histogram(binwidth=0.2, color="yellow", aes(fill=Species)) + 
  xlab("Sepal Width") +  
  ylab("Frequency") + 
  ggtitle("Histogram of Sepal Width")+
  theme_economist()
print(histogram)
dev.copy(pdf,file="HistogramS.pdf") 
dev.off()

## Faceting: Producing multiple charts in one plot
library(ggthemes)
facet <- ggplot(data=trainset, aes(SepalLengthCm, y=SepalWidthCm, color=Species))+
  geom_point(aes(shape=Species), size=1.5) + 
  geom_smooth(method="lm") +
  xlab("Sepal Length") +
  ylab("Sepal Width") +
  ggtitle("Faceting") +
  theme_fivethirtyeight() +
  facet_grid(. ~ Species) # Along rows
print(facet)
dev.copy(pdf,file="Faceting.pdf") 
dev.off()

##### Evaluating Algorithms for Classification tree######
#Test Harness
#We will 10-fold crossvalidation to estimate accuracy.
#This will split our dataset into 10 parts, train in 9 and test on 1 and release for all combinations
#of train-test splits. We will also repeat the process 3 times for each algorithm with different
#splits of the data into 10 groups, in an effort to get a more accurate estimate.
# Run algorithms using 10-fold cross validation

control <- trainControl(method="cv", number=10)
metric <- "Accuracy"

## Model Building ##
#Linear Discriminant Analysis (LDA)
#Classification and Regression Trees (CART).
#k-Nearest Neighbors (kNN).
#Support Vector Machines (SVM) with a linear kernel.
#Random Forest (RF)

# a) linear algorithms
set.seed(5)
fit.lda <- train(Species~., data=trainset, method="lda", metric=metric, trControl=control)
# b) nonlinear algorithms
# CART
set.seed(5)
fit.cart <- train(Species~., data=trainset, method="rpart", metric=metric, trControl=control)
# kNN
set.seed(5)
fit.knn <- train(Species~., data=trainset, method="knn", metric=metric, trControl=control)
# c) advanced algorithms
# SVM
set.seed(5)
fit.svm <- train(Species~., data=trainset, method="svmRadial", metric=metric, trControl=control)
# Random Forest
set.seed(5)
fit.rf <- train(Species~., data=trainset, method="rf", metric=metric, trControl=control)

#Let's see the predictions
pred_lda <- predict(fit.lda,validateset)
pred_lda
confusionMatrix(pred_lda, validateset$Species)
# We can use same code for predicting and checking the accuracy for the rest Models

# Best Model Selection #

# summarize accuracy of models
results <- resamples(list(lda=fit.lda, cart=fit.cart, knn=fit.knn, svm=fit.svm, rf=fit.rf))
summary(results)

# compare accuracy of models
dotplot(results)
dev.copy(pdf,file="Accuracy.pdf") 
dev.off()

# summarize Best Model
print(fit.lda)
importantvar <- varImp(fit.lda)
plot(importantvar)
dev.copy(pdf,file="Important variable.pdf") 
dev.off()

#### Visualize Model Output using rattle ####
library(rattle)
fancyRpartPlot(fit.cart$finalModel)
dev.copy(pdf,file="Decision Tree.pdf") 
dev.off()

#### Confusion Matrix on training and Validation data ####
train.cart<-predict(fit.cart,newdata=trainset)
table(train.cart,trainset$Species)
pred.cart<-predict(fit.cart,newdata=validateset)
table(pred.cart,validateset$Species)
confusionMatrix(pred.cart, validateset$Species)

#### Visualize the prediction and identify incorrect predictions ####
#Using Classification and Regression Trees (CART)
correct <- pred.cart == validateset$Species
scatter2 <- ggplot(data=validateset, aes(x = PetalLengthCm, y = PetalWidthCm)) 
scatter2 + geom_point(aes(color=correct)) +
  xlab("Petal Length") +  ylab("Petal Width") +
  ggtitle("Classification Accuracy")
dev.copy(pdf,file="Classification Accuracy cart.pdf") 
dev.off()

#Using Linear Discriminant Analysis (LDA) THE BEST FIT MODEL
correct1 <- pred_lda == validateset$Species
scatter2 <- ggplot(data=validateset, aes(x = PetalLengthCm, y = PetalWidthCm)) 
scatter2 + geom_point(aes(color=correct1)) +
  xlab("Petal Length") +  ylab("Petal Width") +
  ggtitle("Classification Accuracy")
dev.copy(pdf,file="Classification Accuracy lda.pdf") 
dev.off()


