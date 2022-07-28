install.packages("ggthemes")
install.packages("rpart")
install.packages('caret', dependencies = TRUE)
install.packages("CRAN")
install.packages("MASS")
install.packages("tidyverse")
install.packages("dplyr")
install.packages("e1071")
install.packages("ggthemes")
install.packages("rpart")
install.packages("rpart.plot")
library(tidyverse)
library(caret)
library(Metrics)
library(MASS)
library(dplyr)
library(e1071)
library(ggthemes)
library(rpart)
library(rpart.plot)


#input data
bc=read.csv("./Desktop/breast-cancer-wisconsin.csv",header=FALSE, sep = ",")
names(bc)=c("id","cl.thickness","cell.size","cell.shape","marg.adhesion","epith.c.size","bare.nuclei","bl.cromatin","normal.nucleoli","mitoses","class")
bc

#preprocessing
bc$bare.nuclei <- as.numeric(as.character(bc$bare.nuclei))
bc=bc %>% drop_na("bare.nuclei")


#normalize all the variables for PCA
bc.stand <- as.data.frame(scale(bc[,2:10]))
summary(bc.stand) #make sure the mean in each feature is 0
sapply(bc.stand,sd) #make sure the standard deviation in each feature is 1

#calculate PCA
pca <- prcomp(bc.stand,scale = TRUE, center = TRUE)
summary(pca)
new_bc=predict(pca,bc[,2:10])
new_bc=new_bc[,1:5]
new_bc
#variance of each PC
plot(pca,type="l") #Figure 1
#select top 5 PCs, put them to the dataset bc
bc$PC1=new_bc[,1]
bc$PC2=new_bc[,2]
bc$PC3=new_bc[,3]
bc$PC4=new_bc[,4]
bc$PC5=new_bc[,5]
#plot of PCA
pairs(bc[,12:16],col=bc$class) #Figure 2
biplot(pca,scale=0,cex=0.5) #Figure 3


#data split: train vs test
trainIndex <- createDataPartition(bc$class, p = .8, list=FALSE)
trainset=bc[trainIndex,]
testset=bc[-trainIndex,]
#comparison of trainset and testset 
hist(trainset$PC1)
hist(testset$PC1)
hist(trainset$PC2)
hist(trainset$PC2)


#Classification: SVM 
plot(bc$PC1, bc$PC2, col=bc$class,xlab="PC1",ylab="PC2") #Figure 4. observe the type of classification
svmfit=svm(class~.,data=trainset[,c(11,12:16)], probability=TRUE,kernel="linear",cross=10,type="C-classification")
p=predict(svmfit,testset[,c(11,12:16)])

#evaluation
table(x=testset$class,y=p)
confmat.linear=table(pred = p, true = testset$class) #same concept

#experiment : polynomial kernel
#取前四pc
svmfit1<-svm(class~., data = trainset[,c(11,12:15)], probability=TRUE, cross=10,type="C-classification", kernel="polynomial")
p1=predict(svmfit1,testset[,c(11,12:15)])
table(testset$class,p1) #worse than the linear kernel
#取1 pc
svmfit2<-svm(class~., data = trainset[,c(11,12)], probability=TRUE, cross=10,type="C-classification", kernel="polynomial")
p2=predict(svmfit2,testset[,c(11,12)])
table(testset$class,p2) #slightly better than 4pc
#取2 pc
svmfit3<-svm(class~., data = trainset[,c(11,12:13)], probability=TRUE, cross=10,type="C-classification", kernel="polynomial")
p3=predict(svmfit3,testset[,c(11,12:13)])
table(testset$class,p3) #slightly better than 1pc
#取3 pc
svmfit4<-svm(class~., data = trainset[,c(11,12:14)], probability=TRUE, cross=10,type="C-classification", kernel="polynomial")
p4=predict(svmfit4,testset[,c(11,12:14)])
table(testset$class,p4) #slightly better than pc

#visualization: compare the labeled class and the predicted class
plot(testset$PC1,testset$PC2,xlab="PC1", ylab="PC2",col=p)#Figure 6.
pairs(testset[,12:13],col=p)
pairs(testset[,12:13],col=testset$class)


##Decision Tree
#Generating a decision tree model with 3PCs, classifying 'class' of whether it is 2 or 4
rpart_model_bc=rpart(class~.,data=trainset[,c(11,13:14)], method="class")
#Showing results in all PCs
print(rpart_model_bc) 

#Plotting decision tree and understanding the results in numbers
plot(rpart_model_bc)
text(rpart_model_bc)
summary(rpart_model_bc)

#Visualizing the result
rpart.plot(rpart_model_bc, box.palette="RdBu", shadow.col="gray", nn=TRUE)

#Prediction- inputting testsets
typeColNum_bc <- grep('class',names(bc))
rpart_predict_bc<- predict(rpart_model_bc,testset[,-typeColNum_bc],type='class')
mn_bc <- mean(rpart_predict_bc==testset$class)
mn_bc

#Evaluation- generating a confusion matrix to evaluate the performance
table(pred=rpart_predict_bc,true=testset$class)

