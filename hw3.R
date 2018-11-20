
require(data.table)
library(dplyr)
library("rgl")
library(knnGarden)
library("plot3D")

rm(list = ls())
require(data.table)
install.packages("dplyr")
library(dplyr)
install.packages("rgl")
library("rgl")
install.packages("knnGarden")
library(knnGarden)

install.packages("plyr")

library(plyr)
install.packages("plot3D")
library("plot3D")
install.packages("FNN")
install.packages("TunePareto")

require(FNN)
require(glmnet)
require(TunePareto)
require(data.table)


filename='UWAVE_TRAIN'
Files=list.files(paste0("C:/Users/eda.yildiz/Downloads/",filename))

traindatax <- as.matrix(read.table(paste0("C:/Users/eda.yildiz/Downloads/",filename,"/",Files[1])))  
traindatay <- as.matrix(read.table(paste0("C:/Users/eda.yildiz/Downloads/",filename,"/",Files[2])))  
traindataz <- as.matrix(read.table(paste0("C:/Users/eda.yildiz/Downloads/",filename,"/",Files[3])))  

trainclass=traindatax[,1] # takes -1 and 1

#drop first column
traindatax=traindatax[,2:ncol(traindatax)]
traindatay=traindatay[,2:ncol(traindatay)]
traindataz=traindataz[,2:ncol(traindataz)]


#Check if same
print(dim(traindatax)) #shows that there 100 series (rows) of length 96 time units (columns)
print(dim(traindatay)) #shows that there 100 series (rows) of length 96 time units (columns)
print(dim(traindataz)) #shows that there 100 series (rows) of length 96 time units (columns)

noftimeseries=nrow(traindatax)

unique(trainclass)
#Select instances from each class
trainclass[11]
trainclass[15]
trainclass[4]
trainclass[5]
trainclass[2]
trainclass[1]
trainclass[7]
trainclass[6]



#let's use line format
plot3d(traindatax[11,], traindatay[11,], traindataz[11,],type="l")
plot3d(traindatax[15,], traindatay[15,], traindataz[15,],type="l")
plot3d(traindatax[4,], traindatay[4,], traindataz[4,],type="l")
plot3d(traindatax[5,], traindatay[5,], traindataz[5,],type="l")
plot3d(traindatax[2,], traindatay[2,], traindataz[2,],type="l")
plot3d(traindatax[1,], traindatay[1,], traindataz[1,],type="l")
plot3d(traindatax[7,], traindatay[7,], traindataz[7,],type="l")
plot3d(traindatax[6,], traindatay[6,], traindataz[6,],type="l")


plot3d(cumsum(traindatax[11,]), cumsum(traindatay[11,]), cumsum(traindataz[11,]),type="l")
plot3d(cumsum(traindatax[15,]), cumsum(traindatay[15,]), cumsum(traindataz[15,]),type="l")
plot3d(cumsum(traindatax[4,]), cumsum(traindatay[4,]), cumsum(traindataz[4,]),type="l")
plot3d(cumsum(traindatax[5,]), cumsum(traindatay[5,]), cumsum(traindataz[5,]),type="l")
plot3d(cumsum(traindatax[2,]), cumsum(traindatay[2,]), cumsum(traindataz[2,]),type="l")
plot3d(cumsum(traindatax[1,]), cumsum(traindatay[1,]), cumsum(traindataz[1,]),type="l")
plot3d(cumsum(traindatax[7,]), cumsum(traindatay[7,]), cumsum(traindataz[7,]),type="l")
plot3d(cumsum(traindatax[6,]), cumsum(traindatay[6,]), cumsum(traindataz[6,]),type="l")

###Part B

TRAIN=cbind(traindatax,traindatay,traindataz)

#distMatrix=dist(TRAIN)
#print(str(distMatrix)) #it is distance matrix where first row (or column) has the distances we need
#let's convert it to a regular matrix
#distMatrix=as.matrix(distMatrix)

nofReplications=1
nFolds=10
indices=generateCVRuns(trainclass,nofReplications,nFolds,stratified=TRUE)

Result =as.data.table(t(c(0,0,0,0)))
setnames(Result,c(1:4),c('TotalNRow','Error','k','nFold'))

i=1

  thisReplication=indices[[i]]
  
 
  for(j in 1:10){
    
    testindices=thisReplication[[j]]
    
    cvtrain=TRAIN[-testindices,]        
    cvtest=TRAIN[testindices,]
    TrainClass=trainclass[-testindices]
    TestClass=trainclass[testindices]
   
     
     All=rbind(cvtrain,cvtest)
     Classes=rbind(as.matrix(TrainClass),as.matrix(TestClass))
     
     Distance=as.matrix(dist(All))
     
     Distance=as.data.table(Distance)
     Distance2=Distance
     
     Distance2=Distance2[1:nrow(cvtrain)]
     
     y=nrow(cvtrain)+1
     x=ncol(Distance2)
     
     ff=Distance2[,c(y:x),with=FALSE]
     Neighbors=apply(ff,2,order)
     
     Neighbors2=Neighbors
     z=y-1
     Neighbors2=mapvalues(Neighbors2, 
                          from=c(1:z), 
                          to=TrainClass)
     
     for(k in 1:10) {
     K3=as.data.table(Neighbors2[1:k,])
      
     getmode <- function(v) {
       uniqv <- unique(v)
       uniqv[which.max(tabulate(match(v, uniqv)))]
     }
     
     if (k==1) {  dat <- apply( K3,1, getmode)}
     if (k!=1) {  dat <- apply( K3,2, getmode)}
     
     TestSonuc=as.data.table(cbind(dat,TestClass))
     
     TestSonuc[dat==TestClass,Same:=1]
     TestSonuc[dat!=TestClass,Same:=0]
     ThisResult =as.data.table(t(c(nrow(TestSonuc),sum(TestSonuc$Same),k,j)))
     ThisResult
     setnames(ThisResult,c(1:4),c('TotalNRow','Error','k','nFold'))
     Result=rbind(Result,ThisResult)
     
     print (c(k,j))
  }
 }
 

Result=Result[k!=0]
save(Result,file="C:/Users/eda.yildiz/Desktop/hw3PartBEucledian.rdata")

setnames(Result,'Error','Accuracy')
Result[,Accuracy:= (Accuracy)/ TotalNRow]
Result[,mean(Accuracy),k]



ResultManhattan =as.data.table(t(c(0,0,0,0)))
setnames(ResultManhattan,c(1:4),c('TotalNRow','Error','k','nFold'))


for(j in 1:10){
  
  testindices=thisReplication[[j]]
  
  cvtrain=TRAIN[-testindices,]        
  cvtest=TRAIN[testindices,]
  TrainClass=trainclass[-testindices]
  TestClass=trainclass[testindices]
  
  
  All=rbind(cvtrain,cvtest)
  Classes=rbind(as.matrix(TrainClass),as.matrix(TestClass))
  
  Distance=as.matrix(dist(All,method="manhattan"))
  
  Distance=as.data.table(Distance)
  Distance2=Distance
  
  Distance2=Distance2[1:nrow(cvtrain)]
  
  y=nrow(cvtrain)+1
  x=ncol(Distance2)
  
  ff=Distance2[,c(y:x),with=FALSE]
  Neighbors=apply(ff,2,order)
  
  Neighbors2=Neighbors
  z=y-1
  Neighbors2=mapvalues(Neighbors2, 
                       from=c(1:z), 
                       to=TrainClass)
  
  for(k in 1:10) {
    K3=as.data.table(Neighbors2[1:k,])
    
    getmode <- function(v) {
      uniqv <- unique(v)
      uniqv[which.max(tabulate(match(v, uniqv)))]
    }
    
    if (k==1) {  dat <- apply( K3,1, getmode)}
    if (k!=1) {  dat <- apply( K3,2, getmode)}
    
    
    TestSonuc=as.data.table(cbind(dat,TestClass))
    
    TestSonuc[dat==TestClass,Same:=1]
    TestSonuc[dat!=TestClass,Same:=0]
    ThisResult =as.data.table(t(c(nrow(TestSonuc),sum(TestSonuc$Same),k,j)))
    ThisResult
    setnames(ThisResult,c(1:4),c('TotalNRow','Error','k','nFold'))
    ResultManhattan=rbind(ResultManhattan,ThisResult)
    
    print (c(k,j))
  }
}



ResultManhattan=ResultManhattan[k!=0]
save(ResultManhattan,file="C:/Users/eda.yildiz/Desktop/hw3PartBManhattan.rdata")

setnames(ResultManhattan,'Error','Accuracy')
ResultManhattan[,Accuracy:= (Accuracy)/ TotalNRow]
ResultManhattan[,mean(Accuracy),k]



  ##C Part 
  
k=4

  filename='UWAVE_TEST'
  Files=list.files(paste0("C:/Users/eda.yildiz/Downloads/",filename))
  
  testdatax <- as.matrix(read.table(paste0("C:/Users/eda.yildiz/Downloads/",filename,"/",Files[1])))
  testdatay <- as.matrix(read.table(paste0("C:/Users/eda.yildiz/Downloads/",filename,"/",Files[2])))  
  testdataz <- as.matrix(read.table(paste0("C:/Users/eda.yildiz/Downloads/",filename,"/",Files[3])))  
  
  testclass=testdatax[,1] # takes -1 and 1
  
  #drop first column
  testdatax=testdatax[,2:ncol(testdatax)]
  testdatay=testdatay[,2:ncol(testdatay)]
  testdataz=testdataz[,2:ncol(testdataz)]
  
  
  TEST=cbind(testdatax,testdatay,testdataz)
  
  Start=Sys.time()

  All=rbind(TRAIN,TEST)
  Classes=rbind(as.matrix(trainclass),as.matrix(testclass))
  
  Distance=as.matrix(dist(All))
  
  Distance=as.data.table(Distance)
  Distance2=Distance
  
  Distance2=Distance2[1:nrow(TRAIN)]
  
  y=nrow(TRAIN)+1
  x=ncol(Distance2)
  
  ff=Distance2[,c(y:x),with=FALSE]
  Neighbors=apply(ff,2,order)
  
  Neighbors2=Neighbors
  z=y-1
  Neighbors2=mapvalues(Neighbors2, 
                       from=c(1:z), 
                       to=trainclass)
  
  
  K3=as.data.table(Neighbors2[1:k,])
  
  dat <- apply( K3,2, as.numeric )
  dat=as.data.table(dat)
  
  getmode <- function(v) {
    uniqv <- unique(v)
    uniqv[which.max(tabulate(match(v, uniqv)))]
  }
  
  dat <- apply( K3,2, getmode )
  
  
  TestSonuc=as.data.table(cbind(dat,testclass))
  setnames(TestSonuc,'dat','Prediction')
  setnames(TestSonuc,'testclass','Actual')
  
  table(TestSonuc)
  
  Med=Sys.time()
  
  TimeElapsed=Med-Start
  
  save(TimeElapsed,file="C:/Users/eda.yildiz/Desktop/hw3PartCEucledianTime.rdata")
  save(TestSonuc,file="C:/Users/eda.yildiz/Desktop/hw3PartCEucledian.rdata")
  
  
  
  #############
  
  StartMan=Sys.time()
  
  Distance=as.matrix(dist(All,method="manhattan"))
  
  Distance=as.data.table(Distance)
  Distance2=Distance
  
  Distance2=Distance2[1:nrow(TRAIN)]
  
  y=nrow(TRAIN)+1
  x=ncol(Distance2)
  
  ff=Distance2[,c(y:x),with=FALSE]
  Neighbors=apply(ff,2,order)
  
  Neighbors2=Neighbors
  z=y-1
  Neighbors2=mapvalues(Neighbors2, 
                       from=c(1:z), 
                       to=trainclass)
  
  
  K3=as.data.table(Neighbors2[1:k,])
  
  dat <- apply( K3,2, as.numeric )
  dat=as.data.table(dat)
  
  getmode <- function(v) {
    uniqv <- unique(v)
    uniqv[which.max(tabulate(match(v, uniqv)))]
  }
  
  dat <- apply( K3,2, getmode )
  
  
  TestSonucMan=as.data.table(cbind(dat,testclass))
  setnames(TestSonucMan,'dat','Prediction')
  setnames(TestSonucMan,'testclass','Actual')
  
  table(TestSonucMan)
  
  MedMan=Sys.time()
  
  TimeElapsedMan=MedMan-StartMan
  
  save(TimeElapsedMan,file="C:/Users/eda.yildiz/Desktop/hw3PartCManTime.rdata")
  save(TestSonucMan,file="C:/Users/eda.yildiz/Desktop/hw3PartCMan.rdata")
  
  
  
  
##### Task2 
  
  
  install.packages("penalized")
  library(penalized)
  
  require(data.table)
  
  fname='ecgTRAIN' # data path
  traindata <- read.table((paste0("C:/Users/eda.yildiz/Downloads/NN_Classification/",fname)))  # read data into a matrix named traindata
  #first column is the class variable
  trainclass=traindata[,1] # takes -1 and 1rainclass[trainclass==-1]=0
  
  trainclass <- as.data.table(trainclass)
  trainclass[trainclass == -1, trainclass := 0]
  trainclass <- as.matrix(trainclass)
  
  #drop first column
  traindata=traindata[,2:ncol(traindata)]
  print(dim(traindata)) #shows that there 100 series (rows) of length 96 time units (columns)
  
  
  #read test data
  fname='ecgTEST' # data path
  testdata <- read.table((paste0("C:/Users/eda.yildiz/Downloads/NN_Classification/",fname)))  # read data into a matrix named traindata
  #first column is the class variable
  testclass=testdata[,1] # takes -1 and 1
  
  testclass <- as.data.table(testclass)
  testclass[testclass == -1, testclass := 0]
  testclass <- as.matrix(testclass)
  
  #drop first column
  testdata=testdata[,2:ncol(testdata)]
  
  
  LogReg=optL2(response = trainclass,
               penalized = traindata,
               unpenalized = ~ 0,
               data = traindata,
               fusedl = T,
               standardize = T,
               lambda1 = 1, minlambda2 = 0, maxlambda2 = 3,
               fold = 2,
               model = "logistic")
  
  
  
  TestResult=predict(object =LogReg$fullfit, as.matrix(testdata))
  
  TestResult[TestResult<0.5]=0
  TestResult[TestResult>=0.5]=1
  
  LogisticResult=cbind(TestResult,testclass)
  
  
  LogisticResult=as.data.table(LogisticResult)
  
  setnames(LogisticResult,'V1','Actual')
  setnames(LogisticResult,'TestResult','Prediction')
  
  table(LogisticResult)
  
  
  #######
  
  NewTrain=as.data.table(traindata)
  SubstractedTrain=as.data.table(rep(0,100))
  SubstractedTrain=cbind(SubstractedTrain,NewTrain[,c(1:ncol(NewTrain)-1),with=FALSE])
  
  Conc=NewTrain-SubstractedTrain
  Conc$V2=NULL
  
  LogRegDiff=optL2(response = trainclass,
                   penalized = Conc,
                   unpenalized = ~ 0,
                   data = Conc,
                   fusedl = T,
                   standardize = T,
                   lambda1 = 1, minlambda2 = 0, maxlambda2 = 3,
                   fold = 2,
                   model = "logistic")
  
  
  
  NewTest=as.data.table(testdata)
  SubstractedTest=as.data.table(rep(0,100))
  SubstractedTest=cbind(SubstractedTest,NewTest[,c(1:ncol(NewTest)-1),with=FALSE])
  
  ConcTest=NewTest-SubstractedTest
  ConcTest$V2=NULL
  
  TestResultDiff=predict(object =LogRegDiff$fullfit, as.matrix(ConcTest))
  
  TestResultDiff[TestResultDiff<0.5]=0
  TestResultDiff[TestResultDiff>=0.5]=1
  
  LogisticResultDiff=cbind(TestResultDiff,testclass)
  
  
  LogisticResultDiff=as.data.table(LogisticResultDiff)
  
  sum(LogisticResultDiff[V1==1,TestResultDiff==1])
  sum(LogisticResultDiff[V1==1,TestResultDiff==0])
  sum(LogisticResultDiff[V1==0,TestResultDiff==1])
  sum(LogisticResultDiff[V1==0,TestResultDiff==0])
  
  
  
  #check the coefficients of the best model
  plot(LogRegDiff$fullfit)
  
  
  
  
  
  
  
  
  
  
  
  
  



