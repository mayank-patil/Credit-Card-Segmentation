# loading library
library("DMwR")
library(dplyr)
library(ggplot2)
#install.packages("fastDummies")
library("fastDummies")
library("FactoMineR")
#install.packages("NbClust")
library("NbClust")
#install.packages(c("cluster", "factoextra"))
library(cluster)
library("normtest")

#setting Work directory
setwd("C:/Users/MAYANK/Desktop/project")
# reading file
df = read.csv("credit-card-data.csv")
str(df)

### Handling Missing Value
missing_val = data.frame(apply(df,2,function(x){sum(is.na(x))}))
missing_val$Columns = row.names(missing_val)
missing_val$Missing_percent = apply(df,2,function(x){sum(is.na(x))})
names(missing_val)[1] =  "Missing_count"
missing_val$Missing_percent = (missing_val$Missing_percent/nrow(df)) * 100
missing_val = missing_val[order(-missing_val$Missing_percent),]
row.names(missing_val) = NULL
missing_val = missing_val[,c(2,1,3)]
missing_val

# We can see that only 2 columns have missing value and which are less than 30% of the total values so we can impute them.

###

### Imputing Missing Value
#1. we will delete some know values from the data frame
#2. then we will use diffrent imputing techniques 
#3. we will compare all imputing technique with the actual value
#4. we will choose the imputation technique whose results are closer to the actual values.

df1 =df
df1$MINIMUM_PAYMENTS[c(6,506,5006,8006)] 

df1$MINIMUM_PAYMENTS[6] =NA
df1$MINIMUM_PAYMENTS[506] =NA
df1$MINIMUM_PAYMENTS[5006] = NA
df1$MINIMUM_PAYMENTS[8006] =NA

#Mean Method
df1$MINIMUM_PAYMENTS[is.na(df1$MINIMUM_PAYMENTS)] = mean(df1$MINIMUM_PAYMENTS, na.rm = T)
df1$MINIMUM_PAYMENTS[c(6,506,5006,8006)] 

#Median Method
df1 =df
df1$MINIMUM_PAYMENTS[c(6,506,5006,8006)] 
df1$MINIMUM_PAYMENTS[6] =NA
df1$MINIMUM_PAYMENTS[506] =NA
df1$MINIMUM_PAYMENTS[5006] = NA
df1$MINIMUM_PAYMENTS[8006] =NA

df1$MINIMUM_PAYMENTS[is.na(df1$MINIMUM_PAYMENTS)] = median(df1$MINIMUM_PAYMENTS, na.rm = T)
df1$MINIMUM_PAYMENTS[c(6,506,5006,8006)] 

# kNN Imputation

df1 =df
df1$MINIMUM_PAYMENTS[c(6,506,5006,8006)] 
df1$MINIMUM_PAYMENTS[6] =NA
df1$MINIMUM_PAYMENTS[506] =NA
df1$MINIMUM_PAYMENTS[5006] = NA
df1$MINIMUM_PAYMENTS[8006] =NA
df1 = knnImputation(df1, k = 3)
df1$MINIMUM_PAYMENTS[c(6,506,5006,8006)] 
sum(is.na(df1))
#We can see that KNN has better results then mean and median in 
#filling missing value so we will fill the missing values with KNN.
#####
### Deriving KPIs

#### 1- monthly average purchase and cash advance amount
#* monthly average purchase = total amount of purchases / total tenure 
#* monthly average cash advance = total cash advance / total tenure
df2 = df1
df2 = df2 %>% mutate(
  AVG_MONTH_PURCHASE = round(PURCHASES/TENURE,digits=2),
  AVG_MONTH_CASH_ADVANCE = round(CASH_ADVANCE/TENURE,digits=2))
#df2
#### 2- Purchases by type (one-off, installments)
df2=df2 %>% mutate(.,PURCHASE_TYPE = with(.,case_when(
  (ONEOFF_PURCHASES != 0.0  & INSTALLMENTS_PURCHASES!=0.0) ~ "BOTH",
  (ONEOFF_PURCHASES == 0.0  & INSTALLMENTS_PURCHASES!=0.0) ~ "INSTALLMENTS",
  (ONEOFF_PURCHASES != 0.0  & INSTALLMENTS_PURCHASES==0.0) ~ "ONEOFF",
  (ONEOFF_PURCHASES == 0.0  & INSTALLMENTS_PURCHASES==0.0) ~ "NONE",
)))
#df2$PURCHASE_TYPE = as.factor(df2$PURCHASE_TYPE)
str(df2)
table(df2$PURCHASE_TYPE)
#### 3- Average amount per purchase and cash advance transaction
#*  Average amount per purchase transaction = total amount of purchases / total purchase transaction. 
#*  Average amount cash advance transaction = total cash advance / total cash_advance transcation.
df2 = df2 %>% mutate(
  AVG_AMT_PURCHASE_TRX = PURCHASES/PURCHASES_TRX,
  AVG_AMT_CASH_ADVANCE_TRX = CASH_ADVANCE/CASH_ADVANCE_TRX)
df2[is.na(df2)] <- 0.0
#####
#### 4- Limit usage (balance to credit limit ratio)
#It tells us how much debt someone is carring or how much credit they are using from there existing limit.
#* credit utilization ratio = balance used by the customer / credit limit
df2 = df2 %>% mutate(
  CREDIT_UTLIZATION_RATIO = round(BALANCE/CREDIT_LIMIT,digits=2)*100)
####
#### 5- Payments to minimum payments ratio etc.
df2 = df2 %>% mutate(
  PAY_MIN_PAY_RATIO = round(PAYMENTS/MINIMUM_PAYMENTS,digits=2))
names(df2)
####
#### 6- Using the derived KPIs to gain insight on the customer profiles.
##### Distribution Of Purchase Types

ggplot(df2, aes(PURCHASE_TYPE,fill=PURCHASE_TYPE)) + 
  ggtitle("Distribution Of Users in Diffrent Purchase Types") +
  geom_bar()
####
##### Credit Utlization By Diffrent Purchase Types
ggplot(df2, aes(x=PURCHASE_TYPE,y=CREDIT_UTLIZATION_RATIO,fill=PURCHASE_TYPE)) +
  ggtitle("Credit Utlization By Diffrent Purchase Types") +
  geom_col()
#####
##### Total Of Purchases By Diffrent Purchase Type
ggplot(df2, aes(x=PURCHASE_TYPE,y=PURCHASES,fill=PURCHASE_TYPE)) +
  ggtitle("Total Of Purchases By Diffrent Purchase Type") +
  geom_col()
####
#### 7- Identification of the Relationships/ affinities between services.
df2=df2 %>% mutate(.,PAYMENT_INFO = with(.,case_when(
  (PAY_MIN_PAY_RATIO == 0.0) ~ "PAID_FULL_DUE",
  (PAY_MIN_PAY_RATIO > 0.0) ~ "PAID_MIN_DUE",
)))
#df2
table(df2$PAYMENT_INFO)
#####  Out of 8950 observation only 
#* 240 users paid full dues
#* the rest 8710 users only paid minimum dues of there card bill
#* i.e 97 % user only pay min payment of there card bill
####
##### Joint two way Table of Payment Info and Purchase Type
table(df2$PURCHASE_TYPE,df2$PAYMENT_INFO)

####
ggplot(df2, aes(x=PURCHASE_TYPE,y=PURCHASES,fill=PAYMENT_INFO)) +
  ggtitle("Total Of Purchases By Diffrent Purchase Type and Purchase Info") +
  geom_bar(stat="identity", position=position_dodge())
####
ggplot(df2, aes(x=PURCHASE_TYPE,y=CREDIT_UTLIZATION_RATIO,fill=PAYMENT_INFO)) +
  ggtitle("Credit Utlization By Diffrent Purchase Types") +
  geom_bar(stat="identity", position=position_dodge())
####

### Scaling Variables
#Prior to applying a clustering algorithm we need to recentre the scales of our variables 
#* First we will check convert all the objects into dummy variable.
#* Second we will check for normality of data using jarque-bera test so that we can apply a suitable scaling technique.
####
##### Converting into Dummy Variables

df3 = df2
move_cols <- sapply(df3, is.character)
dum =df2[move_cols]
dum =dummy_cols(dum)
num1 = sapply(dum, is.numeric)
dum = dum[num1]
num2 = sapply(df3, is.numeric)
df3 = df3[num2]
df3 = cbind(df3,dum)

############################################Outlier Analysis#############################################
df4 = df3
# ## BoxPlots - Distribution and Outlier Check
numeric_index = sapply(df,is.numeric) #selecting only numeric

numeric_data = df4[,numeric_index]
cnames = colnames(numeric_data)
cnames
# 
# #Remove outliers using boxplot method
#Detect and delete outliers from data
for(i in cnames){
 print(i)
   val = df4[,i][df4[,i] %in% boxplot.stats(df4[,i])$out]   
   print(length(val))
   df4 = df4[which(!df4[,i] %in% val),] 
   }

###########################################Feature Scaling##############################################
#Prior to applying a clustering algorithm we need to recentre the scales of our variables 
#* first we will check for normality of data using shapiro-wilk  test so that we can apply a suitable scaling technique.

##### Applying shapiro-wilk test on all the numerical variables of data frame
#at alpha value of 0.05
#* Null hypothesis: data comes from a normal distribution.
#* Alternate hypothesis: data doesnot comes from a normal distribution.

alpha = 0.05
for(i in cnames){
  print(i)
  s = ks.test(df4[,i],"pnorm", mean=mean(df4[,i]), sd=sd(df4[,i]))
  a = unlist(s[2])
  print(a)
  
  if ( a > alpha || is.na(a) == T){
    print('Sample looks Gaussian (fail to reject H0)')
  }
  else {
    print('Sample does not look Gaussian (reject H0)')  
  }
} 

#Since p-value is less than our alpha which shows strong evidence against our Null hypothesis 
#* we will Reject the null hypothesis stating that the following columns the dataframe are not normal
#####################
#### Since are data is not normal we will scale the variable using min max normalization technique.
# function to normalize
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

df4 = normalize(df4)

### Feature Selection using  a Technique PCA which is a Derivative Of Factor Analysis


pca_output_ten_v <- PCA(df4, ncp =29, graph = FALSE)
#proportion of variance explained
sum(pca_output_ten_v$eig[,2][1:10])

# calculating optimal no. of compnents
plot(1:29,pca_output_ten_v$eig[,2][1:29],type = "o")
#### From the Above chart we can see after the 10th Component the Variance tends to Increase very slowly.
#So, 10 will be are optimal no. of component which explains around 96% variance #
pca_output_ten_v <- PCA(df4, ncp =10, graph = FALSE)
pcaa = predict(pca_output_ten_v,df4)
print(pcaa)
fin_pca =as.data.frame(pcaa$cos2)
length(fin_pca$Dim.1)

############################applying K-means############

NBclust_res = NbClust(fin_pca, min.nc=2, max.nc=10, method = "kmeans")
#* Among all indices:                                                
#* 4 proposed 2 as the best number of clusters 
#* 12 proposed 3 as the best number of clusters 
#* 4 proposed 4 as the best number of clusters 
#* 1 proposed 6 as the best number of clusters 
#* 2 proposed 10 as the best number of clusters 

# bulding Kmeans model with 3 cluster
kmeans_model = kmeans(fin_pca, 3, nstart=30)
kmeans_model
table(kmeans_model$cluster)
##### Output
#* Almost half of the records were outliers and were removed during the process.
#* It was necessary because Kmeans is affected by outliers and extreem values
#* 3 is the optimum no. of cluster
#######################################################################


##### Kmedoids
#* Since in kmeans removing outliers was a neccesity but it came with a cost of loss 
#of almost 80% of the information in order to prevent the loss of data we need 
#to use a method that is robust to outliers and extreem values and its called k medoids 
#well medoids has almost same math but instead of estimating centroids using mean it 
#takes the closest observation and use it to define the cluster which is called as a medoid.
df5 = df3
#######for full data##
#df5 = normalize(df5)
for(i in cnames){
  print(i)
  df5[,i] = normalize(df5[,i])
} 
df5[is.na(df5)] = 0.0
###############appying PCA on full data#############
### Feature Selection using  a Technique PCA which is a Derivative Of Factor Analysis
length(df5$BALANCE)
pca_output_ten_v <- PCA(df5, ncp =29, graph = FALSE)
#proportion of variance explained
sum(pca_output_ten_v$eig[,2][1:18])
# calculating optimal no. of components
plot(1:29,pca_output_ten_v$eig[,2][1:29],type = "o")
#### From the Above chart we can see after the 18th Component the Variance tends to Increase very slowly.
#So, 18 will be are optimal no. of component which explains around 96% variance #
pca_output_ten_v <- PCA(df5, ncp =18, graph = FALSE)
pcaa = predict(pca_output_ten_v,df5)
print(pcaa)
fin_pca =as.data.frame(pcaa$cos2)
length(fin_pca$Dim.1)
#running pam funtion to run k medoids
l = 1:10
asw = c()
for (k in 2:10){
  print(k)
  asw[k] <- pam(fin_pca, k)$silinfo$avg.width
}


#plotting siloute width to find optimal no. of cluster
plot(l,asw,type = "o")
length(asw)
length(l)
#since the less the no. of cluster the better the clustering is
# i.e why we will choose 4 as the optimal no. of cluster.
pam.res <- pam(fin_pca, 4)

# comparing our cluster with the labels we created earlier.
table(pam.res$clustering,df2$PURCHASE_TYPE)


