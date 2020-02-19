import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# monthly average purchase and cash advance amount
# monthly average purchase = totol amount of purchases

data = pd.read_csv("credit-card-data.csv")

### Attributes Information:
#● CUST_ID Credit card holder ID

### Handling Missing Value

miss = pd.DataFrame(data.isnull().sum())
miss = miss.rename(columns={0:"miss_count"})
miss["miss_%"] = (miss.miss_count/len(data.CUST_ID))*100
miss

#we can see that only 2 columns have missing value and which are less than 30% of the total values so we can impute them.

### Imputing Missing Value
#1. we will delete some know values from the data frame
#2. then we will use diffrent imputing techniques
#3. we will compare all imputing technique with the actual value
#4. we will choose the imputation technique whose results are closer to the actual values.

data1 =data.copy()

data1['MINIMUM_PAYMENTS'].loc[5],data1['MINIMUM_PAYMENTS'].loc[505],data1['MINIMUM_PAYMENTS'].loc[5005], data1['MINIMUM_PAYMENTS'].loc[8005]

#### Imputing Using Mean

data1 =data.copy()
data1['MINIMUM_PAYMENTS'].loc[5] = np.nan
data1['MINIMUM_PAYMENTS'].loc[505] = np.nan
data1['MINIMUM_PAYMENTS'].loc[5005] = np.nan
data1['MINIMUM_PAYMENTS'].loc[8005] = np.nan

data1['MINIMUM_PAYMENTS'] = data1['MINIMUM_PAYMENTS'].fillna(data1['MINIMUM_PAYMENTS'].mean())

data1['MINIMUM_PAYMENTS'].loc[5],data1['MINIMUM_PAYMENTS'].loc[505],data1['MINIMUM_PAYMENTS'].loc[5005], data1['MINIMUM_PAYMENTS'].loc[8005]

#### Imputing using median

data1 =data.copy()
data1['MINIMUM_PAYMENTS'].loc[5] = np.nan
data1['MINIMUM_PAYMENTS'].loc[505] = np.nan
data1['MINIMUM_PAYMENTS'].loc[5005] = np.nan
data1['MINIMUM_PAYMENTS'].loc[8005] = np.nan

data1['MINIMUM_PAYMENTS'] = data1['MINIMUM_PAYMENTS'].fillna(data1['MINIMUM_PAYMENTS'].median())

data1['MINIMUM_PAYMENTS'].loc[5],data1['MINIMUM_PAYMENTS'].loc[505],data1['MINIMUM_PAYMENTS'].loc[5005], data1['MINIMUM_PAYMENTS'].loc[8005]

#### Imputing using interpolation

data1 =data.copy()
data1['MINIMUM_PAYMENTS'].loc[5] = np.nan
data1['MINIMUM_PAYMENTS'].loc[505] = np.nan
data1['MINIMUM_PAYMENTS'].loc[5005] = np.nan
data1['MINIMUM_PAYMENTS'].loc[8005] = np.nan

data1=data1.interpolate()

data1['MINIMUM_PAYMENTS'].loc[5],data1['MINIMUM_PAYMENTS'].loc[505],data1['MINIMUM_PAYMENTS'].loc[5005], data1['MINIMUM_PAYMENTS'].loc[8005]

#### Imputing using Knn imputation

data1 =data.copy()
data1['MINIMUM_PAYMENTS'].loc[5] = np.nan
data1['MINIMUM_PAYMENTS'].loc[505] = np.nan
data1['MINIMUM_PAYMENTS'].loc[5005] = np.nan
data1['MINIMUM_PAYMENTS'].loc[8005] = np.nan

import impyute as impy
aa=list(data.columns)
aa.remove("CUST_ID")
d =impy.fast_knn(data1.iloc[:,1:], k=4)
d.columns =aa
data1.iloc[:,1:] = d

d.iloc[5,14],d.iloc[505,14],d.iloc[5005,14], d.iloc[8005,14]

data1['MINIMUM_PAYMENTS'].loc[5],data1['MINIMUM_PAYMENTS'].loc[505],data1['MINIMUM_PAYMENTS'].loc[5005], data1['MINIMUM_PAYMENTS'].loc[8005]

#We can see that KNN has better results then mean, median, and interpolation in filling missing value so we will fill the missing values with KNN.

### Missing Values Removed.

miss = pd.DataFrame(data1.isnull().sum())
miss = miss.rename(columns={0:"miss_count"})
miss["miss_%"] = (miss.miss_count/len(data1.CUST_ID))*100
miss

#Here we can see that all values are complete now we can move ahead with further analysis.

### Deriving KPIs


#### 1- monthly average purchase and cash advance amount
#* monthly average purchase = total amount of purchases / total tenure
#* monthly average cash advance = total cash advance / total tenure

data2 = data1.copy()

data2["AVG_MONTH_PURCHASE"] = round(data2["PURCHASES"]/data2["TENURE"],2)
data2["AVG_MONTH_CASH_ADVANCE"] = round(data2["CASH_ADVANCE"]/data2["TENURE"],2)

#### 2- Purchases by type (one-off, installments)

data2[["ONEOFF_PURCHASES","INSTALLMENTS_PURCHASES"]].head()

#### From The Above
#we can see there are users which either purchase oneoff, installments, both or neither of the option lets segregate them with a label

# The below function segregate them into diffrent groups as per there characterstics.
def type_check(data2):
    if (data2.ONEOFF_PURCHASES!=0.0) and (data2.INSTALLMENTS_PURCHASES!=0.0 ):
        return "BOTH"
    elif (data2.ONEOFF_PURCHASES==0.0) and (data2.INSTALLMENTS_PURCHASES!=0.0 ):
        return "INSTALLMENTS"
    elif (data2.ONEOFF_PURCHASES!=0.0) and (data2.INSTALLMENTS_PURCHASES==0.0 ):
        return "ONEOFF"
    elif (data2.ONEOFF_PURCHASES==0.0) and (data2.INSTALLMENTS_PURCHASES==0.0):
        return "NONE"

# applying the function
data2["PURCHASE_TYPE"] = data2.apply(type_check,axis=1)

data2["PURCHASE_TYPE"].value_counts()

#### 3- Average amount per purchase and cash advance transaction
#*  Average amount per purchase transaction = total amount of purchases / total purchase transaction
#*  Average amount cash advance transaction = total cash advance / total cash_advance transcation

data2["AVG_AMT_PURCHASE_TRX"] = round(data2["PURCHASES"]/data2["PURCHASES_TRX"],2)
data2["AVG_AMT_CASH_ADVANCE_TRX"] = round(data2["CASH_ADVANCE"]/data2["CASH_ADVANCE_TRX"],2)
data2 = data2.fillna(0.0)

data2.head()

#### 4- Limit usage (balance to credit limit ratio)
#It tells us how much debt someone is carring or how much credit they are using from there existing limit.
#* credit utilization ratio = balance used by the customer / credit limit


data2["CREDIT_UTLIZATION_RATIO"] = round(data2["BALANCE"]/data2["CREDIT_LIMIT"],2)*100

data2.head()

#### 5- Payments to minimum payments ratio etc.

data2["PAY_MIN_PAY_RATIO"] = data2["PAYMENTS"]/data2["MINIMUM_PAYMENTS"]

#### 6- Using the derived KPIs to gain insight on the customer profiles.


data2.columns

##### Distribution Of Purchase Types

import seaborn as sns
#plt.bar(data2.PURCHASE_TYPE,data2.AVG_MONTH_PURCHASE)
sns.countplot(data=data2,x="PURCHASE_TYPE")
plt.title("Distribution Of Users in Diffrent Purchase Types")
plt.show()

##### Credit Utlization By Diffrent Purchase Types

#plt.bar(data2.PURCHASE_TYPE,data2.AVG_MONTH_PURCHASE)
sns.barplot(data=data2,x="PURCHASE_TYPE",y="CREDIT_UTLIZATION_RATIO")
plt.title("Credit Utlization By Diffrent Purchase Types")
plt.show()

##### Total Of Purchases By Diffrent Purchase Type

sns.barplot(data=data2,x="PURCHASE_TYPE",y="PURCHASES")
plt.title("Total Of Purchases By Diffrent Purchase Type")
plt.show()

#### 7- Identification of the Relationships/ affinities between services.


#####  Out of 8950 observation only


def p_check(data2):
    if data2["PAY_MIN_PAY_RATIO"]==0.0 :
        return "paid_full_due"
    if data2["PAY_MIN_PAY_RATIO"]>0.0 :
        return "paid_min_due"
data2["PAYMENT_INFO"] = data2.apply(p_check,axis=1)

data2["PAYMENT_INFO"].value_counts()

#* 240 users paid full dues
#* the rest 8710 users only paid minimum dues of there card bill
#* i.e 97 % user only pay min payment of there card bill

##### Joint two way Table of Payment Info and Purchase Type

pd.crosstab(data2["PURCHASE_TYPE"],data2["PAYMENT_INFO"])

sns.barplot(data=data2,x="PURCHASE_TYPE",y="PURCHASES", hue="PAYMENT_INFO")
plt.title("Total Of Purchases By Diffrent Purchase Type")
plt.show()

sns.barplot(data=data2,x="PURCHASE_TYPE",y="CREDIT_UTLIZATION_RATIO", hue="PAYMENT_INFO")
plt.title("Credit Utlization By Diffrent Purchase Types")
plt.show()

#### Removing Outlier

# using box plot
data3 = data2.copy()

cnames=list(data3.columns)
for i in cnames:
    if isinstance(data3[i].iloc[1] , float) or isinstance(data3[i].iloc[1] , int) :
        print(i)
        q75, q25 = np.percentile(data3.loc[:,i], [75 ,25])
        iqr = q75 - q25

        min = q25 - (iqr*1.5)
        max = q75 + (iqr*1.5)
        print(min)
        print(max)

        data3 = data3.drop(data3[data3.loc[:,i] < min].index)
        data3 = data3.drop(data3[data3.loc[:,i] > max].index)
data3.shape
# creating the list of our purchase type
label = list(data3.PURCHASE_TYPE )

### Scaling Variables
#Prior to applying a clustering algorithm we need to recentre the scales of our variables
#* First we will check convert all the objects into dummy variable.
#* Second we will check for normality of data using jarque-bera test so that we can apply a suitable scaling technique.

##### Converting into Dummy Variables

col = list(data3.columns)
num = []
obj = []
for i in col:
    if isinstance(data3[i].iloc[1] , float) or isinstance(data3[i].iloc[1] , int) :
        num.append(i)
    else :
        obj.append(i)

PURCHASE_TYPE=pd.get_dummies(data3[obj[1]])
PAYMENT_INFO=pd.get_dummies(data3[obj[2]])
data3=pd.concat([data3[num],PURCHASE_TYPE,PAYMENT_INFO],axis=1)

data3.head(2)

##### Applying jarque_bera test on all the numerical variables of data frame
#at alpha value of 0.05
#* Null hypothesis: data comes from a normal distribution.
#* Alternate hypothesis: data doesnot comes from a normal distribution.

from scipy import stats
alpha = 0.05
col = list(data3.columns)
f=1
for i in col:
     if isinstance(data3[i].iloc[1] , float) or isinstance(data3[i].iloc[1] , int) :
        print(i)
        stat,p = stats.jarque_bera(data3[i])
        print(stat, p)
        if p > alpha:
            print('Sample looks normal (fail to reject H0)')
        else:
            print('Sample does not look normal (reject H0)')

#Since p-value is less than our alpha which shows strong evidence against our Null hypothesis
#* we will Reject the null hypothesis stating that the following columns the dataframe are not normal

#### Since are data is not normal we will scale the variable using min max normalization technique

#importing minmax scaler
from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
#scaler = preprocessing.StandardScaler()

# converting our data frame into matrix
data3 = data3.as_matrix().astype(np.float)
data3 =np.nan_to_num(data3)

# applying scaler on our data and coverting i into a data frame
data3_minmax = pd.DataFrame((scaler.fit_transform(data3)))


# renaming columns
data3_minmax.columns = col

data3_minmax.head()

### Feature Selection using  a Technique PCA which is a Derivative Of Factor Analysis

# Importing PCA
from sklearn.decomposition import PCA

#We have 29 features so our n_component will be 29.
pc=PCA(n_components=29)
cr_pca=pc.fit(data3_minmax)

# Sum of variance explained by all components
sum(cr_pca.explained_variance_ratio_)

# calculating optimal no. of compnents
cumm_var={}
for n in range(2,29):
    pc=PCA(n_components=n)
    cr_pca=pc.fit(data3_minmax)
    cumm_var[n]=sum(cr_pca.explained_variance_ratio_)

cumm_var

##### Creating a Skree Plot

com = list(cumm_var.keys())
variance = list(cumm_var.values())

plt.figure(figsize=(10,5))
plt.plot(com,variance, marker ="o")

plt.xticks(range( 0, 30 ))

plt.title('Scree Plot')
plt.xlabel('Factors')
plt.ylabel('Variance')

plt.grid()
plt.show()

#### From the Above chart we can see after the 14th Component the Variance tends to Increase very slowly.
#So, 14 will be are optimal no. of component which explains around 97% variance

# applying and reducing our data set with 14 component
pc_final=PCA(n_components=14).fit(data3_minmax)
reduced_cr=pc_final.fit_transform(data3_minmax)

d_clust = pd.DataFrame(reduced_cr)

d_clust.info()

### Applying Clustering algorithm

##### Kmeans

from sklearn.cluster import KMeans

#Estimate optimum number of clusters
cluster_range = range( 1, 20 )
cluster_errors = []

for num_clusters in cluster_range:
    clusters = KMeans(num_clusters).fit(d_clust)
    cluster_errors.append(clusters.inertia_)

#Create dataframe with cluster errors
clusters_df = pd.DataFrame( { "num_clusters":cluster_range, "cluster_errors": cluster_errors } )

plt.figure(figsize=(10,5))
plt.plot(clusters_df.num_clusters, clusters_df.cluster_errors,marker = "o")#", data=clusters_df)
plt.xlim(1,20)
plt.xticks(cluster_range)
plt.title('Finding Optimal Cluster By Elbow Method')
plt.xlabel('Inertia')
plt.ylabel('No. Of Cluster')

plt.show()

Kmeans = KMeans(4).fit(d_clust)

# Predicting labels usind our model
clust_label = list(Kmeans.predict(d_clust))
len(clust_label)

data3

# converting it to form a data frame
conf = pd.DataFrame({"cluster":clust_label,"label":label})

# creating a Two way table to evaluate our cluuster
pd.crosstab(conf.cluster,conf.label)

##### Output
#* Almost two third of the records were outliers and were removed during the process.
#* It was necessary because Kmeans is affected by outliers and extreem values
#* 4 is the optimum no. of cluster

##### Kmedoids
#* Since kmeans removing outliers was a neccesity but it came with a cost of loss of almost half of the information in order to prevent the loss of data we need to use a method that is robust to outliers and extreem values and its called k medoids well medoids has almost same math but instead of estimating centroids using mean it takes the closest observation and use it to define the cluster which is called as a medoid.

data4 = data2.copy()


col = list(data4.columns)
num = []
obj = []
for i in col:
    if isinstance(data4[i].iloc[1] , float) or isinstance(data4[i].iloc[1] , int) :
        num.append(i)
    else :
        obj.append(i)
obj

PURCHASE_TYPE=pd.get_dummies(data4[obj[1]])
PAYMENT_INFO=pd.get_dummies(data4[obj[2]])
data4=pd.concat([data4[num],PURCHASE_TYPE,PAYMENT_INFO],axis=1)

data4.head(2)

##### Applying jarque_bera test on all the numerical variables of data frame
#at alpha value of 0.05
#* Null hypothesis: data comes from a normal distribution.
#* Alternate hypothesis: data doesnot comes from a normal distribution.

from scipy import stats
alpha = 0.05
col = list(data4.columns)
f=1
for i in col:
     if isinstance(data4[i].iloc[1] , float) or isinstance(data4[i].iloc[1] , int) :
        print(i)
        stat,p = stats.jarque_bera(data4[i])
        print(stat, p)
        if p > alpha:
            print('Sample looks normal (fail to reject H0)')
        else:
            print('Sample does not look normal (reject H0)')

#### Again
#Since p-value is less than our alpha which shows strong evidence against our Null hypothesis
#* we will Reject the null hypothesis stating that the following columns the dataframe are not normal

#### Since are data is not normal we will scale the variable using min max normalization technique

#importing minmax scaler
from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
#scaler = preprocessing.StandardScaler()

# converting our data frame into matrix
data4 = data4.as_matrix().astype(np.float)
data4 =np.nan_to_num(data4)

# applying scaler on our data and coverting i into a data frame
data4_minmax = pd.DataFrame((scaler.fit_transform(data4)))


# renaming columns
data4_minmax.columns = col

data4_minmax.head()

### Feature Selection using  a Technique PCA which is a Derivative Of Factor Analysis

# Importing PCA
from sklearn.decomposition import PCA

#We have 29 features so our n_component will be 29.
pc=PCA(n_components=29)
cr_pca=pc.fit(data4_minmax)

# Sum of variance explained by all components
sum(cr_pca.explained_variance_ratio_)

# calculating optimal no. of compnents
cumm_var={}
for n in range(2,29):
    pc=PCA(n_components=n)
    cr_pca=pc.fit(data4_minmax)
    cumm_var[n]=sum(cr_pca.explained_variance_ratio_)

cumm_var

##### Creating a Skree Plot

com = list(cumm_var.keys())
variance = list(cumm_var.values())

plt.figure(figsize=(10,5))
plt.plot(com,variance, marker ="o")

plt.xticks(range( 0, 30 ))

plt.title('Scree Plot')
plt.xlabel('Factors')
plt.ylabel('Variance')

plt.grid()
plt.show()

#### From the Above chart we can see after the 10th Component the Variance tends to Increase very slowly.
#So, 10 will be are optimal no. of component which explains around 97% variance

# applying and reducing our data set with 10 component
pc_final=PCA(n_components=14).fit(data4_minmax)
reduced_cr=pc_final.fit_transform(data4_minmax)

d_clust1 = pd.DataFrame(reduced_cr)

from sklearn_extra.cluster import KMedoids
#Estimate optimum number of clusters
cluster_range = range( 1, 20 )
cluster_errors = []

for num_clusters in cluster_range:
    clusters = KMedoids(num_clusters).fit(d_clust1)
    cluster_errors.append(clusters.inertia_)

#Create dataframe with cluster errors
clusters_df = pd.DataFrame( { "num_clusters":cluster_range, "cluster_errors": cluster_errors } )

plt.figure(figsize=(10,5))
plt.plot(clusters_df.num_clusters, clusters_df.cluster_errors,marker = "o")#", data=clusters_df)
plt.xlim(1,20)
plt.xticks(cluster_range)
plt.title('Finding Optimal Cluster By Elbow Method')
plt.xlabel('Inertia')
plt.ylabel('No. Of Cluster')
plt.show()

#### The optimum no. of clusters using K-medoids is 4

#now lets construct our model Using Kmedoids and then evaluate  it using our earlier created KPI PURCHASE_TYPE

#Creating cluster using Kmedoids
clust_final = KMeans(4).fit(d_clust1)

# Predicting labels usind our model
clust_label = list(clust_final.predict(d_clust1))
len(clust_label)

# creating the list of our purchase type
label = list(data2.PURCHASE_TYPE )

# converting it to form a data frame
conf = pd.DataFrame({"cluster":clust_label,"label":label})

# creating a Two way table to evaluate our cluuster
pd.crosstab(conf.cluster,conf.label)

### Output
#* On Using K-medoids we got the same cluster and there was no loss of information which was involved while using kmeans.
