import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

df=pd.read_csv("car.csv")

#CLEANING--->
print(df.isna().sum())

df=df.dropna()

df=df.reset_index(drop=True)


#analyze each and every wrt dependent


#1SEATS--->

print(df['seats'].value_counts())



#TORQUE--->

print(df['torque'].dtype)

df=df.drop(columns="torque")
#beacuse of unstructure patterns we have to drop the columns


#MAX POWER---->
print(df['max_power'].dtype)
df['max_power']=df['max_power'].apply(lambda x:x.split()[0])
#df['max_power']=df['max_power'].astype('float32')
#above line give error find the noise value
l=[]
for i in range(len(df)):
    try:
        float(df.iloc[i,-2])
    except:
        l.append(i)
        
df=df.drop(index=l).reset_index(drop=True)        

df['max_power']=df['max_power'].astype('float32')


#engines--->
df['engine']=df['engine'].apply(lambda x:x.split()[0])
l=[]
for i in range(len(df)):
    try:
        float(df.iloc[i,-3])
    except:
        l.append(i)
        
df=df.drop(index=l).reset_index(drop=True)         
df['engine']=df['engine'].astype('float32')



#MILEAGE---->
df['mileage']=df['mileage'].apply(lambda x:x.split()[0]).astype('float32')




#OWNER--->

print(df['owner'].value_counts())

df['owner']=df['owner'].replace({'Fifth':'Fourth & Above Owner'})

print(df['owner'].value_counts())

f=df['owner']=='Test Drive Car'

df=df.drop(index=df[f].index).reset_index(drop=True)

#reason od dropping-->test drive cars were considered as outliers in our data
#thats why we drop it


#transmission--->
print(df['transmission'].value_counts())

print(df['transmission'].dtype)


#SELLER---->
print(df['seller_type'].value_counts())

for x in df['seller_type'].unique():
    f=df['seller_type']==x
    plt.violinplot(df.loc[f,'selling_price'])
    plt.title(x)
    plt.ticklabel_format(style='plain')
    plt.show()



#FUEL
print(df['fuel'].value_counts())

for x in df['fuel'].unique():
    f=df['fuel']==x#filter data of particular fuel
    plt.violinplot(df.loc[f,'selling_price'])
    plt.title(x)
    plt.ticklabel_format(style='plain')
    plt.show()


#CONCLUSION--->there is similar distribution of cng and lpg
#petrol and diessel so merege these categories


df['fuel']=df['fuel'].replace({'CNG':0,'LPG':0,'Petrol':1,'Diesel':1})



#NAME---->
print(df['name'].value_counts())

df['name']=df['name'].apply(lambda x:x.split()[0])

brands=df['name'].unique()


#when number of categories is high than make 
#group of categories according to dependent variable

brand_selling_price=df.groupby('name')['selling_price'].mean()

brand_selling_price=brand_selling_price.sort_values(ascending=False)


def fxn(x):
    if x in brand_selling_price.iloc[:10]:
        return 0
    elif x in brand_selling_price.iloc[10:25]:
        return 1
    
    else:
        return 2
    
       
df['name']=df['name'].apply(fxn)






#STATISTICAL ANALYSIS OF COLUMNS---->

categorical=df[['name','fuel','seller_type','transmission','owner','seats']]
numeric=df[['year','km_driven','mileage','engine','max_power','selling_price']]


from sklearn.preprocessing import LabelEncoder
encoder1=LabelEncoder()
categorical['owner']=encoder1.fit_transform(categorical['owner'])



encoder2=LabelEncoder()
categorical['transmission']=encoder2.fit_transform(categorical['transmission'])



from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

onehotencode=OneHotEncoder(drop='first')
ct=ColumnTransformer([('encode',onehotencode,[2])],remainder='passthrough')
categorical=ct.fit_transform(categorical)


categorical=pd.DataFrame(categorical)



#Feature selection 

#1)method --->pearson correlation method

#in this method we find the corelation coeffiecient

corr=numeric.corr()

#to represent the coreltion matrix in graphic way use heatmap
import seaborn
seaborn.heatmap(corr)
plt.show()



#anova test
#analysis of variance

#anova is used to find out whether categories have same variance wrt
#other numeric (dependent) column  or not


#in hypothesis we assume two assumptions(cases)
#H0--->(null hypothesis)--->all categories have same variance
#H1(alternative hypothesis)--->variance of all categories is dirfferent



#steps to perfomr anova
#make assumption(h0 and h1)
#calcualte within sum of square and between um of square
#calculate degree of freedom of groups and each category
#find f values
#take alpha value and confident interval and find in graph wthere to
#accept or reject nul hypothesis




from sklearn.feature_selection import SelectKBest,f_classif
#select k best is used to sort the best columns accoring to test
#value passed as parameter

sk=SelectKBest(f_classif,k=7)
#k value represnt the number of columns we want
#this value depends upon our domain knowledge

result=sk.fit_transform(categorical,numeric['selling_price'])

print(sk.scores_)





#OUTLIER DETECTION--->

plt.scatter(numeric['km_driven'],numeric['selling_price'])
plt.show()


#plt graph--->
for x in numeric.columns:
    plt.hist(numeric[x])
    plt.title(x)
    plt.show()

def z_score(column):
    mean=column.mean()
    std=column.std()
    z=np.abs((column-mean)/std)
    return column[z>3]


outliers1=z_score(numeric['km_driven'])

outliers2=z_score(numeric['max_power'])
outliers3=z_score(numeric['mileage'])

print(len(outliers1)+len(outliers2)+len(outliers3))


f=~numeric['km_driven'].isin(z_score(numeric['km_driven']))
numeric=numeric[f]
categorical=categorical[f]
f=~numeric['max_power'].isin(z_score(numeric['max_power']))
numeric=numeric[f]
categorical=categorical[f]
f=~numeric['mileage'].isin(z_score(numeric['mileage']))
numeric=numeric[f]
categorical=categorical[f]



'''
DBSCAN
'''
#Outlier detection,fraud detection,clustering

engine=numeric[['engine','selling_price']]# one independent and other dependent selling price independent
#engine dependent


#min max scaling
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
engine=scaler.fit_transform(engine)



from sklearn.neighbors import NearestNeighbors
neighbor=NearestNeighbors(n_neighbors=10)
neighbor.fit(engine)

#step 1--finding nearest neighbours
distance,index=neighbor.kneighbors(engine)


distance=np.sort(distance,axis=0)

distance=distance[:,1]


plt.plot(distance)
plt.title("K-distance graph")
plt.show()


from sklearn.cluster import DBSCAN
dbscan=DBSCAN(eps=0.05,min_samples=10)              
model=dbscan.fit(engine)


points=model.labels_
plt.scatter(numeric['engine'],numeric['selling_price'],c=points)
plt.show()


numeric=numeric[points!=-1]
numeric.reset_index(drop=True)

categorical=categorical[points!=-1]



df1=pd.concat((categorical,numeric),axis=1,ignore_index=True)




'''
scaling
'''


X=df1.iloc[:,:-1]#:-1 means last one exculde
y=df1.iloc[:,-1]#-1 last one exits

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X=sc.fit_transform(X)


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)






'''
random forest regressor
'''




from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor()
regressor.fit(X_train,y_train)
y_pred=regressor.predict(X_test)
print(y_pred)



from sklearn.metrics import mean_absolute_error as mae,r2_score
print('Mean absolute error:',mae(y_test,y_pred))
print('r2 score', r2_score(y_test,y_pred))






