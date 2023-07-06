import seaborn as sn
import matplotlib.pyplot as plt
import  pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.preprocessing import LabelEncoder
data=pd.read_csv('CVD_cleaned.csv')
print(data.describe())
print(data.isna().sum())
print(data.info())
print(data.columns)
'''for i in data.select_dtypes(include='number').columns.values:
    sn.boxplot(data[i])
    plt.title(f"The {i} related data ")
    plt.show()'''


data['z_scores']=(data['BMI']-data['BMI'].mean())/(data['BMI'].std())
df=data[(data['z_scores']>-3)&(data['z_scores']<3)]
q1=df.BMI.quantile(0.25)
q3=df.BMI.quantile(0.75)
iqr=q3-q1
up=q3+1.5*iqr
lo=q1-1.5*iqr
df=df[(df['BMI']>lo)&(df['BMI']<up)]

qu1=df.BMI.quantile(0.25)
qu3=df.BMI.quantile(0.75)
Iqr=qu3-qu1
upp=qu3+1.5*Iqr
low=qu1-1.5*Iqr
df=df[(df['BMI']>low)&(df['BMI']<upp)]


qua1=df.Alcohol_Consumption.quantile(0.25)
qua3=df.Alcohol_Consumption.quantile(0.75)
IqR=qua3-qua1
uppe=qua3+1.5*IqR
lowe=qua1-1.5*IqR
df=df[(df.Alcohol_Consumption>lowe)&(df.Alcohol_Consumption<uppe)]

quan1=df.Alcohol_Consumption.quantile(0.25)
quan3=df.Alcohol_Consumption.quantile(0.75)
IQR=quan3-quan1
upper=quan3+1.5*IQR
lower=quan1-1.5*IQR
df=df[(df.Alcohol_Consumption>lower)&(df.Alcohol_Consumption<upper)]

quantile1=df.Fruit_Consumption.quantile(0.25)
quantile3=df.Fruit_Consumption.quantile(0.75)
iQr=quantile3-quantile1
Upp=quantile3+1.5*iqr
Low=quantile1-1.5*iqr
df=df[(df.Fruit_Consumption >Low)&(df.Fruit_Consumption < Upp)]

print(df.shape)
print(data.shape)

threshold=2
for i in df.select_dtypes(include="number").columns.values:
    lower=df[i].mean()-threshold*df[i].std()
    upper=df[i].mean()+threshold*df[i].std()
    df=df[(df[i]>lower)&(df[i]<upper)]

print(df.shape)
print(df.shape)


'''plt.figure(figsize=(17, 6))
corr = df.corr(method='spearman')
my_m = np.triu(corr)
sn.heatmap(corr, mask=my_m, annot=True, cmap="Set2")
plt.show()

for i in df.select_dtypes(include='number').columns.values:
    for j in df.select_dtypes(include='number').columns.values:
        plt.plot(df[i], marker='o', label=f"{i}", color='red')
        plt.plot(df[j], marker='x', label=f"{j}", color="blue")
        plt.title(f"ITS {i} vs {j}")
        plt.legend()
        plt.show()

for i in df.select_dtypes(include='number').columns.values:
    for j in df.select_dtypes(include='number').columns.values:
        plt.scatter(df[i], df[j], marker='o', color='red')
        plt.scatter(df[i], df[j], marker='x', color='blue')
        plt.xlabel(f'{i}')
        plt.ylabel(f'{j}')
        plt.title(f"ITS {i} vs {j}")
        plt.legend()
        plt.show()'''

'''sn.pairplot(df)
plt.show()'''

'''for i in df.select_dtypes(include='number').columns.values:
    for j in df.select_dtypes(include='number').columns.values:
        sn.histplot(df[i], label=f"{i}", color='red')
        sn.histplot(df[j], label=f"{j}", color="blue")
        plt.title(f"ITS {i} vs {j}")
        plt.legend()
        plt.show()

for i in df.select_dtypes(include='number').columns.values:
    for j in df.select_dtypes(include='number').columns.values:
        sn.distplot(df[i], label=f"{i}", color='red')
        sn.distplot(df[j], label=f"{j}", color="blue")
        plt.title(f"ITS {i} vs {j}")
        plt.legend()
        plt.show()'''''
for i in df.select_dtypes(include='object').columns.values:
    print(df[i].value_counts())
lab=LabelEncoder()
df['dia']=lab.fit_transform(df['Diabetes'])
df['art']=lab.fit_transform(df['Arthritis'])
df['sex']=lab.fit_transform(df['Sex'])
df['smoke']=lab.fit_transform(df['Smoking_History'])
df['health']=lab.fit_transform(df['General_Health'])
print(df.select_dtypes(include='number').columns.values)

x=df[['Height_(cm)', 'Weight_(kg)' ,'BMI', 'Alcohol_Consumption'
 ,'Fruit_Consumption', 'Green_Vegetables_Consumption'
 ,'FriedPotato_Consumption','health','dia','art','smoke']]

y=df['sex']


x_train,x_test,y_train,y_test=train_test_split(x,y)
print(y_train)

lr=LogisticRegression(max_iter=200)
lr.fit(x_train,y_train)
print('The logistic regression: ',lr.score(x_test,y_test))

xgb=XGBClassifier()
xgb.fit(x_train,y_train)
print("the Xgb : ",xgb.score(x_test,y_test))

lgb=LGBMClassifier()
lgb.fit(x_train,y_train)
print('The LGB',lgb.score(x_test,y_test))

tree=DecisionTreeClassifier(criterion='entropy',max_depth=1)
tree.fit(x_train,y_train)
print('Dtree ',tree.score(x_test,y_test))

rforest=RandomForestClassifier(criterion='entropy')
rforest.fit(x_train,y_train)
print('The random forest: ',rforest.score(x_test,y_test))

adb=AdaBoostClassifier()
adb.fit(x_train,y_train)
print('the adb ',adb.score(x_test,y_test))

grb=GradientBoostingClassifier()
grb.fit(x_train,y_train)
print('Gradient boosting ',grb.score(x_test,y_test))

bag=BaggingClassifier()
bag.fit(x_train,y_train)
print('Bagging',bag.score(x_test,y_test))