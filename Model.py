# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 21:15:20 2020

@author: SHANMUKH CHAVA
"""
import pandas

from sklearn import linear_model,preprocessing,svm,neighbors
import sklearn
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression,LinearRegression
import pickle

data= pd.read_csv("train.csv")
data.Loan_Status=data.Loan_Status.map({'Y':1,'N':0})
data.drop("Loan_ID",axis=1,inplace=True)
data.head()
#data.apply(lambda x: sum(x.isnull()),axis=0) checking missing values in each column of train dataset
#preprocessing
data.Gender = data.Gender.fillna('Male')
data.Married = data.Married.fillna('Yes')
data.Dependents = data.Dependents.fillna('0')
data.Self_Employed = data.Self_Employed.fillna('No')
data.LoanAmount = data.LoanAmount.fillna(data.LoanAmount.mean())
data.Loan_Amount_Term = data.Loan_Amount_Term.fillna(360.0)
data.Credit_History = data.Credit_History.fillna(1.0)
#print(data.apply(lambda x: sum(x.isnull()),axis=0))
data.Gender=data.Gender.map({'Male':1,'Female':0})
data.Married=data.Married.map({'Yes':1,'No':0})
data.Dependents=data.Dependents.map({'0':0,'1':1,'2':2,'3+':3})
data.Education=data.Education.map({'Graduate':1,'Not Graduate':0})
data.Self_Employed=data.Self_Employed.map({'Yes':1,'No':0})
data.Property_Area=data.Property_Area.map({'Urban':2,'Rural':0,'Semiurban':1})
X=data.drop(['Loan_Status'],1)
y=data.Loan_Status
X_train,X_test,y_train,y_test=sklearn.model_selection.train_test_split(X,y,test_size=0.1)

clf=LogisticRegression()
clf.fit(X_train,y_train)
acc=clf.score(X_test,y_test)
print(acc)
"""
with open('loan.pickle','wb') as f:
    pickle.dump(clf,f)
"""
    
test=pd.read_csv('test.csv')
Loan_ID=test.Loan_ID
test.Gender = test.Gender.fillna('Male')
test.Married = test.Married.fillna('Yes')
test.Dependents = test.Dependents.fillna('0')
test.Self_Employed = test.Self_Employed.fillna('No')
test.LoanAmount = test.LoanAmount.fillna(data.LoanAmount.mean())
test.Loan_Amount_Term = test.Loan_Amount_Term.fillna(360.0)
test.Credit_History = test.Credit_History.fillna(1.0)

test.drop('Loan_ID',axis=1,inplace=True)
test.Gender=test.Gender.map({'Male':1,'Female':0})
test.Married=test.Married.map({'Yes':1,'No':0})
test.Dependents=test.Dependents.map({'0':0,'1':1,'2':2,'3+':3})
test.Education=test.Education.map({'Graduate':1,'Not Graduate':0})
test.Self_Employed=test.Self_Employed.map({'Yes':1,'No':0})
test.Property_Area=test.Property_Area.map({'Urban':2,'Rural':0,'Semiurban':1})

op=clf.predict(test)
print(op)

opy = pd.DataFrame()
opy['Loan_ID']=Loan_ID
opy['Loan_Status']=op
opy[['Loan_ID','Loan_Status']].to_csv('output3.csv',index=False)




