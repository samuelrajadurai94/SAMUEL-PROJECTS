import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
dk = pd.read_csv('D:\\Downloads\\loan_prediction.csv')
print(dk.head())
print(dk.shape)
print(dk.info())

#To check null values
print(dk.isnull().sum() * 100/len(dk))
print(dk.groupby('Self_Employed')['Self_Employed'].count())
print(dk.groupby('Credit_History')['Credit_History'].count())
#To reomive loanid column which is unnecessory
dk = dk.drop('Loan_ID',axis = 1)
print(dk.head(1))

# To remove nullvalues in less than 5% nullpercentcolumns
nacolumns=['Gender','Married','Dependents','LoanAmount','Loan_Amount_Term']
dk=dk.dropna(subset=nacolumns)
print(dk.isnull().sum() * 100/len(dk))
#To fill null values with max frequency values
print(dk['Self_Employed'].mode())
dk['Self_Employed']=dk['Self_Employed'].fillna(dk['Self_Employed'].mode()[0])
print(dk.groupby('Credit_History')['Credit_History'].count())
dk['Credit_History']=dk['Credit_History'].fillna(dk['Credit_History'].mode()[0])
print(dk.isnull().sum() * 100/len(dk))

#check some numerical columns and change any symbols if present
print(dk.sample(5))
print(dk['Dependents'].unique())
dk['Dependents'] = dk['Dependents'].replace(to_replace='3+',value='4')
print(dk['Dependents'].unique())

print(dk['Property_Area'].unique())

#To convert categorical into numerical
from sklearn.preprocessing import LabelEncoder
lr = LabelEncoder()
dk['Property_Area'] = lr.fit_transform(dk['Property_Area'])
dk['Gender'] = lr.fit_transform(dk['Gender'])
dk['Married'] = lr.fit_transform(dk['Married'])
dk['Education'] = lr.fit_transform(dk['Education'])
dk['Self_Employed'] = lr.fit_transform(dk['Self_Employed'])
dk['Loan_Status'] = lr.fit_transform(dk['Loan_Status'])
print(dk.head(10))
print(dk.info())
print(dk[['ApplicantIncome','CoapplicantIncome']].describe())

# heatmap to see corelation b/w features.
plt.figure(figsize=(12,6))
sns.heatmap(dk.corr(),cmap='BrBG',fmt='.2f',linewidths=2,annot=True)
plt.show()

# Defining feature column and Target column
x = dk.drop('Loan_Status',axis=1)
y = dk['Loan_Status']
print(y)

# To give scaling to numbers
sccol = ['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term']
from sklearn.preprocessing import StandardScaler
st = StandardScaler()
x[sccol] = st.fit_transform(x[sccol])
print(x[sccol])

import warnings
warnings.filterwarnings('ignore')

#Model building
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import numpy as np

modelset ={}
def model_acscore(model,x,y):
    xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.20,random_state=42)
    model.fit(xtrain,ytrain)
    y_pred = model.predict(xtest)
    print(f'{model} accuracy is {accuracy_score(ytest,y_pred)}')

    score = cross_val_score(model,x,y,cv=5)
    print(f'{model} Avg crossvalscore is {np.mean(score)}')
    modelset[model] = round(np.mean(score)*100,2)

# Logistic Regression
from sklearn.linear_model import LogisticRegression
model1 = LogisticRegression()
model_acscore(model1,x,y)
print(modelset)

# Support vector machine
from sklearn import svm
model2 = svm.SVC()
model_acscore(model2,x,y)

# Decision Tree
from sklearn.tree import DecisionTreeClassifier
model3 = DecisionTreeClassifier()
model_acscore(model3,x,y)

# Random forest
from sklearn.ensemble import RandomForestClassifier
model4 = RandomForestClassifier()
model_acscore(model4,x,y)

#Gradient boosting
from sklearn.ensemble import GradientBoostingClassifier
model5 = GradientBoostingClassifier()
model_acscore(model5,x,y)

print(modelset)
print('****************************')

# Hyper parameter tuning for logistic reg
from sklearn.model_selection import RandomizedSearchCV
logr_para={'C':np.logspace(-4,4,20),
           'solver': ['liblinear','lbfgs','newton-cholesky']}
rs_logr = RandomizedSearchCV(LogisticRegression(),
                           param_distributions=logr_para,
                           n_iter=20,cv=5,verbose=True)
rs_logr.fit(x,y)
print(rs_logr.best_score_)
print(rs_logr.best_params_)
print('************************')

# Hyper parameter tuning for svm model
svm_para={'C':[0.25,0.50,0.75,1],
          'kernel':['linear','rbf']}
rs_svm = RandomizedSearchCV(svm.SVC(),param_distributions=svm_para,
                            cv=5,n_iter=20,verbose=True)
rs_svm.fit(x,y)
print(rs_svm.best_score_)
print(rs_svm.best_params_)
print('***********************')

 # Hyper parameter for Random forest classifier
rf_para = {'n_estimators':np.arange(10,1000,10),
           'max_features':['auto','sqrt'],
           'max_depth':[None,3,5,10,20,30],
           'min_samples_split':[2,5,20,50,100],
           'min_samples_leaf':[1,2,5,10]}

rs_rf = RandomizedSearchCV(RandomForestClassifier(),
                           param_distributions=rf_para,
                           cv=5,n_iter=20,verbose=True)
rs_rf.fit(x,y)
print(rs_rf.best_score_)
print(rs_rf.best_params_)
print('***********************')

# selecting Random forest as final best model and use its best parameters
# so we selected Random forest classifier is the best model for this datset
  # its accuracy score is 80.84
rf_final =RandomForestClassifier(n_estimators=270,
                           min_samples_split=5,
                           min_samples_leaf=5,
                           max_features='sqrt',
                           max_depth=5)
rf_final.fit(x,y)
model_acscore(rf_final,x,y)

# save this final model using joblib
import joblib
#joblib.dump(rf_final,'D:\\Downloads\\joblib_loan_predict')
model11=joblib.load('D:\\Downloads\\joblib_loan_predict')

dfpred = pd.DataFrame({
    'Gender':1,
    'Married':1,
    'Dependents':2,
    'Education':0,
    'Self_Employed':0,
    'ApplicantIncome':2889,
    'CoapplicantIncome':0.0,
    'LoanAmount':45,
    'Loan_Amount_Term':180,
    'Credit_History':0,
    'Property_Area':1},index=[0])
target=model11.predict(dfpred)
if target ==1:print('loan will be approved') 
else: print('loan can not be approved')

