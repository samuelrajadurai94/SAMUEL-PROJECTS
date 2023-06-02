from sklearn import datasets
iris = datasets.load_iris()
import pandas as pd
df=pd.DataFrame(iris.data,columns=iris.feature_names)
print(df)
df['flower'] = iris.target
print(df)
df['flower'] = df['flower'].apply (lambda x:iris.target_names[x])
print(df)

from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
cr =cross_val_score(SVC(kernel='linear',C=10,gamma = 'auto'),iris.data,iris.target,cv=5)
print(cr)
cr2 =cross_val_score(SVC(kernel='rbf',C=10,gamma = 'auto'),iris.data,iris.target,cv=5)
print(cr2)

from sklearn.model_selection import GridSearchCV
clf =GridSearchCV(SVC(gamma = 'auto'),
                  {'C':[1,10,20],
                   'kernel':['rbf','linear']},
                  cv=5,return_train_score=False)
clf.fit(iris.data,iris.target)
df2=pd.DataFrame(clf.cv_results_)
print(df2.columns)
print(df2[['param_C','param_kernel','mean_test_score']])

from sklearn.model_selection import RandomizedSearchCV
clf =RandomizedSearchCV(SVC(gamma = 'auto'),
                  {'C':[1,10,20],
                   'kernel':['rbf','linear']},
                  cv=5,return_train_score=False,
                    n_iter=2)      









