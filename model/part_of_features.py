from sklearn.ensemble import RandomForestClassifier
import csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import numpy as  np

from sklearn.model_selection import GridSearchCV

from sklearn import metrics
data=[]
disease_feature=[]
disease_target=[]

csv_file = csv.reader(open('data/cleaned_data.csv'))
title = next(csv_file)[1:]
for content in csv_file:
    content=list(map(float,content))
    if len(content)!=0:
        data.append(content[1:])
        _list = content[1:4] + content[8:14]
        disease_feature.append(_list)
        disease_target.append(content[-1])
print('data=',data)
print('traffic_feature=',disease_feature)
print('traffic_target=',disease_target)
# scaler = StandardScaler() # 标准化转换
# scaler.fit(disease_feature)  # 训练标准化对象
# disease_feature= scaler.transform(disease_feature)   # 转换数据集
# feature_train, feature_test, target_train, target_test = train_test_split(disease_feature, disease_target, test_size=0.3,random_state=0)

X = np.array(StandardScaler().fit_transform(disease_feature))
y = np.array(disease_target)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)

clf = RandomForestClassifier(n_estimators= 20,
                             max_depth=7,
                             min_samples_leaf=7,
                             oob_score=True)
clf.fit(X_train,y_train)
print(clf.score(X_test, y_test))

predict_results=clf.predict(X_test)

print(accuracy_score(predict_results, y_test))
print(clf.oob_score_)
# importances = clf.feature_importances_
# indices = np.argsort(importances)[::-1]
# for f in range(X_train.shape[1]):
#     print("%2d) %-*s %f" % (f + 1, 30, title[indices[f]], importances[indices[f]]))
#


# 0.8646616541353384
# 0.9032258064516129


# print (clf.oob_score_)
# y_predprob = clf.predict_proba(disease_feature)[:,1]
# print ("AUC Score (Train): %f" % metrics.roc_auc_score(disease_target,y_predprob))


# param_test1= {'n_estimators':range(20, 40, 2)}
# gsearch1= GridSearchCV(estimator = RandomForestClassifier(max_depth=7, min_samples_leaf=8, oob_score=True,  max_features=8, random_state=10),
#                        param_grid =param_test1, scoring='roc_auc',cv=5)
# gsearch1.fit(X_train,y_train)
# print(gsearch1.param_grid,gsearch1.best_params_, gsearch1.best_score_)
# {'n_estimators': range(51, 71)} {'n_estimators': 58} 0.9472686733556299

# param_test2= {'max_depth':range(2,8,1), 'min_samples_split':range(2,20,1)}
# gsearch2= GridSearchCV(estimator = RandomForestClassifier(n_estimators= 20,
#                                  max_features='sqrt' ,oob_score=True,random_state=10),
#    param_grid = param_test2,scoring='roc_auc',iid=False, cv=5)
# gsearch2.fit(disease_feature,disease_target)
# print(gsearch2.best_params_, gsearch2.best_score_)
# {'max_depth': 7, 'min_samples_split': 2} 0.9588894110275689
# {'max_depth': 7, 'min_samples_split': 7} 0.956234335839599

# param_test3= {'min_samples_split':range(2,40,2), 'min_samples_leaf':range(2,20,2)}
# gsearch3= GridSearchCV(estimator = RandomForestClassifier(n_estimators= 20,max_depth=7,
#                                  max_features='sqrt' ,oob_score=True, random_state=10),
#    param_grid = param_test3,scoring='roc_auc',iid=False, cv=5)
# gsearch3.fit(disease_feature,disease_target)
# print(gsearch3.best_params_, gsearch3.best_score_)
# {'min_samples_leaf': 8, 'min_samples_split': 18} 0.9492481203007518


# param_test4= {'max_features':range(1,8,1)}
# gsearch4= GridSearchCV(estimator = RandomForestClassifier(n_estimators= 20,max_depth=7, min_samples_leaf=7, oob_score=True,  max_features=8),
#    param_grid = param_test4,  scoring='roc_auc',iid=False, cv=5)
# gsearch4.fit(disease_feature,disease_target)
# print(gsearch4.best_params_, gsearch4.best_score_  )







