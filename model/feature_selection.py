from sklearn.ensemble import RandomForestClassifier
import csv
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import numpy as  np
from collections import defaultdict

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
        disease_feature.append(content[1:14])
        disease_target.append(content[-1])

X = np.array(StandardScaler().fit_transform(disease_feature))
y = np.array(disease_target)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)

clf = RandomForestClassifier(max_depth=7, n_estimators=19, min_samples_leaf=7, max_features='sqrt')
# clf.fit(X_train,y_train)
# print(clf.score(X_test, y_test))
#
# predict_results=clf.predict(X_test)
#
# print(accuracy_score(predict_results, y_test))
# print(clf.score(X_test, y_test))
#
# print ("Features sorted by their score:")
# sorted_importance = sorted(zip(map(lambda x: round(x, 4), clf.feature_importances_), title), reverse=True)
#
# for i, feature in enumerate(sorted_importance):
#     print (i+1, feature)



scores = defaultdict(list)#crossvalidate the scores on a number of different random splits of the data
time=0
for train_idx, test_idx in ShuffleSplit(n_splits=50, test_size=0.7).split(X):
    time+=1
    X_train, X_test = X[train_idx], X[test_idx]
    Y_train, Y_test = y[train_idx], y[test_idx]
    r = clf.fit(X_train, Y_train)
    acc = r2_score(Y_test, clf.predict(X_test))
    for i in range(X.shape[1]):
        X_t = X_test.copy()
        np.random.shuffle(X_t[:, i])
        shuff_acc = r2_score(Y_test, clf.predict(X_t))
        scores[title[i]].append((acc-shuff_acc)/acc)
    if(time%10 == 0):
        print ("splittimes: ",time," Features sorted by their score:")
        sorted_importance = sorted([(round(np.mean(score), 4), feat) for
                feat, score in scores.items()], reverse=True)

        for i, feature in enumerate(sorted_importance):
            print (i+1, feature)




