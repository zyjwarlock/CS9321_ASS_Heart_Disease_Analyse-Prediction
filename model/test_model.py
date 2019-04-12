import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from  sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
import csv



h = .02  # step size in the mesh

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Decision Tree",
         "Random Forest", "AdaBoost", "Naive Bayes",  "LDA", "QDA"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=19, max_features='sqrt'),
    AdaBoostClassifier(),
    GaussianNB(),
    LDA(),
    QDA()]


data=[]
disease_feature=[]
disease_target=[]

csv_file = csv.reader(open('data/cleaned_data.csv'))
next(csv_file)
for content in csv_file:
    content=list(map(float,content))
    if len(content)!=0:
        data.append(content[1:])
        disease_feature.append(content[1:14])
        disease_target.append(content[-1])
print('data=',data)
print('traffic_feature=',disease_feature)
print('traffic_target=',disease_target)
# scaler = StandardScaler() # 标准化转换
# scaler.fit(disease_feature)  # 训练标准化对象
# disease_feature= scaler.transform(disease_feature)   # 转换数据集
#feature_train, feature_test, target_train, target_test = train_test_split(disease_feature, disease_target, test_size=0.3,random_state=0)
#figure = plt.figure(figsize=(27, 9))

X = np.array(StandardScaler().fit_transform(disease_feature))
y = np.array(disease_target)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4)

# i=1
#
# x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
# y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
#                      np.arange(y_min, y_max, h))

# # just plot the dataset first
# cm = plt.cm.RdBu
# cm_bright = ListedColormap(['#FF0000', '#0000FF'])
# ax = plt.subplot(1, len(classifiers) + 1, i)
# # Plot the training points
# ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
# # and testing points
# ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6)
# ax.set_xlim(xx.min(), xx.max())
# ax.set_ylim(yy.min(), yy.max())
# ax.set_xticks(())
# ax.set_yticks(())
# i += 1
rf = RandomForestClassifier(max_depth=5, n_estimators=19, max_features='sqrt')
rf.fit(X_train, y_train)
score = rf.score(X_test, y_test)
print('rf', score)
# iterate over classifiers
for name, clf in zip(names, classifiers):
    # ax = plt.subplot(1, len(classifiers) + 1, i)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print(name, score)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    # if hasattr(clf, "decision_function"):
    #     Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    # else:
    #     Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    #
    # # Put the result into a color plot
    # Z = Z.reshape(xx.shape)
    # ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)
    #
    # # Plot also the training points
    # ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
    # # and testing points
    # ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
    #            alpha=0.6)
    #
    # ax.set_xlim(xx.min(), xx.max())
    # ax.set_ylim(yy.min(), yy.max())
    # ax.set_xticks(())
    # ax.set_yticks(())
    # ax.set_title(name)
    # ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
    #         size=15, horizontalalignment='right')
    # i += 1

# figure.subplots_adjust(left=.02, right=.98)
#
#
# plt.show()

