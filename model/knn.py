import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def train_test():
    csv_file = 'new_data.csv'
    data = pd.read_csv(csv_file)
    # print(data)
    # X_train, y_train, X_test, y_test = split_data(dataframe, percent)
    X=data.drop(["num","slope","thalach"], axis=1)
    y=data['num']
    # print(X)
    # print(y)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
    knn = KNeighborsClassifier(n_neighbors=10)
    sum = 0
    for i in range(100):
        knn.fit(X_train.values, y_train.values)
        sum+=knn.score(X_test.values, y_test.values)
    print(sum/100)
    return knn




def prediction(sex,cp,exang,oldpeak,ca,thal,age_stage,clf):
    label=clf.predict([[sex,cp,exang,oldpeak,ca,thal,age_stage]])[0]
    return label

def train():
    csv_file = 'cleaned.csv'
    # csv_file="new_data.csv"
    data = pd.read_csv(csv_file)
    # data=data.drop(["num"],axis=1)
    # X=data.drop(["num","age","slope","thalach"], axis=1)
    # X=pd.DataFrame(data["age"],data["sex"])
    # X=data.drop(["num"],axis=1)
    # print(data)
    # print(data.columns)
    # print(data.corr().values)

    # X=pd.DataFrame(data["oldpeak"])
    # X["slope"]=data["slope"]
    # new_data=X[:]
    # new_data["num"]=data["num"]
    # # X = pd.DataFrame(data["thal"])
    # # print(X)
    # clf=KMeans(n_clusters=2).fit(X)
    # new_df=new_data[:]
    # # print(new_df)
    # # new_df=new_df.drop(["oldpeak","slope"],axis=1)
    # new_df["o_s"] = clf.labels_
    # print(new_df)
    # pca=PCA(n_components=2)
    # new_pca=pd.DataFrame(pca.fit_transform(new_df))
    # # print(new_df)
    # # new_df.to_csv('new_data.csv')
    # # train_test(new_df)
    # d=new_pca[new_df["o_s"]==0]
    # plt.plot(d[0],d[1],"r.")
    # d=new_pca[new_df["o_s"]==1]
    # plt.plot(d[0],d[1],"go")
    # d=new_pca[new_df["o_s"]==2]
    # plt.plot(d[0],d[1],"b*")
    # d=new_pca[new_df["o_s"]==3]
    # plt.plot(d[0],d[1],"y.")
    # d=new_pca[new_df["o_s"]==4]
    # plt.plot(d[0],d[1],"k")
    # plt.show()

    X=pd.DataFrame(data["age"])
    # X = pd.DataFrame(data["thal"])
    # print(X)
    clf=KMeans(n_clusters=3).fit(X)
    new_df=data[:]
    new_df["age_stage"] = clf.labels_
    pca=PCA(n_components=2)
    new_pca=pd.DataFrame(pca.fit_transform(new_df))
    # print(new_df)
    new_df.to_csv('new_data.csv')
    d=new_pca[new_df["age_stage"]==0]
    plt.plot(d[0],d[1],"r.")
    d=new_pca[new_df["age_stage"]==1]
    plt.plot(d[0],d[1],"go")
    d=new_pca[new_df["age_stage"]==2]
    plt.plot(d[0],d[1],"b*")
    plt.show()

train()
clf=train_test()
print(prediction(0,4,0,1.5,3,7,0,clf))

