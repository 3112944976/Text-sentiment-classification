from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from tool import DataLoad, preprocess_for_ml


# 数据集切分
def set_cut(file_url):
    data_set = pd.read_csv(file_url, header=None, encoding='gbk')
    data_set = data_set.astype(str)
    train_data, test_data = train_test_split(data_set, test_size=0.05, random_state=42)
    with open('.//datasets/train.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for row in train_data.itertuples(index=False):
            writer.writerow((str(row[0]), str(row[1])))
    with open('.//datasets/test.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for row in test_data.itertuples(index=False):
            writer.writerow((str(row[0]), str(row[1])))


# 数据预处理
def set_process(train_filename, test_filename):
    train_dataset = DataLoad(train_filename)
    test_dataset = DataLoad(test_filename)
    train_labels, train_sents = zip(*train_dataset.pairs)
    test_labels, test_sents = zip(*test_dataset.pairs)
    train_sents = preprocess_for_ml(train_sents)
    test_sents = preprocess_for_ml(test_sents)
    tfidf = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
    train_st = tfidf.fit_transform(train_sents)
    test_st = tfidf.transform(test_sents)
    return train_st, test_st, train_labels, test_labels


# 逻辑斯蒂回归
def LogRegression(train_st, test_st, train_lab, test_lab, solver, maxiter):
    lr_clf = LogisticRegression(solver=solver, max_iter=maxiter)
    lr_clf.fit(train_st, train_lab)
    predicted = lr_clf.predict(test_st)
    acc = np.mean(predicted == np.array(test_lab))
    x_label.append("LogisticRegression")
    y.append(acc)
    return


# 朴素贝叶斯
def Bayes(train_st, test_st, train_lab, test_lab):
    nb_clf = MultinomialNB()
    nb_clf.fit(train_st, train_lab)
    predicted = nb_clf.predict(test_st)
    acc = np.mean(predicted == np.array(test_lab))
    x_label.append("NaiveBayes")
    y.append(acc)
    return


# 支持向量机
def Svm(train_st, test_st, train_lab, test_lab):
    sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3)
    sgd_clf.fit(train_st, train_lab)
    predicted = sgd_clf.predict(test_st)
    acc = np.mean(predicted == np.array(test_lab))
    x_label.append("SVM")
    y.append(acc)
    return


# K近邻
def Knn(train_st, test_st, train_lab, test_lab):
    kn_clf = KNeighborsClassifier()
    kn_clf.fit(train_st, train_lab)
    predicted = kn_clf.predict(test_st)
    acc = np.mean(predicted == np.array(test_lab))
    x_label.append("KNN")
    y.append(acc)
    return


# 随机森林
def RandomForest(train_st, test_st, train_lab, test_lab):
    rf_clf = RandomForestClassifier(n_estimators=20)
    rf_clf.fit(train_st, train_lab)
    predicted = rf_clf.predict(test_st)
    acc = np.mean(predicted == np.array(test_lab))
    x_label.append("RandomForest")
    y.append(acc)
    return


# K均值(需要运行很久的时间，并且效果不好)
def Kmeans(train_st, test_st, test_lab):
    km_clf = KMeans(n_clusters=2).fit(train_st)
    predicted = km_clf.predict(test_st)
    acc = np.mean(predicted == np.array(test_lab))
    x_label.append("Kmeans")
    y.append(acc)
    return


def show(x_label, y, title=None):
    y_percent = [str(round(i * 100, 2)) + '%' for i in y]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(x_label, y, color='#A3D1D1')
    for i, v in enumerate(y_percent):
        ax.text(i, y[i], v, ha='center', va='bottom')
    plt.xticks(rotation=45)
    if title:
        ax.set_title(title)
    plt.show()


if __name__ == "__main__":
    # set_cut(".//datasets/new_comment.csv")
    x_label = []
    y = []
    train_st, test_st, train_lab, test_lab = set_process("train", "test")
    LogRegression(train_st, test_st, train_lab, test_lab, "lbfgs", 3000)
    Bayes(train_st, test_st, train_lab, test_lab)
    Svm(train_st, test_st, train_lab, test_lab)
    Knn(train_st, test_st, train_lab, test_lab)
    RandomForest(train_st, test_st, train_lab, test_lab)
    Kmeans(train_st, test_st, test_lab)
    show(x_label, y, title='Comparison of accuracy of various algorithms')
