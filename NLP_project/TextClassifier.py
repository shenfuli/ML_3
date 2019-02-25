# -*- coding: UTF-8 -*-
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

class TextClassifier():
    def __init__(self, classifier=MultinomialNB()):
        '''
        文本分类器构造方法，初始化分类器／ 特征方法
        :param classifier:
        '''
        self.classifier = classifier
        self.vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 4), max_features=20000)
    def features(self, X):
        '''
        根据raw_document 获取特征
        :param X: 格式： [[word1 word2].......] 矩阵
        :return: 通过vectorizer 进行特征转换
        '''
        return self.vectorizer.transform(X)

    def fit(self, X, y):
        '''
        获取词典器－>抽取特征－> 模型训练
        :param X:
        :param y:
        :return:
        '''
        self.vectorizer.fit(X)
        self.classifier.fit(self.vectorizer.transform(X), y)

    def score(self, X, y):
        '''
        获取分类得分
        :param X: 验证集的 raw_documents matrix
        :param y: 验证集的 y 的matrix
        :return:
        '''
        return self.classifier.score(self.features(X), y)

    def predict(self, x):
        '''
        指定的文本，预测分类结果
        :param x: word1 word2 ,分词后的结果作为预测输出
        :return:
        '''
        return self.classifier.predict(self.features([x]))


if __name__ == "__main__":

    # 获取训练和验证集数据
    merge_data_path = "data_sample/data_label.txt"
    documents = []
    with open(merge_data_path, "r") as f:
        for line in f.readlines():
            data = line.split("\t")
            if len(data) == 2:
                documents.append((data[0], data[1].strip()))
    print("documents size is {0}".format(len(documents)))
    import random

    random.shuffle(documents)
    x, y = zip(*documents)
    X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=1234, test_size=0.3)
    print(X_train[0:2])
    print(y_train[0:2])
    print("train data size = {0}".format(len(y_train)))
    print("test  data size = {0}".format(len(y_test)))

    text_classifier = TextClassifier()
    # 模型训练
    text_classifier.fit(X_train, y_train)
    precision = text_classifier.score(X_test, y_test)
    print("text classfier score = {0}".format(precision))


    # # 预测
    x = '这 是 有史以来 最 大 的 一 次 军舰 演习'
    category = text_classifier.predict(x)
    print("predict category is {0}".format(category[0]))
