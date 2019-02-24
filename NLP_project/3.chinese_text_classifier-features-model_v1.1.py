# -*- coding: UTF-8 -*-
'''
 特征工程处理
 1. CountVectorize  特征处理
 2. 分类模型的训练
'''


merge_data_path = "data_sample/data_label.txt"

documents = []
with open(merge_data_path,"r") as f:
    for line in f.readlines():
        data = line.split("\t")
        if len(data) == 2:
             documents.append((data[0],data[1].strip()))

print("documents size is {0}".format(len(documents)))
import random
random.shuffle(documents)

'''
为了一会儿检测一下咱们的分类器效果怎么样，我们需要一份测试集。

所以把原数据集分成训练集的测试集，咱们用sklearn自带的分割函数。
'''
from sklearn.model_selection import train_test_split
x,y = zip(*documents)
X_train, X_test, y_train, y_test = train_test_split(x,y,random_state=1234,test_size=0.3)
print(X_train[0:2])
print(y_train[0:2])
print("train data size = {0}".format(len(y_train)))
print("test  data size = {0}".format(len(y_test)))
'''
是在降噪数据上抽取出来有用的特征啦，我们对文本抽取词袋模型特征-CountVectorizer
CountVectorizer:
Convert a collection of text documents to a matrix of token counts

通过词频－sparevector

ngram 选择1，2，3
max_features = 20000
'''
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(analyzer='word', max_features=20000, ngram_range=(1, 4))
# Learn a vocabulary dictionary of all tokens in the raw documents.
vectorizer.fit(X_train)


'''
训练和预测得分
'''
from sklearn.naive_bayes import MultinomialNB
classifer = MultinomialNB()
classifer.fit(vectorizer.transform(X_train),y_train)
accuracy = classifer.score(vectorizer.transform(X_test),y_test)
print("accuracy = {0}".format(accuracy))


'''
交叉验证： 进一步提供效果
更可靠的验证效果的方式是交叉验证，但是交叉验证最好保证每一份里面的样本类别也是相对均衡的，我们这里使用StratifiedKFold
'''




'''
总结：
分类个数： 5中类别
特征： 采用CountVectorizer  1，2，3 grame 特征组合  20000特征词
模型： 选择MultinomialNB 分类
准确率：accuracy = 0.8837116897565628
'''