# -*- coding: UTF-8 -*-
import pandas as pd
import jieba

#停用词
stopwords_data_path = "data/stopwords.txt"
stopwords_data_df = pd.read_csv(stopwords_data_path,encoding="utf-8",sep="\t",index_col=None,quoting=3,names=["stopword"])

# 准备数据
df_technology = pd.read_csv("./data/technology_news.csv", encoding='utf-8', names=["id", "content"], header=0)
df_technology = df_technology.dropna()

df_car = pd.read_csv("./data/car_news.csv", encoding='utf-8', names=["id", "content"], header=0)
df_car = df_car.dropna()

df_entertainment = pd.read_csv("./data/entertainment_news.csv", encoding='utf-8', names=["id", "content"], header=0)
df_entertainment = df_entertainment.dropna()

df_military = pd.read_csv("./data/military_news.csv", encoding='utf-8', names=["id", "content"], header=0)
df_military = df_military.dropna()

df_sports = pd.read_csv("./data/sports_news.csv", encoding='utf-8', names=["id", "content"], header=0)
df_sports = df_sports.dropna()

stopwords = stopwords_data_df.stopword.values.tolist()
technology = df_technology.content.apply(lambda line: line.strip()).values.tolist()
car = df_car.content.apply(lambda line: line.strip()).values.tolist()
entertainment = df_entertainment.content.apply(lambda line: line.strip()).values.tolist()
military = df_military.content.apply(lambda line: line.strip()).values.tolist()
sports = df_sports.content.apply(lambda line: line.strip()).values.tolist()

# 分词和中文处理
'''
5类数据最终处理的格式：
word1 word2 word3 technology 
word1 word2 word3 word4 technology 
word1 word2 word4 car 
'''

'''
    lines: 文章的一条记录
    sentences: 返回的数据列表
    category： 文章的类别
'''
def preprocess_text(lines, documents):
    for line in lines:
        segs = jieba.lcut(line)
        segs = [seg for seg in segs if len(seg) > 1 and seg not in stopwords]
        data = " ".join(segs)
        documents.append(data)

# 生成fasttext 无监督学习的样本数据
documents = []
preprocess_text(technology, documents)
preprocess_text(entertainment, documents)
preprocess_text(car, documents)
preprocess_text(military, documents)
preprocess_text(sports, documents)

# ## 样本顺序打乱 并打印训练格式的样本
import random
random.shuffle(documents)

print("writing data to fasttext unsupervised learning format...")
data_path = "data_sample/fasttext_unsupervised_train_data.txt"
with open(data_path,"w") as f_write:
    for document in documents:
        f_write.write(document.strip()+"\n")
print("done!")

