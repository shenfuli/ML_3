# -*- coding: UTF-8 -*-

# 导入库
import jieba
import pandas as pd

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
def preprocess_text(lines, file_name, category):

    with open(file_name,"w") as f:
        for line in lines[0:10]:
            segs = jieba.lcut(line)
            segs = [seg for seg in segs if len(seg) > 1 and seg not in stopwords]
            data = " ".join(segs)
            label = category
            f.write(label + " " + data + "\n")

preprocess_text(technology, "data_sample/technology_sample.txt", "technology")
preprocess_text(entertainment, "data_sample/entertainment_sample.txt", "entertainment")
preprocess_text(car, "data_sample/car_sample.txt", "car")
preprocess_text(military, "data_sample/military_sample.txt", "military")
preprocess_text(sports, "data_sample/sports_sample.txt", "sports")

# preprocess_text(car, sentences, "car")
# preprocess_text(entertainment, sentences, "entertainment")
# preprocess_text(military, sentences, "military")
# preprocess_text(sports, sentences, "sports")


## 样本顺序打乱 并打印训练格式的样本

