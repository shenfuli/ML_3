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
        for line in lines:
            segs = jieba.lcut(line)
            segs = [seg for seg in segs if len(seg) > 1 and seg not in stopwords]
            data = " ".join(segs)
            label = category
            f.write(data + "\t" + label + "\n")  # 采用data  label 格式，不采用label  data 格式，后面数据处理使用

## 处理 label data 格式的数据
preprocess_text(technology, "data_sample/technology_sample.txt", "technology")
preprocess_text(entertainment, "data_sample/entertainment_sample.txt", "entertainment")
preprocess_text(car, "data_sample/car_sample.txt", "car")
preprocess_text(military, "data_sample/military_sample.txt", "military")
preprocess_text(sports, "data_sample/sports_sample.txt", "sports")

# ## 样本顺序打乱 并打印训练格式的样本
#
merge_data_path = "data_sample/data_label.txt"
category_lst = ['technology','entertainment','car','military','sports']
with open(merge_data_path,"w") as f_write:
    for category in category_lst:
        file_path = "data_sample/{0}_sample.txt".format(category)
        with open(file_path) as f:
            for line in f.readlines():
                f_write.write(line)

