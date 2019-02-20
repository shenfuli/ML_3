# -*- coding: UTF-8 -*-

## 导入库

import warnings
warnings.filterwarnings("ignore")
import jieba    #分词包
import numpy  as np  #numpy计算包
import codecs   #codecs提供的open方法来指定打开的文件的语言编码，它会在读取的时候自动转换为内部unicode
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.figsize'] = (10.0, 5.0)
from wordcloud import WordCloud#词云包

seg_list = jieba.cut("我来自北京清华大学",cut_all=False)
print("/".join(seg_list))

lcut = jieba.lcut("我来自北京清华大学")
print(np.version,np.version.version)
## 导入新闻库，分词

news_data_path = "./data/entertainment_news.csv"
news_data_df = pd.read_csv(news_data_path,sep=",",encoding="utf-8",names=["id","content"],header=0)
# 删除空行
news_data_df = news_data_df.dropna()

segment=[]
content = news_data_df["content"].tolist()
for line in content:
    try:
        seg_list = jieba.lcut(line.strip())
        for seg in seg_list:
            if len(seg) > 1:
                segment.append(seg)
    except Exception:
        print(line)
        continue


## 去除停用词

stopwords_data_path = "data/stopwords.txt"
stopwords_data_df = pd.read_csv(stopwords_data_path,encoding="utf-8",sep="\t",index_col=None,quoting=3,names=["stopword"])
print(stopwords_data_df.head())
words_df = pd.DataFrame({"segment":segment})
words_df = words_df[~words_df.segment.isin(stopwords_data_df.stopword)]

print(words_df.head())

## 统计词频
words_tf = words_df.groupby(by = "segment")["segment"].agg({"count":np.size})
words_tf = words_tf.reset_index().sort_values(by="count",ascending=False)
print(words_tf.head())


## 词云
word_frequencies = {}
word_frequencies = {word_count[0]:word_count[1] for word_count in words_tf.head(1000).values}
wc = WordCloud(font_path="./data/simhei.ttf",max_font_size=100,background_color="white")
wc.fit_words(word_frequencies)
wc.to_file("./data/news_1.png")
#plt.imshow(wc)
#plt.show()

### 自定义背景图做词云

from scipy.misc import imread
matplotlib.rcParams['figure.figsize'] = (15.0, 15.0)
from wordcloud import WordCloud,ImageColorGenerator
bimg = imread("image/entertainment.jpeg")
wc = WordCloud(font_path="./data/simhei.ttf",max_font_size=100,background_color="white",mask=bimg) # 指定背景图
wc.fit_words(word_frequencies)
bimgColors = ImageColorGenerator(bimg) # 基于背景图生成颜色
plt.axis("off")
plt.imshow(wc.recolor(color_func=bimgColors))
wc.to_file("./data/news_2.png")
