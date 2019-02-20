#!/anaconda3/lib/python3.7
# -*- coding: UTF-8 -*-

from gensim import corpora,models,similarities
import  gensim
from gensim.models import LdaModel
'''
https://blog.csdn.net/l7h9ja4/article/details/80220939

词袋模型中，文档的特征就是其包含的word
texts = [['human', 'interface', 'computer'],
['survey', 'user', 'computer', 'system', 'response', 'time'],
['graph', 'minors', 'survey']]
其中，corpus的每一个元素对应一篇文档。


步骤一：训练语料的预处理
Gensim提供的API建立语料特征（此处即是word）的索引字典，并将文本特征的原始表达转化成词袋模型对应的稀疏向量的表达
from gensim import corpora
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
print corpus[0] # [(0, 1), (1, 1), (2, 1)]


出于内存优化的考虑，Gensim支持文档的流式处理。我们需要做的，只是将上面的列表封装成一个Python迭代器；每一次迭代都返回一个稀疏向量即可。
class MyCorpus(object):
def __iter__(self):
    for line in open('mycorpus.txt'):
        # assume there's one document per line, tokens                   separated by whitespace
        yield dictionary.doc2bow(line.lower().split())

步骤二：主题向量的变换
每一个向量变换的操作都对应着一个主题模型


步骤三：文档相似度的计算
得到每一篇文档对应的主题向量后，我们就可以计算文档之间的相似度，进而完成如文本聚类、信息检索之类的任务
一篇待检索的query，我们的目标是从文本集合中检索出主题相似度最高的文档。

'''

import jieba
import pandas as pd
news_data_path = "./data/technology_news.csv"
news_data_df = pd.read_csv(news_data_path, sep=",", encoding="utf-8", names=["id", "content"], header=0)
news_data_df = news_data_df.dropna()
news_data_df["content"] = news_data_df.content.apply(lambda x: x.strip())
print(news_data_df.head())
## 去除停用词
stopwords_data_path = "data/stopwords.txt"
stopwords_data_df = pd.read_csv(stopwords_data_path,encoding="utf-8",sep="\t",index_col=None,quoting=3,names=["stopword"])
print(stopwords_data_df.head())

stopwords_list = stopwords_data_df.stopword.tolist()
print(stopwords_list[0:4])

lines = news_data_df.content.tolist()
sentences=[]
for line in lines:
    try:
        segs = jieba.lcut(line)
        segs_list=[]
        for seg in segs:
            if(len(seg) > 1) and (seg not in stopwords_list):
                segs_list.append(seg)
        sentences.append(segs_list)
    except Exception:
        continue

# 词典模型
dictionary = corpora.Dictionary(sentences)
corpus = [dictionary.doc2bow(sentence) for sentence in sentences]
# 基于词袋模型训练LDA 模型
lda = gensim.models.ldamodel.LdaModel(corpus=corpus,id2word=dictionary,num_topics=20)

# lda 模型保持
from gensim.test.utils import datapath
temp_file = datapath("lda_20190222.model")
lda.save(temp_file)

# 查看第三号分类，以及常出现的词语
lda.print_topic(3,topn=10)
# 所有的主题打印出来看看
for topic in lda.print_topics(num_topics=20,num_words=10):
    print(str(topic[0]) + "---->" + topic[1])



