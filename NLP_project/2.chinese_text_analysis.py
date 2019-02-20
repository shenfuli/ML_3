#!/anaconda3/lib/python3.7
# -*- coding: UTF-8 -*-
import sys
import pandas as pd
import jieba
from jieba import analyse

# from optparse import OptionParser
#
# USAGE = "usage:    python extract_tags_stop_words.py [file name] -k [top k]"
# parser = OptionParser(USAGE)
# parser.add_option("-k", dest="topK")
# opt,args = parser.parse_args()
# if len(args) < 1:
#     print(USAGE)
#     sys.exit(1)
#
# file_name = args[0]
#
# if opt.topK is None:
#     topK = 10
# else:
#     topK = int(opt.topK)


# 关键词提取

# 基于tf-idf 关键词提取

'''

import jieba.analyse

jieba.analyse.extract_tags(sentence, topK=20, withWeight=False, allowPOS=())
sentence 为待提取的文本
topK 为返回几个 TF/IDF 权重最大的关键词，默认值为 20
withWeight 为是否一并返回关键词权重值，默认值为 False
allowPOS 仅包括指定词性的词，默认值为空，即不筛选
jieba.analyse.TFIDF(idf_path=None) 新建 TFIDF 实例，idf_path 为 IDF 频率文件
代码示例 （关键词提取）

https://github.com/fxsjy/jieba/blob/master/test/extract_tags.py
'''
# 设置停止词
jieba.analyse.set_stop_words("./data/stopwords.txt")

news_data_path = "./data/technology_news.csv"
news_data_df = pd.read_csv(news_data_path, sep=",", encoding="utf-8", names=["id", "content"], header=0)
news_data_df = news_data_df.dropna()
news_data_df["content"] = news_data_df.content.apply(lambda x: x.strip())
print(news_data_df.head())
news_lst = news_data_df.content.values.tolist()
lines = "".join(news_lst)
tfidf_tags = jieba.analyse.extract_tags(lines, topK=30, withWeight=False, allowPOS=['ns', 'n', 'vn', 'v', 'nr'])
print(tfidf_tags)


# 基于textrank 提取关键词
'''
jieba.analyse.textrank(sentence, topK=20, withWeight=False, allowPOS=('ns', 'n', 'vn', 'v')) 直接使用，接口相同，注意默认过滤词性。
jieba.analyse.TextRank() 新建自定义 TextRank 实例

基本思想:
将待抽取关键词的文本进行分词
以固定窗口大小(默认为5，通过span属性调整)，词之间的共现关系，构建图
计算图中节点的PageRank，注意是无向带权图
'''

textrank_tags = jieba.analyse.textrank(lines, topK=20, withWeight=False, allowPOS=('ns', 'n', 'vn', 'v'))
print(textrank_tags)