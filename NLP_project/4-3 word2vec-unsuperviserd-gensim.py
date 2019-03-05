# -*- coding: UTF-8 -*-
'''

在gensim 中已经基于fastText 即成来c++ 和 python 接口的版本，我们可以直接使用gensim fasttext

目的： 通过fasttext库 需要学习的内容
1. training word－embedding models
2. saving & loading models
3. performing simililary operations & vector
4. fasttext vs word2vec


[1]gensim 官方网站
https://radimrehurek.com/gensim/models/fasttext.html
[2]fastText word vector-Enriching word vector with Subword information
https://arxiv.org/abs/1607.04606
[3]fastText by gensim version . notebook tutorial
https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/FastText_Tutorial.ipynb
[4]gensim-word2vec
http://www.52nlp.cn/tag/gensim-word2vec
'''

# gensim 的 word2vec 方式 获取词表示
## 1.1 可以根据词表示获取 某个词的相关词  1.2 词向量表示可以计算文本的相似度
import gensim
import logging
import multiprocessing
from time import time
import numpy as np
from gensim.models.word2vec import Word2Vec

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)
begin = time()
sentences = []
data_path = "data_sample/fasttext_unsupervised_train_data.txt"
'''
    >>> from gensim.models import Word2Vec
    >>> sentences = [["cat", "say", "meow"], ["dog", "say", "woof"]]
    >>> model = Word2Vec(sentences, min_count=1)
'''
with open(data_path, "r", encoding='UTF-8') as f:
    lines = f.readlines()
    print("load total sample size = {0}".format(len(lines)))
    for line in lines:
        line_lst = line.strip().split(" ")
        sentences.append(line_lst)

print(sentences[0:2])
print("training word2vec model start....")
model = Word2Vec(sentences, size=100, window=5, min_count=5, workers=multiprocessing.cpu_count())
print("training word2vec model end....")
# model.save("model/word2vec_gensim")
# model.wv.save_word2vec_format("model/word2vec_org", "model/vocabulary", binary=False)
end = time()
print("Total procesing time: %d seconds" % (end - begin))

word_similar = model.wv.most_similar("中国", topn=10)
print(word_similar)


# 对每个句子的所有词向量取均值，来生成一个句子的vector
def build_sentence_vector(text, size, imdb_w2v):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in text:
        try:
            vec += imdb_w2v[word].reshape((1, size))
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec
