# -*- coding: UTF-8 -*-
import fastText

data_path = "data_sample/fasttext_unsupervised_train_data.txt"
'''
    fastText 中采用两种方式对文本无监督训练
    1. skip－gram
    2. cbow
'''

# skip-gram model

# skip-gram 用中心词汇来预测其上下文
# fastText_skipgram_model = fastText.train_unsupervised(data_path,model="skipgram")
# fastText_skipgram_model.save_model("fastText_skipgram_model")

skipgram_model = fastText.load_model("fastText_skipgram_model")
words = skipgram_model.get_words() # list of words in dictionary
print(words)
print(skipgram_model.get_word_vector('中国'))

# cbow model train
# fastText_cbow_model = fastText.train_unsupervised(data_path, model="cbow", dim=100)
# fastText_cbow_model.save_model("fastText_cbow_model")
fastText_cbow_model = fastText.load_model("fastText_cbow_model")
fastText_cbow_words = fastText_cbow_model.get_words()
print(fastText_cbow_words) # list of words in dictionary

print(fastText_cbow_model.get_word_vector("中国"))


'''
总结，
fastText 中提供word和sentence的vector 表示，但没有提供词相关词，我们可以通过
计算cos 来判断两个文本的相似度。

在gensim 中已经基于fastText 即成来c++ 和 python 接口的版本，我们可以直接使用

[1]gensim 官方网站
https://radimrehurek.com/gensim/models/fasttext.html
[2]fastText word vector-Enriching word vector with Subword information 
https://arxiv.org/abs/1607.04606
[3]fastText by gensim version . notebook tutorial 
https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/FastText_Tutorial.ipynb
'''