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

'''

from gensim.models.fasttext import FastText as FT_gensim
from gensim.test.utils import datapath
