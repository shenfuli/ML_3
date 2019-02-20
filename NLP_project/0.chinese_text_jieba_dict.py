#!/anaconda3/lib/python3.7
# -*- coding: UTF-8 -*-
from __future__ import print_function, unicode_literals
import jieba
jieba.load_userdict("userdict.txt") # 可以通过自定义词典来提高分词准备性，也可以添加上一些新词
import jieba.posseg as pseg


test_sent = (
"李小福是创新办主任也是云计算方面的专家; 什么是八一双鹿\n"
"例如我输入一个带“韩玉赏鉴”的标题，在自定义词库中也增加了此词为N类\n"
"「台中」正確應該不會被切開。mac上可分出「石墨烯」；此時又可以分出來凱特琳了。"
)

# 分词
words = jieba.cut(test_sent)
print('/'.join(words))


# 分词 标注词性
print("="*40)
result = pseg.cut(test_sent)
for w in result:
    print(w.word, "/", w.flag, ", ", end=' ')
