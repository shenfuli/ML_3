# -*- coding: UTF-8 -*-
import fastText
input = "data_sample/fasttext_train_data.txt"
output = "fasttext_20190226.model"
## fasttext train
classifier = fastText.train_supervised(input).save_model(output)

## fasttext evaluate
fasttext_evaluate = fastText.load_model(output).test(input)
rows = fasttext_evaluate[0]
precision = fasttext_evaluate[1]
recall = fasttext_evaluate[2]

print("test rows={0}".format(rows))
print("precision={0}".format(precision))
print("recall={0}".format(recall))

## fasttext precision
text = ['这 是 中国 第 一 次 军舰 演习']
lables = fastText.load_model(output).predict(text)
print(lables)

cate_dic = {'technology':1, 'car':2, 'entertainment':3, 'military':4, 'sports':5}
