#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from libshorttext.classifier import *
'''
https://www.csie.ntu.edu.tw/~cjlin/libshorttext/doc/libshorttext.html#installation-and-data-format

https://www.csie.ntu.edu.tw/~cjlin/libshorttext/doc/_sources/classifier.txt
LibShortText是一个开源的Python短文本（包括标题、短信、问题、句子等）分类工具包。它在LibLinear的基础上针对短文本进一步优化，
主要特性有：
支持多分类
直接输入文本，无需做特征向量化的预处理
二元分词（Bigram），不去停顿词，不做词性过滤
基于线性核SVM分类器（参见SVM原理简介：最大间隔分类器），训练和测试的效率极高
提供了完整的API，用于特征分析和Bad Case检验
'''
# training
model_path="m1.model"
train_file_path = "demo/train_file"
model,svm = train_text(train_file_path)
model.save(model_path, True)

# testing
test_file_path = "demo/test_file"
m = TextModel()
m.load(model_path)
predict_result = predict_text(test_file_path,m)
print("Accuracy = {0:.4f}% ({1}/{2})".format(
		predict_result.get_accuracy()*100,
		sum(ty == py for ty, py in zip(predict_result.true_y, predict_result.predicted_y)),
		len(predict_result.true_y)))



# ***
# optimization finished, #iter = 8
# Objective value = -986.801900
# nSV = 18974
# Accuracy = 87.1800% (4359/5000)
