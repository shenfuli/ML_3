# -*- coding: UTF-8 -*-

from libshorttext.classifier import *
'''
https://www.csie.ntu.edu.tw/~cjlin/libshorttext/doc/libshorttext.html#installation-and-data-format

https://www.csie.ntu.edu.tw/~cjlin/libshorttext/doc/_sources/classifier.txt


'''
# training
model,svm_file = train_text("demo/train_file")
# test
results = predict_text('demo/test_file', model)
results.save('result_path')