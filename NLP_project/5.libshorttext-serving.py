#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from sys import argv, stderr
from os import system, path
from libshorttext.classifier import *

model_path="m1.model"
m = TextModel()
m.load(model_path)
# 3  226:1 227:1 233:1 1133:1 3215:1 3998:1 5500:1
text = "226:1 227:1 233:1 1133:1 3215:1 3998:1 5500:1"
predict_results = predict_single_text(text,m)
print(predict_results)
print(predict_results.labels)

text1="96:1 229:2 520:1 1236:1 24237:1 27612:1 43433:1"
predict_results2 = predict_single_text(text1,m)
print(predict_results2)


# unanalyzable result: Stamps
# ['Tickets', 'Stamps', 'Music', 'Jewelry & Watches', 'Books', 'Art', 'Baby', 'Travel', 'Crafts', 'Antiques']
# unanalyzable result: Crafts
