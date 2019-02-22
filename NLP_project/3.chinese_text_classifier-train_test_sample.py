# -*- coding: UTF-8 -*-

documents = []
with open("data_sample/data_label.txt") as f:
       for line in f.readlines():
           documents.append(line)

import random
random.shuffle(documents)
