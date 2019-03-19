#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 21:29:47 2017

@author: llh
"""
from itertools import groupby


def show_text_len_distribution(data):
    len_list = [len(text) for text in data]
    print(len_list[1:100])
    step = 1000
    for k, g in groupby(sorted(len_list), key=lambda x: (x - 1) // step):
        print('{}-{}'.format(k * step + 1, (k + 1) * step) + ":" + str(len(list(g))))


def count_vocab_size(data):
    '''
    格式：统计不同词汇的个数
        [['公诉', '机关', '海口市', '龙华区'], ['公诉', '机关', '平湖市', '人民检察院']]
    方法： 通过两个set 数据求并集操作
    '''
    vocab_set = set()
    for text in data:
        vocab_set |= set(text)
    return len(vocab_set)

if __name__ == "__main__":
    print("hello")
    data = [
        '公诉 机关 海口市 龙华区'.split(" "),
        "公诉 机关 平湖市 人民检察院".split(" ")
    ]
    print(data)
    print(count_vocab_size(data))

    s1 = set('公诉 机关 海口市 龙华区'.split(" "))
    s2 = set("公诉 机关 平湖市 人民检察院".split(" "))
    print(s1 | s2 ) # 求两个set的并集