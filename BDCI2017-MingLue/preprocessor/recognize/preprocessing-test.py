#!/usr/bin/python3
# -*- coding: UTF-8 -*-
# # https://blog.csdn.net/jianglianye21/article/details/78086768-
import sys
# # 注意：如果要导入该项目其他模块的包名，应将导入的方法写在上面方法的后面，如下：
sys.path.append('/home/ubuntu/work/AI/ML_3/BDCI2017-MingLue')

from preprocessor.recognize import toolkit


def main():
    print("hello")
    my_string = '将麻古0．44克、冰毒O．19克贩卖给买毒人员'
    print(toolkit.full2half(my_string))


if __name__ == '__main__':
    main()
