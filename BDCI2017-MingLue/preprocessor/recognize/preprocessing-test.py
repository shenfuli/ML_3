#!/usr/bin/python
# -*- coding: UTF-8 -*-
# # https://blog.csdn.net/jianglianye21/article/details/78086768-
import sys
# # 注意：如果要导入该项目其他模块的包名，应将导入的方法写在上面方法的后面，如下：
sys.path.append('./')
from preprocessor.recognize import toolkit
def main():
    print("hello")
    print(toolkit.full2half("将麻古0．44克、冰毒O．19克贩卖给买毒人员"))
    print(toolkit.zhnum2int("三千五百二十三"))

if __name__ == '__main__':
    main()