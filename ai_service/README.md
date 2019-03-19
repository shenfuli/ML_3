# Tornado web serving 
python编写的web服务器兼web应用框架 <br>

https://blog.csdn.net/xc_zhou/article/details/80637714 <br>
http://www.tornadoweb.org/en/stable/
## 1. Tornado的优势
* 轻量级web框架
* 异步非阻塞IO处理方式
* 出色的抗负载能力
* 优异的处理性能，不依赖多进程/多线程，一定程度上解决C10K问题
* WSGI全栈替代产品，推荐同时使用其web框架和HTTP服务器

## 2. Tornado VS Django
* 内置管理后台 
* 内置封装完善的ORM操作 
* session功能 
* 后台管理 
* 缺陷：高耦合
* Tornado：轻量级web框架，功能少而精，注重性能优越 
* HTTP服务器 
* 异步编程 
* WebSocket 
* 缺陷：入门门槛较高

## 3. Tornado 安装
pip install tornado <br>
**备注：** Tornado应该运行在类Unix平台，为了达到最佳的性能和扩展性，仅推荐Linux和BSD(充分利用Linux的epoll工具和BSD的kqueue达到高性能处理的目的)

## 4. 搭建tornado web 示例程序
### 4.1 main.py 程序
```
# -*- coding: UTF-8 -*-
import tornado.web
import tornado.ioloop


class IndexHandler(tornado.web.RequestHandler):
    '''
        定义处理类型
    '''

    def get(self):
        '''
         添加一个处理get请求方式的方法
        :return:
        '''
        # 向响应中，添加数据
        self.write('好看的皮囊千篇一律，有趣的灵魂万里挑一。')


if __name__ == "__main__":
    # 创建一个应用对象
    app = tornado.web.Application([(r'/', IndexHandler)])
    # 绑定一个监听端口
    app.listen(8080)
    # 启动web程序，开始监听端口的连接
    tornado.ioloop.IOLoop.current().start()
```

### 4.2 运行并访问
python main.py <br>

http://localhost:8080
