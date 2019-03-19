# -*- coding: UTF-8 -*-
import tornado.web
import tornado.ioloop


class IndexHandler(tornado.web.RequestHandler):
    '''
        tornado.web : tornado的基础web框架定义处理类型
        RequestHandler：封装对请求处理的所有信息和处理方法
        get/post/..：封装对应的请求方式
        write()：封装响应信息，写响应信息的一个方法
    '''

    def get(self):
        '''
         添加一个处理get请求方式的方法
        :return:
        '''
        # 向响应中，添加数据
        self.write('好看的皮囊千篇一律，有趣的灵魂万里挑一。')

    def post(self):
        pass

if __name__ == "__main__":
    # 创建一个应用对象
    app = tornado.web.Application([(r'/', IndexHandler)])
    # 绑定一个监听端口
    app.listen(8080)
    # 启动web程序，开始监听端口的连接
    tornado.ioloop.IOLoop.current().start()
