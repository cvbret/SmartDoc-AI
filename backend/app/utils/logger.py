import logging


logger = logging.getLogger("smartdoc")#创建日志对象logger

logger.setLevel(logging.INFO)#设置日志级别为INFO


handler = logging.StreamHandler()#创建控制台处理器handler

formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    #时间 - 日志器名称 - 日志级别 - 日志消息    
)


handler.setFormatter(formatter)#为处理器设置格式化器formatter

logger.addHandler(handler)#为处理器添加到日志器logger