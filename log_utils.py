# -*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import logging

def logger_fn(name, filepath, level = logging.DEBUG):
	""" Function for creating log manager
		Args:
			name: name for log manager
			filepath: file path for log file
			level: log level (CRITICAL > ERROR > WARNING > INFO > DEBUG)
		Return:
			log manager
	"""
	logger = logging.getLogger(name)
	logger.setLevel(level)

	sh = logging.StreamHandler(sys.stdout)
	fh = logging.FileHandler(filepath, mode = 'w')

	# formatter = logging.Formatter('[%(asctime)s][%(levelname)s][%(filename)s][line:%(lineno)d] %(message)s')
	# formatter = logging.Formatter('[%(asctime)s][%(filename)s][line:%(lineno)d] %(message)s')
	formatter = logging.Formatter('[%(asctime)s] %(message)s')
	"""
	%(levelno)s: 打印日志级别的数值
	%(levelname)s: 打印日志级别名称
	%(pathname)s: 打印当前执行程序的路径，其实就是sys.argv[0]
	%(filename)s: 打印当前执行程序名
	%(funcName)s: 打印日志的当前函数
	%(lineno)d: 打印日志的当前行号
	%(asctime)s: 打印日志的时间
	%(thread)d: 打印线程ID
	%(threadName)s: 打印线程名称
	%(process)d: 打印进程ID
	%(message)s: 打印日志信息
	"""
	sh.setFormatter(formatter)
	fh.setFormatter(formatter)

	logger.addHandler(sh)
	logger.addHandler(fh)

	return logger