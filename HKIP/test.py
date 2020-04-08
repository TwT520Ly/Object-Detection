"""
    @Author: Tan.wt
    @Time: 2018/05/28
    @File: test.py.py
    @License: Apache License
"""


import HKIPcamera
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2


ip = str('10.0.9.160')  # 摄像头IP地址，要和本机IP在同一局域网
name = str('admin')       # 管理员用户名
pw = str('8023hwjm.')        # 管理员密码
HKIPcamera.init(ip, name, pw)
# HKIPcamera.getfram()
for i in range(100):
    t = time.time()
    fram = HKIPcamera.getframe()
    t2 = time.time()
    cv2.imshow('123', np.array(fram))
    cv2.waitKey(1)
    print(t2-t)
    time.sleep(0.1)
HKIPcamera.release()
time.sleep(5)
HKIPcamera.init(ip, name, pw)
# HKIPcamera.getfram()
for i in range(100):
    t = time.time()
    fram = HKIPcamera.getframe()
    t2 = time.time()
    cv2.imshow('123', np.array(fram))
    cv2.waitKey(1)
    print(t2-t)
    time.sleep(0.1)
HKIPcamera.release()