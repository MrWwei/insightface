import cv2
import numpy as np
import parser

import os

import shutil


def checkGray(path):
    chip = cv2.imread(path)
    # chip_gray = cv2.cvtColor(chip, cv2.COLOR_BGR2GRAY)
    r, g, b = cv2.split(chip)
    r = r.astype(np.float32)
    g = g.astype(np.float32)
    b = b.astype(np.float32)
    s_w, s_h = r.shape[:2]
    x = (r + b + g) / 3
    # x = chip_gray
    r_gray = abs(r - x)
    g_gray = abs(g - x)
    b_gray = abs(b - x)
    r_sum = np.sum(r_gray) / (s_w * s_h)
    g_sum = np.sum(g_gray) / (s_w * s_h)
    b_sum = np.sum(b_gray) / (s_w * s_h)
    gray_degree = (r_sum + g_sum + b_sum) / 3
    if gray_degree < 10:
        return 1
        # print "Gray"
    else:
        return 0
        # print "NOT Gray"


source = '/media/heisai/My Passport/heisai_5000/heisai_5000/images'
# color_dir = ''
# img = cv2.imread('/home/heisai/Pictures/gray.png')
# test = checkGray(img)
# print(test)

# 读取文件夹每张图片，判断是否为color

for root, dirs, files in os.walk(source):
    for img in files:
        sourceImg = os.path.join(root, img)
        print('deal' + sourceImg)
        isGray = checkGray(sourceImg)

        # if os.path.isdir(os.path.join('/home/heisai/'))
        if not isGray:
            # 判断文件夹是否存在，
            dir = os.path.basename(root)
            target_dir = os.path.join('/home/heisai/colors', dir)
            if not os.path.isdir(target_dir):
                os.mkdir(target_dir)
            shutil.copy(sourceImg, target_dir)
            # test1 = os.path.isdir(os.path.basename(root))
            # 同名文件夹，复制图片到此文件夹
    # print('test')
    # for dir in dirs:
    #     # 遍历里面的图片
    #     test = os.listdir(os.path.join(root, dir))
    #     print(test)
