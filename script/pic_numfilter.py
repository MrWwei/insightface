# -*- coding: utf-8 -*-
# 参考链接：http://www.runoob.com/python/os-walk.html
import os, os.path
import shutil
import argparse

'''
将图片数量大于30的文件提取出来
'''
def copydir(source, target):
    # 要查找的文件夹地址
    number = 0
    # os.walk()方法是一个简单易用的文件、目录遍历器
    # root正在遍历的这个文件夹的本身的地址
    # dirname是一个list,内容是该文件夹中所有的目录的名字(不包括子目录)
    # filenames同样是list,内容是该文件夹中所有的文件名字(不包括子目录)
    for root, dirname, filenames in os.walk(source):
        for filename in filenames:
            print(filename)
            # os.path.splitext()是一个元组,类似于('188739', '.jpg')，索引1可以获得文件的扩展名
            if os.path.splitext(filename)[1] != '.xml':
                number += 1
    ext = os.path.basename(source)
    # ext = source.split('/')[-1]
    if number > 30:
        # 复制到另一个文件夹
        shutil.copytree(source, os.path.join(target, ext))


parser = argparse.ArgumentParser(description='Package LFW images')
parser.add_argument('--source-dir', default='/home/heisai/disk/HeisAI_data/face_bp', help='')
parser.add_argument('--target-dir', default='/home/heisai/disk/HeisAI_data/face_30', help='')
args = parser.parse_args()
source = args.source_dir
target = args.target_dir

for root, dirname, filenames in os.walk(source):
    for dir in dirname:
        copydir(os.path.join(source, dir), target)
