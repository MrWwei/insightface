import os
import shutil

'''
通过读取txt中的索引,把对应的图片提取出来
'''
# read txt method one
root = '/home/heisai/sambashare/HeisAI/facestore/v02/face'
list1 = os.listdir(root)
list2 = []
f = open("/home/heisai/sambashare/HeisAI/facestore/v02/faceid.txt")
line = f.readline()
while line:
    list2.append(line.strip())
    line = f.readline()
f.close()
count = 0
for id in list1:
    for id_txt in list2:
        if id_txt == id:
            source = os.path.join(root, id_txt)
            target = os.path.join('/media/heisai/My Passport/testData', id_txt)
            print('achieve:%d' % count)
            shutil.copytree(source, target)
            count += 1
