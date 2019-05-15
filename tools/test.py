import os
import util
import mxnet as mx
import numpy as np
import matplotlib.pyplot as plt
import png.dom.minidom  #处理png数据

#首先定义一个读取png文件的函数：
def pngDecode(path):
    annotation = png.dom.minidom.parse(path)

    size = annotation.getElementsByTagName('size')[0]
    width = size.getElementsByTagName('width')[0].firstChild.data
    height = size.getElementsByTagName('height')[0].firstChild.data

    obj = annotation.getElementsByTagName('object')[0]
    cla = obj.getElementsByTagName('name')[0].firstChild.data  #类别
    bndbox = obj.getElementsByTagName('bndbox')[0]              #坐标
    x1 = bndbox.getElementsByTagName('xmin')[0].firstChild.data
    x2 = bndbox.getElementsByTagName('xmax')[0].firstChild.data
    y1 = bndbox.getElementsByTagName('ymin')[0].firstChild.data
    y2 = bndbox.getElementsByTagName('ymax')[0].firstChild.data
    

    width = int(width)
    height = int(height)
    x1 = int(x1)
    x2 = int(x2)
    y1 = int(y1)
    y2 = int(y2)
    result = [cla,(width,height),(x1,y1),(x2,y2)]
    return result


#定义保存数据和标签文件夹路径
path = 'your/data/set/folder/'

#假设图片名和对应的标签名称一致，这里直接替换png为jpg
#TODO:read png and write as lst file, jpg name as path + png name
#format:0  4  5  640(width)  480(height)  1(class)  0.1  0.2  0.8  0.9(xmin, ymin, xmax, ymax)  2  0.5  0.3  0.6  0.8  data/xxx.jpg
names = os.listdir(path)
lst = []
i=0
f = open(path+'train.lst','w')
for name in names:
    if name.endswith('.png'):
        result = pngDecode(path+name)
        img_name = path+name.replace('png','jpg')
        lst_tmp =str(i)+'\t4'+'\t5'+'\t'+str(result[1][0])+'\t'+str(result[1][1])+'\t'\
        +str(result[0])+'\t'\
        +str(result[2][0]/result[1][0])+'\t'+str(result[2][1]/result[1][1])+'\t'\    #计算图像参数的相对值
        +str(result[3][0]/result[1][0])+'\t'+str(result[3][1]/result[1][1])+'\t'\
        +img_name+'\n'
        #print(lst_tmp)
        f.write(lst_tmp)
        i+=1
f.close()

#运行结束就可以在path对应文件夹下看到lst文件了。
