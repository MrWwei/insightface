import codecs
import matplotlib.pylab as plt
import numpy as np
import sys
def read_data(file_name):

    with codecs.open(file_name,mode='r',encoding='utf-8') as fr:
        lines = fr.readlines()
    
    xx = []
    yy = []
    for line in lines:
        infos = line.strip().split()
        xx.append(float(infos[0]))
        yy.append(float(infos[1]))

    return yy,xx

algs = sys.argv[1]

algs = algs.split(',')
plt.figure()
for alg in algs:
    roc_txt = '%s_roc.txt'%alg
    x,y=read_data(roc_txt)
    plt.plot(np.array(x),np.array(y),label=alg)
plt.legend(loc = 'upper right')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.xlim(0,1)
plt.ylim(0,1)
plt.title('ROC')
plt.savefig('./roc_compare.png')
plt.show()