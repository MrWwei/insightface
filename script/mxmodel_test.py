import mxnet as mx
import cv2
import numpy as np
from collections import namedtuple
import sklearn

from sklearn import preprocessing

'''
使用单张图片测试mxnet版mobilefacenet模型
'''
Batch = namedtuple('Batch', ['data'])


def get_image(url):
    # download and show the image

    img = cv2.cvtColor(cv2.imread(url), cv2.COLOR_BGR2RGB)
    if img is None:
        return None
    img = cv2.resize(img, (112, 112))
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    img = img[np.newaxis, :]
    return img


def get_testFeature(pretrained, img_path):
    vec = pretrained.split(',')
    sym, arg_params, aux_params = mx.model.load_checkpoint(vec[0], int(vec[1]))

    all_layers = sym.get_internals()
    ctx = mx.cpu()
    sym = all_layers['fc1_output']
    model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
    model.bind(data_shapes=[('data', (1, 3, 112, 112))])
    model.set_params(arg_params, aux_params)

    img = get_image(img_path)
    model.forward(Batch([mx.nd.array(img)]), is_train=False)
    embedding = model.get_outputs()[0].asnumpy()

    embedding = sklearn.preprocessing.normalize(embedding)
    embedding = preprocessing.normalize(embedding)
    return embedding
