# coding:utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import mxnet as mx
import numpy as np
import sklearn
from mxnet import nd
import cv2
from easydict import EasyDict as edict
from sklearn import preprocessing

def read_img(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (112, 112))
    return img


def get_feature(imgs, nets):
    count = len(imgs)
    data = mx.nd.zeros(shape=(count * 2, 3, imgs[0].shape[0], imgs[0].shape[1]))
    for idx, img in enumerate(imgs):
        img = img[:, :, ::-1]  # to rgb
        img = np.transpose(img, (2, 0, 1))
        for flipid in [0, 1]:
            _img = np.copy(img)
            if flipid == 1:
                _img = _img[:, :, ::-1]
            _img = nd.array(_img)
            data[count * flipid + idx] = _img

    F = []
    for net in nets:
        db = mx.io.DataBatch(data=(data,))
        net.model.forward(db, is_train=False)
        x = net.model.get_outputs()[0].asnumpy()
        embedding = x[0:count, :] + x[count:, :]
        embedding = sklearn.preprocessing.normalize(embedding)
        # print('emb', embedding.shape)
        F.append(embedding)
    F = np.concatenate(F, axis=1)
    F = preprocessing.normalize(F)
    # print('F', F.shape)
    return F


if __name__ == '__main__':

    model = '/home/heisai/disk/HeisAI_data/model_20190612/model,9'
    ctx = mx.cpu()
    nets = []
    image_shape = [112, 112]
    # for model in args.model.split('|'):
    vec = model.split(',')
    assert len(vec) > 1
    prefix = vec[0]
    epoch = int(vec[1])
    print('loading', prefix, epoch)
    net = edict()
    net.ctx = ctx
    net.sym, net.arg_params, net.aux_params = mx.model.load_checkpoint(prefix, epoch)
    all_layers = net.sym.get_internals()
    net.sym = all_layers['fc1_output']
    net.model = mx.mod.Module(symbol=net.sym, context=net.ctx, label_names=None)
    net.model.bind(for_training=False, data_shapes=[('data', (1, 3, 112, 112))])
    net.model.set_params(net.arg_params, net.aux_params)
    nets.append(net)

    imgs = []

    buffer = []
    image_dir = '/home/heisai/Pictures/test_rec'
    # 遍历文件夹
    imgs_list = os.listdir(image_dir)

    for img in imgs_list:
        image_path = os.path.join(image_dir, img)
        img = read_img(image_path)
        imgs.append(img)

    F = get_feature(imgs, nets)
