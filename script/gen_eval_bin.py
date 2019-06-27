# coding:utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import mxnet as mx
import numpy as np
import sklearn
from mxnet import nd
import cv2
from easydict import EasyDict as edict
import datetime

from sklearn import preprocessing
import struct

import argparse

'''
测试模型，生成bin，用于存储图片的特征向量(探测库和底库)
'''


def read_img(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (112, 112))
    return img


def get_dataset_common(input_dir):
    # print('get_dataset_common')
    ret = []

    for img in os.listdir(input_dir):
        fimage = edict()
        fimage.image_path = os.path.join(input_dir, img)
        ret.append(fimage)
    return ret


def gen_lst(lst_file):
    count = 0
    dataset = get_dataset_common('/home/heisai/Pictures/test_rec')
    with open(lst_file, "w") as text_file:
        for cls in dataset:
            count += 1
            text_file.write('%s\n' % cls.image_path)
    print(count)


def get_and_write(buffer, nets):
    imgs = []
    for k in buffer:
        imgs.append(k[0])
    features = get_feature(imgs, nets)
    # print(np.linalg.norm(feature))
    assert features.shape[0] == len(buffer)
    for ik, k in enumerate(buffer):
        out_path = k[1]
        feature = features[ik].flatten()
        write_bin(out_path, feature)
        load_bin(out_path)


def load_achieve(archieve_path):
    F = np.load(archieve_path)
    return F


feature_dim = 512
feature_ext = 0


def load_bin(path, fill=0.0):
    with open(path, 'rb') as f:
        bb = f.read(4 * 4)
        # print(len(bb))
        v = struct.unpack('4i', bb)
        # print(v[0])
        bb = f.read(v[0] * 4)
        v = struct.unpack("%df" % (v[0]), bb)
        feature = np.full((feature_dim + feature_ext,), fill, dtype=np.float32)
        feature[0:feature_dim] = v
        # feature = np.array( v, dtype=np.float32)
    # print(feature.shape)
    # print(np.linalg.norm(feature))
    return feature


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
        embedding = preprocessing.normalize(embedding)
        # print('emb', embedding.shape)
        F.append(embedding)
    F = np.concatenate(F, axis=1)
    F = preprocessing.normalize(F)
    # print('F', F.shape)
    return F


def write_bin(path, feature):
    feature = list(feature)
    with open(path, 'wb') as f:
        f.write(struct.pack('4i', len(feature), 1, 4, 5))
        f.write(struct.pack("%df" % len(feature), *feature))


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


def main(args):
    nets = []
    ctx = mx.cpu()
    image_shape = [int(x) for x in args.image_size.split(',')]
    # 加载模型
    for model in args.model.split('|'):
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
        net.model.bind(data_shapes=[('data', (1, 3, image_shape[1], image_shape[2]))])
        net.model.set_params(net.arg_params, net.aux_params)
        nets.append(net)

    succ = 0

    buffer = []
    # 遍历每张图片，提取特征并创建对应的bin来存储特征
    time_a = datetime.datetime.now()
    for path, dirs, files in os.walk(args.input_dir):
        for fname in files:
            if not os.path.splitext(fname)[-1][1:] == "xml":
                print(fname)
                image_path = os.path.join(args.input_dir, path, fname)
                img = read_img(image_path)
                if img is None:
                    print('read error:', image_path)
                    continue
                # bin存储的位置，在新目录里创建一个同名文件夹
                path_bin = os.path.join(args.output_dir, os.path.basename(path))
                if not os.path.exists(path_bin):
                    os.makedirs(path_bin)
                out_path = os.path.join(path_bin, "%s_%s.bin" % (fname, args.algo))
                item = (img, out_path)
                buffer.append(item)
                if len(buffer) == args.batch_size:
                    get_and_write(buffer, nets)
                    buffer = []
                succ += 1
        if len(buffer) > 0:
            get_and_write(buffer, nets)
            buffer = []
    time_b = datetime.datetime.now()
    diff = time_b - time_a
    print('time cost', diff.total_seconds() / 200)
    print('total number is %d' % succ)


def parse_arguments(argv):
    parser = argparse.ArgumentParser(description='generate bin')

    parser.add_argument('--batch-size', type=int, help='', default=8)
    parser.add_argument('--image-size', type=str, help='', default='3,112,112')

    parser.add_argument('--algo', type=str, help='', default='heisai')
    parser.add_argument('--input-dir', type=str, help='', default='/home/heisai/disk/HeisAI_data/aligned_lfw_50000/')
    parser.add_argument('--output-dir', type=str, help='',
                        default='/home/heisai/disk/HeisAI_data/50000/total_aligned/total_bin/')
    parser.add_argument('--model', type=str, help='', default='/home/heisai/model_20190612/model,9')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
