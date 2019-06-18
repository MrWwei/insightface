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
from sklearn import preprocessing

import struct

import argparse

from collections import namedtuple


def read_img(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (112, 112))
    return img


# def get_features(imgs, nets):
#     count = len(imgs)
#     data = mx.nd.zeros(shape=(count * 2, 3, imgs[0].shape[0], imgs[0].shape[1]))
#     for idx, img in enumerate(imgs):
#         img = img[:, :, ::-1]  # to rgb
#         img = np.transpose(img, (2, 0, 1))
#         for flipid in [0, 1]:
#             _img = np.copy(img)
#             if flipid == 1:
#                 _img = _img[:, :, ::-1]
#             _img = nd.array(_img)
#             data[count * flipid + idx] = _img
#
#     F = []
#     for net in nets:
#         db = mx.io.DataBatch(data=(data,))
#         net.model.forward(db, is_train=False)
#         x = net.model.get_outputs()[0].asnumpy()
#         embedding = x[0:count, :] + x[count:, :]
#         embedding = sklearn.preprocessing.normalize(embedding)
#         # print('emb', embedding.shape)
#         F.append(embedding)
#     F = np.concatenate(F, axis=1)
#     F = preprocessing.normalize(F)
#     # F = np.asarray(F)
#     print('F', F.shape)
#     np.save('archieve.npy', F)
#     return


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


# def import_achieve():
#     # 导入底库，
#     model = '/home/heisai/disk/HeisAI_data/model_20190612/model,9'
#     ctx = mx.cpu()
#     nets = []
#     image_shape = [112, 112]
#     # for model in args.model.split('|'):
#     vec = model.split(',')
#     assert len(vec) > 1
#     prefix = vec[0]
#     epoch = int(vec[1])
#     print('loading', prefix, epoch)
#     net = edict()
#     net.ctx = ctx
#     net.sym, net.arg_params, net.aux_params = mx.model.load_checkpoint(prefix, epoch)
#     all_layers = net.sym.get_internals()
#     net.sym = all_layers['fc1_output']
#     net.model = mx.mod.Module(symbol=net.sym, context=net.ctx, label_names=None)
#     net.model.bind(for_training=False, data_shapes=[('data', (1, 3, 112, 112))])
#     net.model.set_params(net.arg_params, net.aux_params)
#     nets.append(net)

#     imgs = []

#     image_dir = '/home/heisai/Pictures/test_rec'
#     # 遍历文件夹
#     imgs_list = os.listdir(image_dir)

#     for img in imgs_list:
#         image_path = os.path.join(image_dir, img)
#         img = read_img(image_path)
#         imgs.append(img)
#     F = get_features(imgs, nets)
#     print('导入底库成功！')


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
        embedding = sklearn.preprocessing.normalize(embedding)
        # print('emb', embedding.shape)
        F.append(embedding)
    F = np.concatenate(F, axis=1)
    F = sklearn.preprocessing.normalize(F)
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


# def get_testFeature(pretrained, img_path):
#     Batch = namedtuple('Batch', ['data'])
#     vec = pretrained.split(',')
#     sym, arg_params, aux_params = mx.model.load_checkpoint(vec[0], int(vec[1]))
#
#     all_layers = sym.get_internals()
#     ctx = mx.cpu()
#     sym = all_layers['fc1_output']
#     model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
#     model.bind(data_shapes=[('data', (1, 3, 112, 112))])
#     model.set_params(arg_params, aux_params)
#
#     img = get_image(img_path)
#     model.forward(Batch([mx.nd.array(img)]), is_train=False)
#     embedding = model.get_outputs()[0].asnumpy()
#
#     embedding = sklearn.preprocessing.normalize(embedding)
#     embedding = preprocessing.normalize(embedding)
#     return embedding


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
    # 导入底库:生成lst，读取lst，加载模型
    # lst_file = os.path.join(os.getcwd(), 'achieve.lst')
    # gen_lst(lst_file)

    i = 0
    succ = 0
    buffer = []
    input_dir = '/home/heisai/Desktop/test1'
    # 遍历每张图片，提取特征并创建对应的bin来存储特征
    for path, dirs, files in os.walk(input_dir):
        for fname in files:
            if not os.path.splitext(fname)[-1][1:] == "xml":

                print(fname)
                image_path = os.path.join(input_dir, path, fname)
                img = read_img(image_path)
                if img is None:
                    print('read error:', image_path)
                    continue
                # bin存储的位置，在新目录里创建一个同名文件夹
                path_bin = os.path.join('/home/heisai/Desktop/test3', os.path.basename(path))
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

    # for line in open(lst_file, 'r'):
    #     if i % 1000 == 0:
    #         print("writing fs", i, succ)
    #     i += 1
    #     image_path = line.strip()
    #     _path = image_path.split('/')
    #     a, b = _path[-2], _path[-1]
    #     out_dir = os.path.join('', a)
    #     if not os.path.exists(out_dir):
    #         os.makedirs(out_dir)
    #     image_path = os.path.join(args.facescrub_root, image_path)
    #     img = read_img(image_path)
    #     if img is None:
    #         print('read error:', image_path)
    #         continue
    #     # bin存储的位置，底库的话随便创建一个文件夹
    #     out_path = os.path.join(out_dir, b.split('.')[0] + "_%s.bin" % (args.algo))
    #     item = (img, out_path)
    #     buffer.append(item)
    #     if len(buffer) == args.batch_size:
    #         get_and_write(buffer, nets)
    #         buffer = []
    #     succ += 1
    # if len(buffer) > 0:
    #     get_and_write(buffer, nets)
    #     buffer = []


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, help='', default=8)
    parser.add_argument('--image_size', type=str, help='', default='3,112,112')
    parser.add_argument('--gpu', type=int, help='', default=0)
    parser.add_argument('--algo', type=str, help='', default='heisai')
    parser.add_argument('--facescrub-lst', type=str, help='', default='./data/facescrub_lst')
    parser.add_argument('--megaface-lst', type=str, help='', default='./data/megaface_lst')
    parser.add_argument('--facescrub-root', type=str, help='', default='./data/facescrub_images')
    parser.add_argument('--megaface-root', type=str, help='', default='./data/megaface_images')
    parser.add_argument('--output', type=str, help='', default='./feature_out')
    parser.add_argument('--model', type=str, help='', default='/home/heisai/model_20190612/model,9')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
