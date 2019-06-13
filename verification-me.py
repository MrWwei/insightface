"""Helper for evaluation on the Labeled Faces in the Wild dataset 
"""

# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import sys
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy import misc
from sklearn.model_selection import KFold
from scipy import interpolate
import sklearn
import cv2
import math
import datetime
import pickle
from sklearn.decomposition import PCA
import mxnet as mx
from mxnet import ndarray as nd

d = os.path.dirname(__file__)
sys.path.append(os.path.join(os.path.dirname(d), 'symbols'))
import fmobilefacenet

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'common'))
import face_image
from collections import namedtuple

Batch = namedtuple('Batch', ['data'])


def get_symbol(args, arg_params, aux_params):
    # data_shape = (args.image_channel, args.image_h, args.image_w)
    data_shape = (3, 112, 112)
    image_shape = ",".join([str(x) for x in data_shape])
    margin_symbols = []
    # print('init mobilefacenet', args.num_layers)
    embedding = fmobilefacenet.get_symbol(args.emb_size, bn_mom=args.bn_mom, version_output=args.version_output)

    all_label = mx.symbol.Variable('softmax_label')
    gt_label = all_label
    extra_loss = None
    _weight = mx.symbol.Variable("fc7_weight", shape=(args.num_classes, args.emb_size), lr_mult=args.fc7_lr_mult,
                                 wd_mult=args.fc7_wd_mult)

    if args.loss_type == 4:
        s = args.margin_s
        m = args.margin_m
        assert s > 0.0
        assert m >= 0.0
        assert m < (math.pi / 2)
        _weight = mx.symbol.L2Normalization(_weight, mode='instance')
        nembedding = mx.symbol.L2Normalization(embedding, mode='instance', name='fc1n') * s
        fc7 = mx.sym.FullyConnected(data=nembedding, weight=_weight, no_bias=True, num_hidden=args.num_classes,
                                    name='fc7')
        zy = mx.sym.pick(fc7, gt_label, axis=1)
        cos_t = zy / s
        cos_m = math.cos(m)
        sin_m = math.sin(m)
        mm = math.sin(math.pi - m) * m
        # threshold = 0.0
        threshold = math.cos(math.pi - m)
        if args.easy_margin:
            cond = mx.symbol.Activation(data=cos_t, act_type='relu')
        else:
            cond_v = cos_t - threshold
            cond = mx.symbol.Activation(data=cond_v, act_type='relu')
        body = cos_t * cos_t
        body = 1.0 - body
        sin_t = mx.sym.sqrt(body)
        new_zy = cos_t * cos_m
        b = sin_t * sin_m
        new_zy = new_zy - b
        new_zy = new_zy * s
        if args.easy_margin:
            zy_keep = zy
        else:
            zy_keep = zy - s * mm
        new_zy = mx.sym.where(cond, new_zy, zy_keep)

        diff = new_zy - zy
        diff = mx.sym.expand_dims(diff, 1)
        gt_one_hot = mx.sym.one_hot(gt_label, depth=args.num_classes, on_value=1.0, off_value=0.0)
        body = mx.sym.broadcast_mul(gt_one_hot, diff)
        fc7 = fc7 + body

    out_list = [mx.symbol.BlockGrad(embedding)]
    softmax = mx.symbol.SoftmaxOutput(data=fc7, label=gt_label, name='softmax', normalization='valid')
    out_list.append(softmax)

    out = mx.symbol.Group(out_list)
    return (out, arg_params, aux_params)


def get_image(url, show=False):
    # download and show the image

    img = cv2.cvtColor(cv2.imread(url), cv2.COLOR_BGR2RGB)
    if img is None:
        return None
    # if show:
    #      plt.imshow(img)
    #      plt.axis('off')
    # convert into format (batch, RGB, width, height)
    img = cv2.resize(img, (112, 112))
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    img = img[np.newaxis, :]
    return img


# def test(data_set, mx_model, batch_size, nfolds=10, data_extra=None, label_shape=None):
#     img = get_image('/home/heisai/Desktop/test1.png')
#     model.forward(Batch([mx.nd.array(img)]), is_train=False)
#     net_out = model.get_outputs()
#     test = np.argmax(net_out)
#     print(test)
# _arg, _aux = model.get_params()
# __arg = {}
# for k,v in _arg.iteritems():
#  __arg[k] = v.as_in_context(_ctx)
# _arg = __arg
# _arg["data"] = _data.as_in_context(_ctx)
# _arg["softmax_label"] = _label.as_in_context(_ctx)
# for k,v in _arg.iteritems():
#  print(k,v.context)
# exe = sym.bind(_ctx, _arg, args_grad=None, grad_req="null", aux_states=_aux)
# exe.forward(is_train=False)
# net_out = exe.outputs
# _embeddings = net_out[0].asnumpy()
# print(_embeddings)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='do verification')
    # general
    parser.add_argument('--data-dir', default='/home/heisai/Heisai/insightface/datasets/test/', help='')
    parser.add_argument('--model', default='/home/heisai/Desktop/model_mxnet_20190527/model,4',
                        help='path to load model.')
    parser.add_argument('--gpu', default=0, type=int, help='gpu id')
    parser.add_argument('--max', default='', type=str, help='')
    parser.add_argument('--mode', default=0, type=int, help='')

    args = parser.parse_args()

    test = os.path.join(os.path.dirname(__file__), '..')
    sym, arg_params, aux_params = mx.model.load_checkpoint('/home/heisai/disk/HeisAI_data/model_20190612/model', 9)
    all_layers = sym.get_internals()
    sym = all_layers['fc1_output']
    # sym, arg_params, aux_params = get_symbol(args, arg_params, aux_params)
    ctx = [mx.cpu()]
    # model = mx.mod.Module(
    #     context=ctx,
    #     symbol=sym,
    # )
    img1 = get_image('/home/heisai/Pictures/img4.jpg')
    img2 = get_image('/home/heisai/Pictures/img3.jpg')
    imgs = [img1, img2]

    # sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
    all_layers = sym.get_internals()
    # sym = all_layers['heatmap_output']
    # image_size = (128, 128)
    # self.image_size = image_size
    model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
    # model = mx.mod.Module(symbol=sym, context=ctx)
    model.bind(for_training=False, data_shapes=[('data', (1, 3, 112, 112))])
    model.set_params(arg_params, aux_params)
    # batch_data = mx.io.DataBatch(eval_batch.data)
    # model.forward(batch_data, is_train=False)

    # path_imgidx = path_imgrec[0:-4] + ".idx"
    # self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
    embedings = []
    for img in imgs:
        model.forward(Batch([mx.nd.array(img)]), is_train=False)
        net_out = model.get_outputs()[0].asnumpy()
        embedings.append(net_out)

    # file = open('./test.txt', 'a')
    # file.writelines(str(net_out))
    # for i in range(len(net_out)):
    #     s = str(net_out[i]).replace('[', '').replace(']', '')
    #     s = s.replace("'", '').replace(',', '') + '\n'
    #     file.write(s)
    # file.close()
    embeding1 = embedings[0]
    embeding2 = embedings[1]
    cosine_dis = cosine_similarity(embeding1, embeding2)
    print(cosine_dis)
