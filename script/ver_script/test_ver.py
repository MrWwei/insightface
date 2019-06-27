import os
import sys
import struct
import matplotlib.pylab as plt
import numpy as np
from sklearn import preprocessing
from tqdm import tqdm

'''
读取bin得到特征，探测库与底库，far、frr、roc
'''
# reg_dir = sys.argv[1]
reg_dir = '/home/heisai/disk/HeisAI_data/50000/total_bin/total/'
# rec_dir = sys.argv[2]
rec_dir = '/home/heisai/disk/HeisAI_data/50000/test_bin/'
# feature_dim = int(sys.argv[3])
feature_dim = 512
# alg_name = sys.argv[4]
alg_name = 'heisai'

feature_ext = 1


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
        feature = preprocessing.normalize(feature.reshape(1, -1)).flatten()
    return feature


def get_top1(feature, reg_all):
    max_score = -1
    max_index = -1
    for i in range(len(reg_all)):
        ft_tt = reg_all[i][1]
        # print('feature: ',feature,'\nreg: ',ft_tt)  
        score = np.dot(feature, ft_tt.T)
        # print(score)
        if score - 1 > 0.1:
            print(score)
            sys.exit()
        if score > max_score:
            max_score = score
            max_index = i
    return reg_all[max_index][0], max_score


def get_top5(feature, reg_all):
    reg_dict = {}
    score_list = []
    for id, ft_tt in reg_all:
        score = np.dot(feature, ft_tt.T)
        # score_list.append((score+1)*50)
        reg_dict[id] = (score + 1) * 0.5

    # 对字典的值进行排序
    sort_dict = sorted(reg_dict.items(), key=lambda item: item[1], reverse=True)

    top5_scores = sort_dict[-5:]
    # 获取5个高分数，如果有一个符合则说明找到
    # {('id1',score1),('id2',socre2)}
    return top5_scores


def get_compare_roc(ret_pair):
    print('total compare times: ', len(ret_pair))
    fpr_list = []
    trp_list = []
    with open('%s_roc.txt' % alg_name, 'w') as fw:
        for i in range(100):
            th = 0.01 * i
            tp = 0  # same accept
            fp = 0  # diff accept
            tn = 0  # diff reject
            fn = 0  # same reject
            for pair in ret_pair:
                # print(pair)
                if pair[0] == pair[1]:
                    if pair[2] >= th:
                        tp += 1
                    else:
                        fn += 1
                else:
                    if pair[2] >= th:
                        fp += 1
                    else:
                        tn += 1
            print(th, tp, fp, tn, fn)
            tpr = tp * 1.0 / (tp + fn)
            fpr = fp * 1.0 / (fp + tn)
            trp_list.append(tpr)
            fpr_list.append(fpr)
            fw.write(str(tpr) + ' ' + str(fpr) + ' ' + str(th) + '\n')

    plt.figure()
    plt.plot(np.array(fpr_list), np.array(trp_list))
    plt.savefig('./%s_roc.png' % alg_name)
    # plt.show()

def top5_accuracy():
    # 将探测库每个特征与底库比对，分数从大到小排序，取前5，判断name是否在前5中，如果在，则说明在库，并且找对。accuracy=判断正确的数量（正例和负例）/探测库数量
    # 输入正例，如果判断在库，则分子++； 输入负例，如果判断不在库，则分子++。
    
    print('top5_accuracy')

def top1_accuracy():
    # 将探测库每个特征与底库比对，分数从大到小排序，取最大分数，判断name和id是否相等，如果相等，则分子++，如果不相等且本来就不在库，则分子++。
    # 读取
    for ft in tqdm(feats):
        feature = load_bin(os.path.join(rec_dir, ft))
        id, score = get_top1(feature, reg_all)
        name = ft.split('_')[0]
        # top5 = get_top5(feature, reg_all)
        ret_pair.append([name, id, score])
    # 遍历ret_pair
    for pair in ret_pair:
                # print(pair)
                if pair[0] == pair[1]:
                    #分子++
                    right_num+=1
                else:
                    #对探测库加个正例负例标签flag 1：在库 0：不在库
                    #不在库,
                    #分子++
                    if flag == 0:
                        right_num+=1
                    print()
    acc = right_num/len(feats)
    print('top1_accuracy')

def get_compare_ret(ret_pair, reg_names):
    far_list = []
    frr_list = []
    th_list = []
    with open('%s_far_frr.txt' % alg_name, 'w') as fw:
        for i in range(100):
            th = 0.01 * i
            ta = 0
            tr = 0
            fa = 0  # false accept
            fr = 0  # false reject
            for pair in ret_pair:
                # print(pair)
                rec = pair[1]
                if rec not in reg_names:
                    if pair[2] > th:
                        fa += 1
                    # else:
                    #     tn+=1
                else:
                    if pair[2] > th:
                        if pair[0] == pair[1]:
                            ta += 1
                        else:
                            fa += 1
                    else:
                        fr += 1
            print(th, fa, fr)
            far = fa * 1.0 / (len(ret_pair))
            frr = fr * 1.0 / (len(ret_pair))
            far_list.append(far)
            frr_list.append(frr)
            th_list.append(th)
            fw.write(str(far) + ' ' + str(frr) + ' ' + str(th) + '\n')

    plt.figure()
    plt.plot(np.array(th_list), np.array(frr_list), label='frr')
    plt.plot(np.array(th_list), np.array(far_list), label='far')
    plt.savefig('./%s_frr_far.png' % alg_name)
    # plt.show()


reg_all = []
# temps_ids = os.listdir(reg_dir)
reg_names = []
# for id in temps_ids:
#     new_path = os.path.join(reg_dir,id)
#     if not os.path.isdir(new_path):
#         continue
feats = os.listdir(reg_dir)
# if len(feats) == 0:
#     continue
print('')
# 导入底库
for ft in tqdm(feats):
    id = ft.split('.')[0]
    reg_names.append(id)
    # print(feats)
    feature = load_bin(os.path.join(reg_dir, ft))
    # reg_all存储底库id、feature
    reg_all.append([id, feature])
print('get reg feature done %d items' % len(reg_all))

ret_pair = []
# 获取底库的特征，获取探测图的特征(底库id，探测库name)
feats = os.listdir(rec_dir)
flag = os.path.join(rec_dir, feats[0])
if os.path.isdir(flag):
    # 遍历探测库
    for id in tqdm(feats):
        new_path = os.path.join(rec_dir, id)
        if not os.path.isdir(new_path):
            continue
        feats = os.listdir(new_path)
        if len(feats) == 0:
            continue
        for ft in feats:
            feature = load_bin(os.path.join(new_path, ft))
            name, score = get_top1(feature, reg_all)
            name = ft.split('_')[0]
            ret_pair.append([name, id, score])

else:
    # 遍历探测库（name，feature），底库（id，feature）
    for ft in tqdm(feats):
        feature = load_bin(os.path.join(rec_dir, ft))
        name = ft.split('_')[0]
        id, score = get_top1(feature, reg_all)
        name = ft.split('_')[0]
        # top5 = get_top5(feature, reg_all)
        ret_pair.append([name, id, score])
# 文件形式的探测库
# temps_ids = os.listdir(rec_dir)

print('get rec feature and compare done %d items' % (len(ret_pair)))

get_compare_roc(ret_pair)
get_compare_ret(ret_pair, reg_names)
top1_accuracy()
top5_accuracy()
