import os
import sys
import struct
import matplotlib.pylab as plt
import numpy as np
from sklearn import preprocessing
import codecs
from tqdm import tqdm

reg_dir = sys.argv[1]
rec_dir = sys.argv[2]
feature_dim = int(sys.argv[3])
alg_name = sys.argv[4]

feature_ext = 1
def load_bin(path, fill = 0.0):
  with open(path, 'rb') as f:
    bb = f.read(4*4)
    # print(len(bb))
    v = struct.unpack('4i', bb)
    # print(v[0])
    bb = f.read(v[0]*4)
    v = struct.unpack("%df"%(v[0]), bb)
    feature = np.full( (feature_dim+feature_ext,), fill, dtype=np.float32)
    feature[0:feature_dim] = v
    feature = preprocessing.normalize(feature.reshape(1, -1)).flatten()
  return feature

def get_top1(feature,reg_all):
    max_score = -1
    max_index = -1
    for i in range(len(reg_all)):
        ft_tt = reg_all[i][1]
        # print('feature: ',feature,'\nreg: ',ft_tt)  
        score = np.dot(feature, ft_tt.T)
        # print(score)
        if score-1 > 0.1:

            print(score)
            sys.exit()
        if score > max_score:
            max_score = score
            max_index = i
    return reg_all[max_index][0],max_score

def get_compare_roc(ret_pair):
    print('total compare times: ',len(ret_pair))
    fpr_list = []
    trp_list = []
    with open('%s_roc.txt'%alg_name,'w') as fw:
        for i in range(100):
            th = 0.01*i
            tp=0  # same accept
            fp=0  # diff accept
            tn=0  # diff reject
            fn=0  # same reject
            for pair in ret_pair:
                # print(pair)
                if pair[0] == pair[1]:
                    if pair[2] >= th:
                        tp+=1
                    else:
                        fn+=1
                else:
                    if pair[2] >= th:
                        fp+=1
                    else:
                        tn+=1
            print(th,tp,fp,tn,fn)
            tpr = tp*1.0/(tp+fn)
            fpr = fp*1.0/(fp+tn)
            trp_list.append(tpr)
            fpr_list.append(fpr)
            fw.write(str(tpr)+' '+str(fpr)+' '+str(th)+'\n')
    
    plt.figure()
    plt.plot(np.array(fpr_list),np.array(trp_list))
    plt.savefig('./%s_roc.png'%alg_name)
    # plt.show()

def get_compare_ret(ret_pair,reg_names):
    far_list = []
    frr_list = []
    th_list = []
    with open('%s_far_frr.txt'%alg_name,'w') as fw:
        for i in range(100):
            th = 0.01*i
            ta=0
            tr=0
            fa = 0   # false accept
            fr = 0   # false reject
            for pair in ret_pair:
                # print(pair)
                rec=pair[1]
                if rec not in reg_names:
                    if pair[2] > th:
                        fa+=1
                    # else:
                    #     tn+=1
                else:
                    if pair[2] > th:
                        if pair[0] == pair[1]:
                            ta+=1
                        else:
                            fa+=1
                    else:
                        fr+=1
            print(th,fa,fr)
            far = fa*1.0/(len(ret_pair))
            frr = fr*1.0/(len(ret_pair))
            far_list.append(far)
            frr_list.append(frr)
            th_list.append(th)
            fw.write(str(far)+' '+str(frr)+' '+str(th)+'\n')
    
    plt.figure()
    plt.plot(np.array(th_list),np.array(frr_list),label='frr')
    plt.plot(np.array(th_list),np.array(far_list),label='far')
    plt.savefig('./%s_frr_far.png'%alg_name)
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
for ft in tqdm(feats):
    id=ft.split('.')[0]
    reg_names.append(id)
    # print(feats)
    feature = load_bin(os.path.join(reg_dir,ft))
    reg_all.append([id,feature])
print('get reg feature done %d items'%len(reg_all))

ret_pair = []
temps_ids = os.listdir(rec_dir)
for id in tqdm(temps_ids):
    new_path = os.path.join(rec_dir,id)
    if not os.path.isdir(new_path):
        continue
    feats = os.listdir(new_path)
    if len(feats) == 0:
        continue
    for ft in feats:
        feature = load_bin(os.path.join(new_path,ft))
        name,score = get_top1(feature,reg_all)
        ret_pair.append([name,id,score])
print('get rec feature and compare done %d items'%(len(ret_pair)))

get_compare_roc(ret_pair)
get_compare_ret(ret_pair,reg_names)


