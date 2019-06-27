import os, random, shutil
import sys

'''
在文件夹中随机抽取图片(探测图)
'''


def moveFile(fileDir, target_person):
    pathDir = os.listdir(fileDir)

    filenumber = len(pathDir)
    rate = 0.1  # 自定义抽取图片的比例，比方说100张抽10张，那就是0.1
    # picknumber = int(filenumber * rate)  # 按照rate比例从文件夹中取一定数量图片
    picknumber = 1
    sample = random.sample(pathDir, picknumber)  # 随机选取picknumber数量的样本图片
    print(sample)
    count = 0
    for name in sample:
        # 目标name使用id命名
        test = target_person + '_' + str(count) + ext
        shutil.copy(os.path.join(fileDir, name), os.path.join(tarDir, test))
        count += 1
    return


if __name__ == '__main__':
    '''
    遍历所有人文件夹，每个文件夹获取一张图片
    '''
    fileDir = "/home/heisai/disk/HeisAI_data/aligned_lfw_50000/"  # 源图片文件夹路径
    # fileDir = sys.argv[1]  # 源文件
    tarDir = '/home/heisai/disk/HeisAI_data/50000/test_pic/'  # 移动到新的文件夹路径
    person_num = 2000
    ext = '.jpg'
    # tarDir = sys.argv[2]  # 目标文件
    print(fileDir, tarDir)
    persons_dir = os.listdir(fileDir)
    for idx, person in enumerate(persons_dir):
        if idx < person_num:
            moveFile(os.path.join(fileDir, person), person)
        else:
            break
