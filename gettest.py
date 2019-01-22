import os
import shutil
import random
import cv2 as cv
from getit2 import getJPG, getCoordinate
from getnum import clearDir


clearDir('./sampTestPic')

path = './5300All/'
jpg_list = []
dic_dir = {}
for file in os.listdir(path):
    file_path = os.path.join(path, file)
    if os.path.isdir(file_path):
        '''
        TSFIX:半截
        TSDFS:糊
        TGOTS:整体变色
        '''
        # if file in ['TSFIX', 'TSDFS', 'TGOTS']:
        #     continue
        os.mkdir('./sampTestPic/%s' % file)
        os.mkdir('./sampTestPic/%s/input' % file)
        os.mkdir('./sampTestPic/%s/output' % file)
        if file not in dic_dir.keys():
            dic_dir[file] = []
        jpgs = getJPG(file_path, li=1)
        for jpg in jpgs:
            dic_dir[jpg[0][-5:]].append(jpg)
print('一共有%d张照片' % len(jpg_list))


sum_dic = {}
for d in dic_dir.keys():
    print('开始处理%s文件夹，共%d张照片' % (d, len(dic_dir[d])))
    samp_sum = 0
    for jpg in dic_dir[d]:
        if len(dic_dir[d]) < 11:
            samp_sum += 1
            print('    图片%d' % samp_sum)
            target = './sampTestPic/%s/input/%s' % (d, jpg[1])
            shutil.copyfile(os.path.join(jpg[0], jpg[1]), target)
            [flag, img] = getCoordinate(cv.imread(target), getimg=1, showimg=0)
            cv.imwrite('./sampTestPic/%s/output/%s' % (d, jpg[1]), img)
        elif random.randint(1, len(dic_dir[d])) < 16:
            samp_sum += 1
            print('    图片%d' % samp_sum)
            target = './sampTestPic/%s/input/%s' % (d, jpg[1])
            shutil.copyfile(os.path.join(jpg[0], jpg[1]), target)
            [flag, img] = getCoordinate(cv.imread(target), getimg=1, showimg=0)
            cv.imwrite('./sampTestPic/%s/output/%s' % (d, jpg[1]), img)
    print('%s文件夹一共%d张照片，抽取了%d张' % (d, len(dic_dir[d]), samp_sum))
    sum_dic[d] = samp_sum

for d in dic_dir.keys():
    print('%s\t%d' % (d, sum_dic[d]))
