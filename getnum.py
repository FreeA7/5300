from getit2 import getJPG
from getit2 import gaussianThreshold
import cv2 as cv
import os
import random
import shutil
import numpy as np


def clearDir(path):
    if not os.path.exists(path):
        os.mkdir(path)

    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        os.remove(file_path)

# ------------------------- 获取所有图片 -------------------------
path = './5300All/'
jpg_list = []
for file in os.listdir(path):
    file_path = os.path.join(path, file)
    if os.path.isdir(file_path):
        '''
        TSFIX:半截
        TSDFS:糊
        TGOTS:整体变色
        '''
        if file in ['TSFIX', 'TSDFS', 'TGOTS']:
            continue
        jpg_list += getJPG(file_path, li=1)
print('一共有%d张照片' % len(jpg_list))
# 一共有16267张照片
# 一共有15420张照片

# # ------------------------- 随机抽取图片 -------------------------
m = 0
clearDir('./testp/432x576/')
clearDir('./testp/576x768/')
clearDir('./testp/768x1024/')
for i in jpg_list:
    img = cv.imread(os.path.join(i[0], i[1]))
    if random.randint(1, 100) < 2:
        f_name = './testp/%dx%d/%s_%s' % (
            img.shape[0], img.shape[1], i[0][-5:], i[1])
        shutil.copyfile(os.path.join(i[0], i[1]), f_name)
        m += 1
        print('Get %d, %s' % (m, f_name))



# # ------------------------- 测试高斯化 -------------------------
# img = cv.imread(
#     './testp/576x768/TPPPP_5300_TA8C1162BA_TAAOL9C0_3_-90.284_-22.096__S_20181216_050254.jpg')

# img = gaussianThreshold(img, showimg=1)

# cv.imwrite('./binary.jpg', img)

# cv.waitKey(0)
# cv.destroyAllWindows()

# # shape[:2] == [768, 1024]
# img = np.rot90(img)
# img = cv.resize(img, (634, 846))

# # shape[:2] == [576, 432]
# img = cv.resize(img, (558, 421))



'''
432x576     156 469   0.9743589743589744 0.9680170575692964 421 558
576x768     152 454
768x1024    184 550   1024x768  0.8260869565217391  0.8254545454545455  846 634
'''
