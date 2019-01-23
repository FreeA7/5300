import cv2 as cv
import numpy as np
import random
import os

import datetime


def gaussianThreshold(img, showimg=0):
    # 图片进行二值化
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    binary = cv.adaptiveThreshold(
        gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 15, 10)
    if showimg:
        cv.namedWindow('GAUSSIAN', cv.WINDOW_AUTOSIZE)
        cv.imshow('GAUSSIAN', binary)
    return binary


def getModel():
    # 获取模板
    m1 = cv.imread('./feature/feature1.jpg', cv.IMREAD_GRAYSCALE)

    return m1


def getMore(img, m):
    # 对目标图片进行模板的匹配，返回最大值与位置
    res0 = cv.matchTemplate(img, m, cv.TM_CCOEFF_NORMED)
    min_val0, max_val0, min_loc0, max_loc0 = cv.minMaxLoc(res0)

    return max_val0, max_loc0


def getBest(img, m):
    ''' resize 和返回值
    '''

    if img.shape[:2] == (432, 576):
        img = cv.resize(img, (558, 421), cv.INTER_CUBIC)
    elif img.shape[:2] == (768, 1024):
        img = np.rot90(img)
        img = np.rot90(img)
        img = np.rot90(img)
        img = cv.resize(img, (634, 846), cv.INTER_CUBIC)

    max_val, max_loc = getMore(img, m)

    print('    END:%f' % (max_val))

    if max_val >= 0.4:
        return max_val, max_loc, img
    else:
        return 0, 0, img


def getWhich(img, m):
    # 获取目标图片的最好匹配val和位置loc
    max_val, max_loc, img = getBest(img, m)
    if max_val == 0:
        return 0, 0, img
    # best是指匹配的最好的模板是这个问题的哪一个模板
    return max_val, max_loc, img


def getColor():
    # 画图的时候随机返回一个颜色
    return (random.randint(1, 255), random.randint(1, 255), random.randint(1, 255))


def getUDMirror(ptsx, tl, loc):
    # 将列表中的多边形统一以一个水平轴进行对称
    for key in ptsx.keys():
        ptsx[key] = np.array(
            [[j[0], ((tl[1] + loc) + ((tl[1] + loc) - j[1]))] for j in ptsx[key]])
    return ptsx


def getLRMirror(ptsx, tl, loc):
    # 将列表中的多边形统一以一个垂直轴进行对称
    for key in ptsx.keys():
        ptsx[key] = np.array(
            [[((tl[0] + loc) + ((tl[0] + loc) - j[0])), j[1]] for j in ptsx[key]])
    return ptsx


def getZeroFlag(pts, img):
    # 判断这个多边形是否有任何一个顶点是在图片的像素范围之内
    flag = 0
    h = img.shape[0]
    w = img.shape[1]
    for i in pts:
        if i[0] > 0 and i[1] > 0 and i[0] < w and i[1] < h:
            flag = 1
            break
    return flag


def getMove(pts, gap, hov):
    # 将列表中的多边形统一向水平或者垂直方向移动gap
    if hov:
        return np.array([[i[0], i[1] + gap] for i in pts])
    else:
        return np.array([[i[0] + gap, i[1]] for i in pts])


def getAllTarget(ptsx, img, gap, hov, ptdic, offset=0):
    ''' 获取一个多边形list的一个方向（横向或者纵向）上的所有有点的拷贝
        ---------------------------                ---------------------------
        |                         |                |                         |
        |                         |                |                         |
        |                         |                |                         |
        |           |-|           |       -->      ||-||-||-||-||-||-||-||-| |
        |                         |                |                         |
        |                         |                |                         |
        |                         |                |                         |
        |                         |                |                         |
        ---------------------------                ---------------------------
    '''

    # 获取ptsx原始拷贝
    ptss = ptsx.copy()

    # 画出初始图像
    for key in ptsx.keys():
        pts = ptsx[key]
        ptdic[key].append(pts.copy())
        pts = pts.reshape(-1, 1, 2)
        img = cv.polylines(img, [pts], True, (0, 255, 0))

    # 是否改变方向，0为没有1为有
    changeFlag = 0

    # 偏移修正系数
    if not offset:
        offset_num = 0
    else:
        offset_num = 1

    while 1:

        # 整体移动ptsx
        for key in ptsx.keys():
            # print(gap - offset_num // (offset + 1))
            ptsx[key] = getMove(
                ptsx[key], gap - offset_num // (offset + 1), hov)
        if offset_num > 0:
            offset_num += 1
        elif offset_num < 0:
            offset_num -= 1

        # 定义是否还有多边形存在点，0为没有1为有
        zeroFlag = 0

        # 逐个多边形进行判断并绘点
        for key in ptsx.keys():
            # 判断这个多边形是否有点
            if getZeroFlag(ptsx[key], img):
                pts = ptsx[key]
                pts = pts.reshape(-1, 1, 2)
                img = cv.polylines(img, [pts], True, (255, 0, 0))
                # 只要有一个多边形有点就可以继续移动
                zeroFlag = 1

        # 绘点结束，判断是否要进行下一次移动，如果任何多边形有点则继续移动
        if zeroFlag:
            for key in ptsx.keys():
                ptdic[key].append(ptsx[key].copy())
            continue
        # 所有多边形没点，并且还未改变过移动方向
        elif not changeFlag:
            # 改变移动方向，ptss回归原始拷贝
            changeFlag = 1
            if offset:
                offset_num = -1
            gap = (-1) * gap
            ptsx = ptss
            continue
        # 所有多边形没点并且改变过一次移动方向
        else:
            break
    return img


def get1LR(ptsx, img, hgap, vgap, ptdic, offset=0):
    ''' 获取一个多边形list左右一定gap的上下所有多边形
        ---------------------------                ---------------------------
        |                         |                |   |-|               |-| |
        |                         |                |   |-|               |-| |
        |                         |                |   |-|               |-| |
        |           |-|           |       -->      |   |-| hgap |-| hgap |-| |
        |                         |                |   |-|               |-| |
        |                         |                |   |-|               |-| |
        |                         |                |   |-|               |-| |
        |                         |                |   |-|               |-| |
        ---------------------------                ---------------------------
    '''
    ptss = ptsx.copy()
    for key in ptsx.keys():
        ptsx[key] = getMove(ptsx[key], hgap, 0)
    img = getAllTarget(ptsx, img, vgap, 1, ptdic, offset)

    ptsx = ptss

    for key in ptsx.keys():
        ptsx[key] = getMove(ptsx[key], (-1) * hgap, 0)
    img = getAllTarget(ptsx, img, vgap, 1, ptdic, offset)

    return img


def get1UD(ptsx, img, hgap, vgap, ptdic, offset=0):
    ''' 获取一个多边形list上下一定gap的左右所有多边形
        --------------------------                 --------------------------
        |                        |                 |                        |
        |                        |                 ||-||-||-||-||-||-||-||-||
        |                        |                 |            vgap        |
        |           |-|          |        -->      |            |-|         |
        |                        |                 |            vgap        |
        |                        |                 ||-||-||-||-||-||-||-||-||
        |                        |                 |                        |
        |                        |                 |                        |
        --------------------------                 --------------------------
    '''
    ptss = ptsx.copy()
    for key in ptsx.keys():
        ptsx[key] = getMove(ptsx[key], vgap, 1)
    img = getAllTarget(ptsx, img, hgap, 0, ptdic, offset)

    ptsx = ptss

    for key in ptsx.keys():
        ptsx[key] = getMove(ptsx[key], (-1) * vgap, 1)
    img = getAllTarget(ptsx, img, hgap, 0, ptdic, offset)

    return img


def getTarget(img, tl):

    ptsx = {}
    ptdic = {}

    # M1-1
    ptdic['M1-1'] = []
    ptsx['M1-1'] = np.array([[tl[0] + 19, tl[1] + 174], [tl[0] + 19, tl[1] + 140],
                             [tl[0] + 27, tl[1] + 131], [tl[0] + 40, tl[1] + 131],
                             [tl[0] + 47, tl[1] + 140], [tl[0] + 47, tl[1] + 154],
                             [tl[0] + 458, tl[1] + 154], [tl[0] + 458, tl[1] + 164],
                             [tl[0] + 47, tl[1] + 164], [tl[0] + 47, tl[1] + 174]])

    # M1-2
    ptdic['M1-2'] = []
    ptsx['M1-2'] = np.array([[tl[0] - 23, tl[1] + 78], [tl[0] - 23, tl[1] + 85],
                             [tl[0] + 36, tl[1] + 85], [tl[0] + 36, tl[1] + 78]])

    # M1-3
    ptdic['M1-3'] = []
    ptsx['M1-3'] = np.array([[tl[0] - 3, tl[1] + 110], [tl[0] - 3, tl[1] + 117],
                             [tl[0] + 19, tl[1] + 117], [tl[0] + 19, tl[1] + 110]])

    # M2-1
    ptdic['M2-1'] = []
    ptsx['M2-1'] = np.array([[tl[0] + 28, tl[1] + 137], [tl[0] + 23, tl[1] + 143],
                             [tl[0] + 22, tl[1] + 138], [tl[0] + 15, tl[1] + 138],
                             [tl[0] + 15, tl[1] + 153], [tl[0] + 1, tl[1] + 153],
                             [tl[0] + 1, tl[1] + 17], [tl[0] + 15, tl[1] + 17],
                             [tl[0] + 15, tl[1] + 130], [tl[0] + 24, tl[1] + 130]])

    # M2-2
    ptdic['M2-2'] = []
    ptsx['M2-2'] = np.array([[tl[0] + 23, tl[1] + 167], [tl[0] + 28, tl[1] + 167],
                             [tl[0] + 28, tl[1] + 144], [tl[0] + 31, tl[1] + 141],
                             [tl[0] + 34, tl[1] + 141], [tl[0] + 38, tl[1] + 144],
                             [tl[0] + 38, tl[1] + 167], [tl[0] + 43, tl[1] + 167],
                             [tl[0] + 43, tl[1] + 143], [tl[0] + 38, tl[1] + 137],
                             [tl[0] + 28, tl[1] + 137], [tl[0] + 23, tl[1] + 143]])

    # M2-3
    ptdic['M2-3'] = []
    ptsx['M2-3'] = np.array([[tl[0] + 31, tl[1] + 144], [tl[0] + 35, tl[1] + 144],
                             [tl[0] + 35, tl[1] + 179], [tl[0] + 39, tl[1] + 179],
                             [tl[0] + 39, tl[1] + 193], [tl[0] + 24, tl[1] + 193],
                             [tl[0] + 24, tl[1] + 179], [tl[0] + 31, tl[1] + 179]])

    # Main
    ptdic['Main'] = []
    ptsx['Main'] = np.array([[tl[0], tl[1]], [tl[0], tl[1] + 152],
                             [tl[0] + 454, tl[1] + 152], [tl[0] + 454, tl[1]]])

    # add
    ptdic['add'] = []
    ptsx['add'] = np.array([[tl[0] - 29, tl[1] + 83], [tl[0] - 29, tl[1] + 155],
                            [tl[0] + 53, tl[1] + 155], [tl[0] + 53, tl[1] + 83]])

    hgap = 454
    vgap = 152
    offset = 0
    '''
      变换思路：
      1.获取所有纵向图像
      2.获取横向2h距离的所有纵向图像
      3.原始图像进行上下对称
      4.获取对称图像横向h距离的所有纵向图像
    '''
    ptss = ptsx.copy()
    img = getAllTarget(ptsx, img, hgap, 0, ptdic, offset)

    ptsx = ptss.copy()
    img = get1UD(ptsx, img, hgap, 2 * vgap, ptdic, offset)

    ptsx = ptss.copy()
    img = get1UD(ptsx, img, hgap, 4 * vgap, ptdic, offset)

    ptsx = ptss.copy()
    img = get1UD(ptsx, img, hgap, 6 * vgap, ptdic, offset)

    ptsx = ptss.copy()
    img = get1UD(ptsx, img, hgap, 8 * vgap, ptdic, offset)

    ptsx = getLRMirror(ptss, tl, 234)
    ptss = ptsx.copy()

    img = get1UD(ptsx, img, hgap, vgap, ptdic, offset)

    ptsx = ptss.copy()
    img = get1UD(ptsx, img, hgap, 3 * vgap, ptdic, offset)

    ptsx = ptss.copy()
    img = get1UD(ptsx, img, hgap, 5 * vgap, ptdic, offset)

    ptsx = ptss.copy()
    img = get1UD(ptsx, img, hgap, 7 * vgap, ptdic, offset)

    return img, ptdic


def getCoordinate(img, getimg=0, showimg=0):
    # 获取图像的所有零件位置

    # 1、保存原始未处理图像便于画图
    oimg = img.copy()

    # 2、或者模板并对图像进行二值化处理
    m = getModel()
    img = gaussianThreshold(img)

    print('进行匹配：')
    # 3、获取是q1还是q2问题以及最好匹配位置
    max_val, max_loc, img = getWhich(img, m)

    # return 没有一个大于0.75的匹配
    if max_val == 0:
        print('Error')
        if showimg:
            cv.namedWindow("match", cv.WINDOW_AUTOSIZE)
            cv.imshow("match", oimg)
        if not getimg:
            return 0
        else:
            return [0, oimg]

    # 4、如果二值化的图片有过拉伸处理这里对要进行画图的原图也进行同样的处理
    if oimg.shape[:2] == (432, 576):
        oimg = cv.resize(oimg, (558, 421), cv.INTER_CUBIC)
    elif oimg.shape[:2] == (768, 1024):
        oimg = np.rot90(oimg)
        oimg = np.rot90(oimg)
        oimg = np.rot90(oimg)
        oimg = cv.resize(oimg, (634, 846), cv.INTER_CUBIC)

    # 5、获取最佳位置以及模板大小，并把最佳匹配在图中画出来
    th, tw = m.shape[:2]
    tl = max_loc
    br = (tl[0] + tw, tl[1] + th)
    cv.rectangle(oimg, tl, br, (0, 0, 255), 1)

    # 6、由最佳匹配点找到pix起始点
    tl = [tl[0] + 37, tl[1] + 23]

    # 7、根据q1还是q2获取所有零件位置并在图中画出来
    oimg, ptdic = getTarget(oimg, tl)

    # 8、图片的展示和返回
    if showimg:
        cv.namedWindow("match", cv.WINDOW_AUTOSIZE)
        cv.imshow("match", oimg)
    if not getimg:
        # 返回多边形dic、图片形状、q1还是q2
        return ptdic
    else:
        return [1, oimg]


def getOverlapping(ptss, target, shape):
    im1 = np.zeros(shape, dtype=np.uint8)
    for pts in ptss:
        im1 = cv.fillConvexPoly(im1, pts, 1)

    im2 = np.zeros(shape, dtype=np.uint8)
    target = cv.fillConvexPoly(im2, target, 1)

    # if target.shape[:2] == (432, 576):
    #     target = cv.resize(target, (558, 421), cv.INTER_CUBIC)
    # elif target.shape[:2] == (768, 1024):
    #     target = np.rot90(target)
    #     target = np.rot90(target)
    #     target = np.rot90(target)
    #     target = cv.resize(target, (634, 846), cv.INTER_CUBIC)
    # target = target // 255

    img = im1 + target

    if (img > 1).any():
        return 1
    else:
        return 0


def getReturn(m1, m2, m):
    result = {}
    result['AFFECTEDPIXELNUM'] = m
    if not m1:
        result['ISM1OPEN'] = False
        result['M1M1'] = False
    else:
        result['ISM1OPEN'] = True
        if m1 > 1:
            result['M1M1'] = True
        else:
            result['M1M1'] = False
    if not m2:
        result['ISM2OPEN'] = False
        result['M2M2'] = False
    else:
        result['ISM2OPEN'] = True
        if m2 > 1:
            result['M2M2'] = True
        else:
            result['M2M2'] = False
    if m1 > 0 and m2 > 0:
        result['M1M2'] = True
    else:
        result['M1M2'] = False
    return result


def getQOut(ptdic, target, shape):
    sum_m1 = 0
    sum_m2 = 0
    sum_m = 0

    for i in range(len(ptdic['Main'])):
        m1 = []
        for key in ['M1-1', 'M1-2', 'M1-3']:
            m1.append(ptdic[key][i])
        if getOverlapping(m1, target, shape):
            sum_m1 += 1

    for i in range(len(ptdic['Main'])):
        m2 = []
        for key in ['M2-1', 'M2-2', 'M2-3']:
            m2.append(ptdic[key][i])
        if getOverlapping(m2, target, shape):
            sum_m2 += 1

    for i in range(len(ptdic['Main'])):
        m = []
        for key in ['Main']:
            m.append(ptdic[key][i])
        if getOverlapping(m, target, shape):
            sum_m += 1

    return getReturn(sum_m1, sum_m2, sum_m)


def getJPG(path, li=0):
    # 返回一个文件夹下所有jpg文件名
    list_name = []
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if file[-3:].lower() == 'jpg' and not os.path.isdir(file_path):
            if not li:
                list_name.append(file_path)
            else:
                list_name.append([path, file])
        if os.path.isdir(file_path):
            list_name += getJPG(file_path, li)
    return list_name


# -------------------------- 处理指定path下所有图片并展示 --------------------------
# path = './testp/432x576/'
# # path = '../3350/testp/tp/q2/'
# for i in getJPG(path):
#     start = datetime.datetime.now()
#     dic = getCoordinate(cv.imread(i), showimg=1)
#     end = datetime.datetime.now()
#     print(i)
#     print('    本次匹配费时%fs:' % (((end - start).microseconds) / 1e6))
#     cv.waitKey(0)
#     cv.destroyAllWindows()

# cv.waitKey(0)
# cv.destroyAllWindows()


# -------------------------- 处理指定图片并展示 --------------------------
# img = './testp/432x576/TPDS0_5300_TA8B2553AA_TAAOL8C0_7_917.649_-1056.08__S_20181201_070454.jpg'
# start = datetime.datetime.now()
# dic = getCoordinate(cv.imread(img), showimg=1)
# # cv.imwrite('./testp/432x576/TPDS0_5300_TA8B2553AA_TAAOL8C0_7_917.649_-1056.08__S_20181201_070454.jpg', out[1])
# end = datetime.datetime.now()
# print('本次匹配费时%fs:' % (((end - start).microseconds) / 1e6))

# cv.waitKey(0)
# cv.destroyAllWindows()


# -------------------------- 输出单个重叠情况的测试 --------------------------
# img = './testp/432x576/TPDS0_5300_TA8B2553AA_TAAOL8C0_7_917.649_-1056.08__S_20181201_070454.jpg'
# img = cv.imread(img)
# shape = img.shape
# start = datetime.datetime.now()
# dic = getCoordinate(img, showimg=1)

# target = np.array([[0, 0], [0, 10], [10, 10], [10, 0]])

# re = getQOut(dic, target, shape)

# end = datetime.datetime.now()
# print('单次比较费时%fs:' % (((end - start).microseconds) / 1e6))
# print(re)

# cv.waitKey(0)
# cv.destroyAllWindows()


# -------------------------- 大量测试输出重叠情况的速度^~^ --------------------------
# start = datetime.datetime.now()
# s = 0

# for img in getJPG('./testp/432x576/'):
#     s += 1
#     dic = getCoordinate(cv.imread(img))
#     img = cv.imread(img)
#     shape = img.shape
#     if not dic:
#         continue
#     target = np.array([[0, 0], [0, 100], [100, 100], [100, 0]])
#     re = getQOut(dic, target, shape)
#     print(re)

# end = datetime.datetime.now()
# all_time = (end - start).seconds + (((end - start).microseconds) / 1e6)
# one_time = all_time / s

# print('共处理%d张图片，共费时%fs，平均每张图片费时%fs' % (s, all_time, one_time))
