# Copyright (c) 2025 UUI-D
# Licensed under the MIT License. See LICENSE for details.
import os
import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import AgglomerativeClustering
import pickle  # 保存文件的头文件
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.cluster import KMeans
from pandas.core.frame import DataFrame
from skimage import io
import copy
import scipy.signal
from scipy import signal   #滤波等
import torch

area = 0
P_m = 0
null = None
area_zong = 0
colors = sns.color_palette('hls', 16)
elem_count = 0
ttk=8

# tiff转png
def tiffread(infilename, outpath):
    img = cv2.imread(infilename, -1)
    # 路径和文件名
    splitpath = os.path.split(infilename)
    # 文件名和扩展名
    splitfile = os.path.splitext(splitpath[1])
    filename = splitfile[0]
    # 保存中间过程图
    # plt.imsave(''+outpath+"/"+filename+".png",img,cmap="gray")#相对路径
    plt.imsave(outpath + "/" + filename + ".png", img, cmap="gray")  # 绝对路径，需要自己改


def multyfiel(path):
    listfile = []
    for filename in os.listdir(path):
        listfile += [filename]
    return listfile


# 找峰值点
def findpeaks(fielpath):
    img = cv2.imread(fielpath, 0)
    # 横纵坐标xy，d（list）是x中包含数据
    y, x, d = plt.hist(img.ravel(), 256)
    # plt.show()

    peaks = scipy.signal.find_peaks_cwt(y, 5)
    countp = len(peaks)
    '''#数据可视化输出
    print(len(peaks))'''
    print(peaks)

    '''#直方图显示
    plt.plot(y)
    plt.plot(peaks, b[peaks], "o")
    plt.title("峰值")
    plt.show()'''
    return countp,peaks

def find_g(path):
    # 读入图像
    hd_list_max = list()
    hd_list_min = list()
    # 读入图片并转化为矩阵
    img = plt.imread(path)

    x = np.array(img)
    x = x.flatten()  # 转化成一维数组

    # print(x)
    # print(x.max())
    # print(x.min())
    # 掩码去掉x=0和x=1的背景等
    mask0 = x != 0
    b = x[mask0]
    print(b)
    x = b
    mask1 = x != 1
    c = b[mask1]
    x = c
    # [0,1]化成[0,255]
    x = x * 255
    # print("x",x)
    # 将一维数组化成30分，统计每一份的灰度数量，其中m为长度为30的一维数组，每个数字代表这个灰度处于第几组
    m = pd.cut(x, 255, labels=False)
    # print("m",m)
    # print("len_m",len(m))
    # print("max_m",max(m))
    # print("min_m",min(m))
    # print(type(m))
    # mask为0-29
    mask = np.unique(m)
    mask = list(range(0, 255, 1))
    # print("mask",mask)
    # print("len_mask",len(mask))
    # print("max_mask",max(mask))
    # print("min_mask",min(mask))
    # tmp为每组灰度值的数量
    tmp = []
    for v in mask:
        tmp.append(np.sum(m == v))
    # print("tmp",tmp)
    # print("len_tmp",len(tmp))
    # print("max_tmp",max(tmp))
    # print("min_tmp",min(tmp))
    # 图像横坐标y为[0,30]->其实也可以写成0-255中间间隔多少
    y = list(range(0, 255, 1))
    # print(y)
    # print("tmp",tmp)
    # print("len_tmp",len(tmp))
    # print("max_tmp",max(tmp))
    # print("min_tmp",min(tmp))
    # print("y",y)
    # print("len_y",len(y))
    # print("max_y",max(y))
    # print("min_y",min(y))
    plt.plot(y, tmp)

    xxx = y
    yyy = tmp

    z1 = np.polyfit(xxx, yyy, 30)  # 用7次多项式拟合
    p1 = np.poly1d(z1)  # 多项式系数
    print(p1)  # 在屏幕上打印拟合多项式
    yvals = p1(xxx)

    plt.plot(xxx, yyy, '*', label='original values')
    plt.plot(xxx, yvals, 'r', label='polyfit values')
    plt.xlabel('x axis')
    plt.ylabel('y axis')
    plt.legend(loc=4)
    plt.title('polyfitting')
    plt.show()

    # 直接函数分别求极大值和极小值：signal.argrelextrema 函数
    print(yvals[signal.argrelextrema(yvals, np.greater)])  # 极大值的y轴, yvals为要求极值的序列
    print(signal.argrelextrema(yvals, np.greater))  # 极大值的x轴
    peak_ind = signal.argrelextrema(yvals, np.greater)[0]  # 极大值点，改为np.less即可得到极小值点
    plt.plot(xxx, yyy, '*', label='original values')
    plt.plot(xxx, yvals, 'r', label='polyfit values')
    plt.xlabel('x axis')
    plt.ylabel('y axis')
    plt.legend(loc=4)
    plt.title('polyfitting')
    plt.plot(signal.argrelextrema(yvals, np.greater)[0], yvals[signal.argrelextrema(yvals, np.greater)], 'o',
             markersize=10)  # 极大值点
    plt.plot(signal.argrelextrema(yvals, np.less)[0], yvals[signal.argrelextrema(yvals, np.less)], '+',
             markersize=10)  # 极小值点
    plt.show()
    hd_list_max = signal.argrelextrema(yvals, np.greater)[0].tolist()
    hd_list_max = list(map(int, hd_list_max))
    print(hd_list_max)
    for i in hd_list_max[:]:  # 拷贝列表
        if i < 65:
            hd_list_max.remove(i)
        elif i > 230:
            hd_list_max.remove(i)
    hd_list_max_huidu = [i for i in hd_list_max]

    hd_list_min = signal.argrelextrema(yvals, np.less)[0].tolist()
    hd_list_min = list(map(int, hd_list_min))
    for i in hd_list_min[:]:#拷贝列表
        if i < 40:
            hd_list_min.remove(i)
        elif i > 230:
            hd_list_min.remove(i)
    hd_list_min_huidu = [i for i in hd_list_min]
    print("极大值：")
    print("x：", hd_list_max, "y", yvals[signal.argrelextrema(yvals, np.greater)])
    print("灰度：", hd_list_max_huidu)
    print("极小值：")
    print("x：", hd_list_min, "y", yvals[signal.argrelextrema(yvals, np.less)])
    print("灰度：", hd_list_min_huidu)

    print("最终灰度极大值数：", len(hd_list_max), "最终灰度极大值：", hd_list_max_huidu)
    print("最终灰度极小值数：", len(hd_list_min), "最终灰度极小值：", hd_list_min_huidu)
    x=len(hd_list_max)
    peaks=hd_list_max_huidu
    peaks_less=hd_list_min_huidu
    return x,peaks,peaks_less


# 去除杂点
def extract(filepath):
    # 新建文件夹
    # lastpath = os.path.dirname(filepath)
    # chulifilepath=r''+lastpath+"/"+"blackbg"
    chulifilepath = r'blackbg_mdf'
    exist = os.path.exists(chulifilepath)
    if not exist:
        os.makedirs(chulifilepath)

    # 遍历文件夹里的文件修改
    for filename in os.listdir(filepath):
        pic = Image.open(r'' + filepath + '\\' + filename)
        pic = pic.convert('RGBA')  # 转为RGB模式
        width, height = pic.size
        array = pic.load()  # 获取图片像素操作入口
        for i in range(width):
            for j in range(height):
                pos = array[i, j]  # 获得某个像素点，格式为(R,G,B,A)元组
                # 判断是不是目标颜色
                # 黑色0 白色255
                notColor = (sum([1 for x in pos[0:3] if x <= 30]) == 3)
                if notColor:
                    # 更改为目标颜色
                    array[i, j] = (0, 0, 0, 255)
        # 保存文件
        # 文件名和扩展名
        splitfile = os.path.splitext(filename)
        filename = splitfile[0]
        pic.save(chulifilepath + "/" + filename + ".png")


# 分灰度
def step1_threshold(in_path, out_path, x=6):
    grey_all_n = [[0 for col in range(1)] for row in range(7)]
    # 读取原始图像灰度颜色
    img = cv2.imread(in_path, 0)
    # plt.imsave("huidu.png",img,cmap='gray')
    # print(img.shape)

    # 获取图像高度、宽度
    rows, cols = img.shape[:]

    # 图像二维像素转换为一维
    data = img.reshape((rows * cols, 1))
    data = np.float32(data)

    # 定义中心 (type,max_iter,epsilon)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    # K-Means聚集成x类
    compactness, labels, centers = cv2.kmeans(data, x, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # 生成最终图像
    dstx = labels.reshape((img.shape[0], img.shape[1]))
    #plt.imsave(r'' + out_path + "zong.png", dstx, cmap='gray')
    #labels_numpy = np.array(dstx) #用来判别cv2.kmeans聚类各聚类结果所在灰度范围
    labels_torch = torch.tensor(dstx)

    # 分割x类灰度
    for n in range(x):
        g_a_n=0
        tempx = copy.deepcopy(dstx)

        #if (tempx[0][0] == n) or (tempx[-1][0] == n) or (tempx[-1][-1] == n) or (tempx[0][-1] == n):
        #print(tempx[0][0])
        if (img[0][0]==0 and tempx[0][0] == n)or(img[rows - 1][cols - 1] == 0 and tempx[rows - 1][cols - 1] == n):
            n_f=n
            continue

        if n==0:
            list_numpy = np.array(dstx)
            labels_torch_index=np.argwhere(list_numpy==0)
            print("这是值为0的坐标：",labels_torch_index)
        else:
            labels_torch_index0 = torch.nonzero(torch.where(labels_torch == n, torch.tensor(n), torch.tensor(0)))
            print(labels_torch_index0)
            labels_torch_index=labels_torch_index0.numpy()
            print("这是值不为0的坐标：", labels_torch_index)

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if dstx[i][j] != n:
                    tempx[i][j] = 255
        plt.imsave(r'' + out_path + "/" + str(n) + ".png", tempx, cmap='gray')
        for index_n in labels_torch_index:
            g_a_n+=img[index_n[0]][index_n[1]]
            #print("打印索引：", index_n)
            #print("我在尝试打印灰度值：",img[index_n[0]][index_n[1]])
        #print("我打印的是整张图：",img)
        #print("我在打印灰度值加起来……",g_a_n)
        grey_all_n[n]=(g_a_n/len(labels_torch_index))
    del grey_all_n[n_f]
    c_gre=0
    cc=0
    print(grey_all_n)
    for gre in grey_all_n:
        print(type(gre))
        if gre==[0]:
            print(c_gre-cc)
            del grey_all_n[(c_gre-cc)]
            print(grey_all_n)
            cc+=0
        c_gre += 1
    print(grey_all_n)

    '''
    另一种分割图像的办法，快一点
    但是类号0的图像因为与逻辑出不来
    mask=dstx==n
    tempx=copy.deepcopy(dstx)
    tempx=dstx*mask
       plt.imsave(r''+out_path+"/"+str(n)+".png", tempx, cmap='gray')
    '''
    """   
    # 保存灰度图
    plt.imshow(dstx, 'gray')
    plt.show()

    plt.imsave("try.png", dstx, cmap='gray')
    """
    return grey_all_n
#分灰度2
def step2_threshold(in_path,out_path,peaks_less):
    img = plt.imread(in_path)
    im = np.array(img)
    # 绘制原图窗口
    plt.figure()
    #plt.imshow(img, cmap='gray')
    #plt.title('original')
    spot = peaks_less
    #spot =[25.5, 42.5, 85, 136, 178, 204]
    print(spot)
    for i in range(len(spot) - 1):
        imm = np.where(np.logical_and(im[..., :] > spot[i], im[..., :] < spot[i + 1]), 0, 255)
        cv2.imwrite(r'' + out_path+'/'+ str(i) + ".png", imm)

# 255-img
def reverse_huidu(img0):
    img0 = 255 - img0
    return img0


# 元素图对比度亮度调节
def contrasten(inpath, outpath):
    files = os.listdir(inpath)
    # 读图像
    for file in files:
        img = cv2.imread(r'' + inpath + '/' + file, cv2.IMREAD_GRAYSCALE)
        # 线性变换
        a = 3
        constrast = float(a) * img
        # 进行数据截断，大于255 的值要截断为255
        constrast[0 > 10] = 255
        # 数据类型转换
        constrast = np.round(constrast)
        # uint8类型
        constrast = constrast.astype(np.uint8)
        # 文件名和扩展名
        splitfile = os.path.splitext(file)
        filename = splitfile[0]
        # 显示原图和线性变换后的效果
        # plt.imsave(r''+outpath+"/"+filename+"_orignal.png",img,cmap="gray")
        # plt.imsave(r''+outpath+"/"+filename+".png", constrast, cmap='gray')#相对路径
        plt.imsave(outpath + "/" + filename + ".png", constrast, cmap='gray')  # 绝对路径，需要自己改


# 改透明背景
def read_path(file_pathname):
    # for filename in os.listdir(file_pathname):
    pic = Image.open(file_pathname)
    # pic = Image.open(r''+file_pathname+'\\'+filename)

    pic = pic.convert('RGBA')  # 转为RGBA模式
    width, height = pic.size
    array = pic.load()  # 获取图片像素操作入口
    for i in range(width):
        for j in range(height):
            # 获得某个像素点，格式为(R,G,B,A)元组
            pos = array[i, j]
            # 黑色背景为0，白色背景改255或者>=240
            # 其他颜色背景也适用但是分开判断
            isEdit = (sum([1 for x in pos[0:3] if x < 50]) == 3)
            # 判断黑色背景修改为透明，实际颜色仍为白色
            if isEdit:
                # 更改为透明
                # 有需要可以改成其他颜色但通道透明仍为0
                # 透明通道重现时，值需要改成255
                array[i, j] = (255, 255, 255, 0)
    # 保存图片
    # 会保存在文件同目录下，图片名字不变
    pic.save(r'yuan/transparent.png')
    print("successful")


# print保存到doc，不然打印上限
def text_create(GREY, m):
    # 此处是服务器地址，可以改为本地
    desktop_path = r"julei/G{}/".format(GREY)
    # 新创建的txt文件的存放路径
    full_path = desktop_path + "元素密度范围".format(m) + '.doc'  # 也可以创建一个.txt的word文档
    file = open(full_path, 'w')


# 自动创建文件夹用的
def folder_create():
    file_path = os.path.abspath(r'')
    a1 = "julei"  # 聚类图
    file_name1 = file_path + "\\" + a1
    os.makedirs(file_name1)
    a3 = "SPOTSALL"  # 存密度的
    file_name3 = file_path + "\\" + a3
    os.makedirs(file_name3)
    a11 = "same_element"  # 相同元素范围，即相同矿物
    file_name11 = file_path + "\\" + a11
    afi = "biaoge"  # 表格+图片
    file_namefi = file_path + "\\" + afi
    os.makedirs(file_namefi)
    adb1 = "duibi1"  # tiff转png
    file_name_adb1 = file_path + "\\" + adb1
    os.makedirs(file_name_adb1)
    adb2 = "duibi2"  # 对比度调节
    file_name_adb2 = file_path + "\\" + adb2
    os.makedirs(file_name_adb2)
    ay = "yuan"  # 放原图tiff转png，因为有些函数啊，tiff不是很方便操作
    file_name_yuan = file_path + "\\" + ay
    os.makedirs(file_name_yuan)
    aG = "Grayscale"
    file_name_G = file_path + "\\" + aG
    os.makedirs(file_name_G)
    # 如果你要生成每一块长啥样的图，请把以下解除注释，包括re_mark（），没必要，说实话
    # a5 = "zong"
    # file_name5 = file_path + "\\" + a5
    # os.makedirs(file_name5)
    for i in range(1, 7):
        a2 = "G{}".format(i)
        file_name2 = file_name1 + "\\" + a2  # 每个灰度的聚类
        file_name4 = file_name3 + "\\" + a2  # 每个灰度的元素密度值
        # file_name6 = file_name5 + "\\" + a2 #zong
        file_name12 = file_name11 + "\\" + a2  # 相同元素
        os.makedirs(file_name2)
        os.makedirs(file_name4)
        # os.makedirs(file_name6)
        os.makedirs(file_name12)
    file_name_P = file_name11 + "\\" + "P"  # 高P
    os.makedirs((file_name_P))


# 这是算区域内的点的
def ostu(img):
    global area
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转灰度
    blur = cv2.GaussianBlur(image, (5, 5), 0)  # 阈值一定要设为 0 ！高斯模糊
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # 二值化 0 = black ; 1 = white
    # cv2.imshow('image', th3)
    # a = cv2.waitKey(0)
    # print a
    # height, width = th3.shape
    # for i in range(height):
    #   for j in range(width):
    #        if th3[i, j] == 255:
    #            area += 1
    area = np.count_nonzero(th3)
    return area  # 返回区域内点的数量


# 看分几类的，除非你第一次跑，一般用不到！调用它的函数我注释了，要改的话，您随意
def get_k(df):
    # '利用SSE选择k'
    SSE = []  # 存放每次结果的误差平方和
    for k in range(1, 9):
        estimator = KMeans(n_clusters=k)  # 构造聚类器
        # df = [[i] for i in df]
        estimator.fit(np.array(df))
        SSE.append(estimator.inertia_)
    X = range(1, 9)
    plt.xlabel('k:1-9')
    plt.ylabel('SSE')
    plt.plot(X, SSE, 'o-')
    plt.show()


# 这玩意是聚类的（在咱这主要给密度spots聚类）
# data就是列表，n_cluster你想聚几类。不过还要人工瞄一眼，打印自行print（list_cluster(data, n_cluster)）
def list_cluster(data, n_cluster):
    new_data = [[i, 1] for i in data]
    new_data = np.array(new_data)
    cluster_rst = AgglomerativeClustering(n_clusters=n_cluster, affinity='euclidean', linkage='ward').fit_predict(
        new_data)

    return_data = []
    for i in range(n_cluster):
        subData = new_data[cluster_rst == i]
        return_data.append(list(subData[:, 0]))

    return return_data


# 这个是遍历元素图的，咱这里是elem_count张
# 不知道多少张可以把这个函数单独拎出去，看看print，（闲的没事的人才回去看吧~
def getFileList(dir, Filelist, ext=None):
    """
    获取文件夹及其子文件夹中文件列表
    输入 dir：文件夹根目录
    输入 ext: 扩展名
    返回： 文件路径列表
    """
    newDir = dir
    if os.path.isfile(dir):
        if ext is None:
            Filelist.append(dir)
        else:
            if ext in dir[-3:]:
                Filelist.append(dir)

    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            newDir = os.path.join(dir, s)
            getFileList(newDir, Filelist, ext)

    return Filelist


# 这是算连通区域的
def connected_test(img):
    global num_labels, labels, stats, centroids, area_avg, area_all
    # 中值滤波，去噪
    img = cv2.medianBlur(img, 3)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.namedWindow('original', cv2.WINDOW_AUTOSIZE)

    # 阈值分割得到二值化图片
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # 膨胀操作
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bin_clo = cv2.dilate(binary, kernel2, iterations=2)

    # 连通域分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_clo, connectivity=8)

    # 查看各个返回值
    # 连通域数量
    print('num_labels = ', num_labels)
    # 连通域的信息：对应各个轮廓的x、y、width、height和面积
    print('stats = ', stats)
    # 连通域的中心点
    print('centroids = ', centroids)
    # 每一个像素的标签1、2、3.。。，同一个连通域的标签是一致的
    print('labels = ', labels)

    # 计算平均面积
    areas = list()
    area_a = 0
    for i in range(num_labels):
        areas.append(stats[i][-1])
        print("轮廓%d的面积:%d" % (i, stats[i][-1]))
        if (i > 0):
            area_a += stats[i][-1]
    area_all = area_a
    area_avg = np.average(areas[1:-1])
    print("轮廓平均面积:", area_avg)

    # 筛选超过平均面积的连通域，打出来看看，平均面积已经是全局变量了，之后算的时候加个if条件就好,想看解注释
    # image_filtered = np.zeros_like(img)
    # for (i, label) in enumerate(np.unique(labels)):
    #    # 如果是背景，忽略
    #    if label == 0:
    #        continue
    #    if stats[i][-1] > area_avg:
    #        image_filtered[labels == i] = 255

    # cv2.imshow("image_filtered", image_filtered)
    # cv2.imshow("img", img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    # for i in range(num_labels):
    #    if stats[i][-1] < area_avg:
    #        del stats[i]
    # 不同的连通域赋予不同的颜色
    # output = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    # for i in range(1, num_labels):

    #  mask = labels == i
    #  output[:, :, 0][mask] = np.random.randint(0, 255)
    #  output[:, :, 1][mask] = np.random.randint(0, 255)
    #  output[:, :, 2][mask] = np.random.randint(0, 255)
    # cv2.imwrite('E:/re/re26/working/result/output1.jpg', output)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    return num_labels, labels, stats, centroids, area_avg


# 高P连通区域
def P_connected_test(img):
    global P_num_labels, P_labels, P_stats, P_centroids
    # 中值滤波，去噪
    img = cv2.medianBlur(img, 3)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.namedWindow('original', cv2.WINDOW_AUTOSIZE)

    # 阈值分割得到二值化图片
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # 膨胀操作
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bin_clo = cv2.dilate(binary, kernel2, iterations=2)

    # 连通域分析
    P_num_labels, P_labels, P_stats, P_centroids = cv2.connectedComponentsWithStats(bin_clo, connectivity=8)

    # 查看各个返回值
    # 连通域数量
    print('num_labels = ', P_num_labels)
    # 连通域的信息：对应各个轮廓的x、y、width、height和面积
    print('stats = ', P_stats)
    # 连通域的中心点
    print('centroids = ', P_centroids)
    # 每一个像素的标签1、2、3.。。，同一个连通域的标签是一致的
    print('labels = ', P_labels)
    return P_num_labels, P_labels, P_stats, P_centroids


# 在G1下面新建一个spotsData文件夹保存spots数据
def saveFile(GREY, m, spots_temp):
    my_data = spots_temp
    output = open(r'SPOTSALL/G{}/{}.pkl'.format(GREY, m), 'wb')  # 灰度要写吗？？？六个GREY
    # 写入到文件
    pickle.dump(my_data, output)
    output.close()


# 读取文件
def readFile(GREY, m):
    pkl_file = open(r'SPOTSALL/G{}/{}.pkl'.format(GREY, m), 'rb')

    # 从文件中读取
    spots_temp = pickle.load(pkl_file)
    pkl_file.close()
    return spots_temp


# 聚成四类后，每一类对应块号，重新组合并存储
def re_spots_blocks(spots_temp, minn, maxx, num_labels_fil_blocks):  # 因为是自动识别最大最小值，所以各种元素的范围都是变量，可以在文件夹中查看
    blocks = [[0 for col in range(1)] for row in range(2)]  # 初始化blocks列表 1*4为0，不然会报错
    count = 0
    sum_area = 0  # 聚类后，每个类面积占比大小
    area1 = 0
    # area2=0
    # area3=0
    area4 = 0
    for t in (spots_temp):
        count += 1
        # 分元素密度，记录i块位置
        if t >= minn[0]:
            blocks[0].append(num_labels_fil_blocks[count - 1])
            area1 += stats[num_labels_fil_blocks[count - 1]][-1]
        # 这里注释掉是因为原本是四类嘛，直接改不是很好改，想要四类可以使用备份的四类代码
        # elif minn[1] <= t <= maxx[1]:
        #    blocks[1].append(num_labels_fil_blocks[count-1])
        #    area2 += stats[num_labels_fil_blocks[count-1]][-1]
        # elif minn[2] <= t <= maxx[2]:
        #    blocks[2].append(num_labels_fil_blocks[count-1])
        #    area3 += stats[num_labels_fil_blocks[count-1]][-1]
        else:
            blocks[1].append(num_labels_fil_blocks[count - 1])
            area4 += stats[num_labels_fil_blocks[count - 1]][-1]
    for bl in range(2):
        del blocks[bl][0]

    for i in range(2):
        for j in blocks[i]:
            sum_area += stats[j][-1]
    print("第一类面积占比:", (area1 / sum_area), "\n第四类面积占比:", (area4 / sum_area), "\n", file=outputfile1)  # 存入word
    print("第一类面积占比:", (area1 / sum_area), "\n第四类面积占比:", (area4 / sum_area), "\n")  # 打印出来，主要是我想看，你不想看注释掉
    perc_area1 = area1 / sum_area
    perc_area4 = area4 / sum_area
    area = (min(area1, area4) / area_all)  # 占当前灰度面积
    # 因为是二分类，如果元素差距不大，聚类结果应为二等分，表现为面积同理；如果元素差距较大，则面积会在二类间有所偏颇。
    # 这里反向思维，取两个类的面积比，如果差异较大，即二八开以上，判定差异较大，分两类，反之合并为一类。（这里试过三七开，效果不理想）
    # 举个例子：p_a1为0.8，p_a2为0.2,有0.8-0.2=0.6，依次为基准，假设p_a1为0.79，p_a2为0.21,0.79-0.21=0.58<0.6,分为一类，大于0.6分为两类
    # 举个例子：p_a1为0.75，p_a2为0.25,有0.75-0.25=0.5，依次为基准，假设p_a1为0.79，p_a2为0.21,0.79-0.21=0.58>0.5,分为两类，小于0.5合并类
    # area是当前元素密度范围下的块占该灰度的面积，如果占比过小，可以认为是一些散点，或小块，没有比较的意义，不纳入考虑
    # area_quan当前元素密度范围下的块占全图，如果占比过小，可以认为是一些散点，或小块，没有比较的意义，不纳入考虑，和上面的区别是，对于一些本身就是散点或是很小的块，，上面一个相对面积处理的效果一般
    # 如果你有更好的合并依据，请友善交流
    area_quan = (min(area1, area4) / stats[0][-1])
    if ((max(perc_area1, perc_area4) - min(perc_area1, perc_area4) < 0.5) and (area < 0.20)) or (area_quan < 0.01):
    #if ((max(perc_area1, perc_area4) - min(perc_area1, perc_area4) < 0.4) and (area < 0.10)) or (area_quan < 0.01):
        blocks[0] += blocks[1]
        blocks[1] = 0
    # if(blocks[1]!=0):
    #    for sor in range(2): #只取前面两类
    #      blocks_sorted[sor]=sorted(blocks[sor])
    # else:
    #    blocks_sorted[0] = sorted(blocks[0])
    return blocks


# 筛选高P
def P_re_spots_blocks(m, spots_temp):
    count = 0
    element = m
    blocks = [0 for col in range(1)]
    # P
    if (m == 0):
        while (count < len(spots_temp)):
            if (spots_temp[count] > 0.1):
                blocks.append(count + 1)
            count += 1
    if (m == 1):
        while (count < len(spots_temp)):
            if (spots_temp[count] > 0.1):
                blocks.append(count + 1)
            count += 1
    if (m == 2):
        while (count < len(spots_temp)):
            if (spots_temp[count] > 0.1):
                blocks.append(count + 1)
            count += 1
    if (m == 3):
        while (count < len(spots_temp)):
            if (spots_temp[count] > 0.1):
                blocks.append(count + 1)
            count += 1
    if (m == 4):
        while (count < len(spots_temp)):
            if (spots_temp[count] > 0.1):
                blocks.append(count + 1)
            count += 1
    if (m == 5):
        while (count < len(spots_temp)):
            if (spots_temp[count] > 0.1):
                blocks.append(count + 1)
            count += 1
    return element, blocks


def re_mark(img, labels, blocks, m, GREY, num_labels):  # 着色
    # 看每一连通区域长啥样，其实可以注释掉，有点好奇
    # for ii in range(num_labels):
    #   output1 = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    #   # print(i)
    #   mask = labels == ii
    #   output1[:, :, 0][mask] = 0
    #   output1[:, :, 1][mask] = 255
    #   output1[:, :, 2][mask] = 0
    #   cv2.imwrite(r'zong/G{}/{}.png'.format(GREY, ii), output1)

    for j in range(2):
        if (blocks[GREY - 1][m][j] != 0):
            output = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
            inter = blocks[GREY - 1][m][j][:]
            print("第", j, "次遍历的inter ：", inter)
            for i in range(len(inter)):
                # print(i)
                mask = labels == inter[i]
                output[:, :, 0][mask] = 0
                output[:, :, 1][mask] = 255
                output[:, :, 2][mask] = 0
            cv2.imwrite(r'julei/G{}/{}_{}.png'.format(GREY, m, j), output)
            print("sucessfully painting G{} -- {}_{} ~".format(GREY, m, j))  # 用来看进度的
            inter.clear()
    return


# 筛选高P
def P_re_mark(img, P_labels, blocks, m, output1):  # 着色
    painting1 = np.random.randint(0, 255)
    painting2 = np.random.randint(0, 255)
    painting3 = np.random.randint(0, 255)
    if (blocks[m] is not null):
        output = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
        inter = blocks[m][:]
        # print("第",c+1,"次遍历的inter ：",inter)
        for i in range(len(inter)):
            # print(i)
            mask = P_labels == inter[i]
            output[:, :, 0][mask] = painting1
            output[:, :, 1][mask] = painting2
            output[:, :, 2][mask] = painting3
            # 这个是全图
            output1[:, :, 0][mask] = painting1
            output1[:, :, 1][mask] = painting2
            output1[:, :, 2][mask] = painting3
            cv2.imwrite(r'same_element/P/{}.png'.format(m), output)
        print("painting {} sucessfully~".format(m))  # 用来看进度的
        inter.clear()
    return output1


# 掩码抠图
def mask_test(img, img2, stats, num_labels, m, labels, GREY):
    # columns_list = ["Al", "Ca", "F", "Fe", "K", "Mg", "Na", "O", "P", "S", "Si", "Ti"]
    blocks = [[0 for col in range(1)] for row in range(4)]  # 1*4
    num_labels_fil_blocks = list()  # 用来记录筛选后的块号
    global res, temp, temp_s
    spots = list()  # 点密度列表
    spots_sorted2 = [[0 for col in range(1)] for row in range(4)]  # 点密度排序
    minn = list()
    maxx = list()

    # 第一次未保存文件前要注掉
    # spots_temp = readFile(m)
    # spots = spots_temp.tolist()  # 还原成list

    for i in range(1, num_labels):  # 这里不从0开始遍历是因为0应该是背景图
        if stats[i][-1] > area_avg:
            num_labels_fil_blocks.append(i)  # 用来记录筛选后的块号
            # i = 10
            maker = labels == i
            mask = np.zeros(img.shape, dtype=np.uint8)
            # mask[stats[i][0]:stats[i][0] + stats[i][2], stats[i][1]:stats[i][1] + stats[i][3]] = 255  # 生成掩码
            mask[:, :, :][maker] = 255  # 生成掩码
            res = cv2.bitwise_and(img2, mask)  # 这里的img2是元素图，res是掩码后的图

            temp = stats[i][-1]  # 提取连通区域面积
            temp_s = ostu(res) / temp  # 调用ostu求点的数量，数量除面积得点的密度
            spots.append(temp_s)  # 存入列表
            spots_temp = np.array(spots)
            # spots_temp2=pd.DataFrame(spots)
            # print(spots_temp2)
            # spots_df[columns_list[m]]=spots_df.append(spots_temp2)
            saveFile(GREY, m, spots_temp)  # 存最初的那一个，不然块号会乱掉

    print('spots{}_{} :'.format(GREY, m), spots)
    # spots_df= DataFrame(spots)
    # get_k(spots_df) #这个是看聚类分几类的,要看自己取消注释
    if len(num_labels_fil_blocks) > 2:  # 原本是四类，这里改了，在消除小块后聚两类，足够了，后期还是要合并
        spots_lc = list_cluster(spots, 2)
        spots_sorted = sorted(spots_lc)[::-1]
        for sor in range(2):  # 只取前面两类
            spots_sorted2[sor] = sorted(spots_sorted[sor])[::-1]
        print('聚类排序后的 spots{}_{} :'.format(GREY, m), spots_sorted)
        print('二次排序后的 spots{}_{} :'.format(GREY, m), spots_sorted2)
        for j in range(2):  # 打印出来省的自己看了
            print("{}_{}中，第{}类，密度最大值：".format(GREY, m, j), max(spots_sorted[j]), "  验证排序：", spots_sorted2[j][0])
            print("{}_{}中，第{}类，密度最小值：".format(GREY, m, j), min(spots_sorted[j]), "  验证排序：", spots_sorted2[j][-1])
            sys.stdout = outputfile1  # 密度范围
            print("{}_{}中，第{}类，密度范围：".format(GREY, m, j), min(spots_sorted[j]), "—", max(spots_sorted[j]), "\n",
                  file=outputfile1)
            print("{}_{}中,第{}类, 密度均值为：".format(GREY, m, j), np.average(spots_sorted2[j]), "\n", file=outputfile1)
            minn.append(min(spots_sorted[j]))
            maxx.append(max(spots_sorted[j]))
            sys.stdout = __console__  # 除密度范围,密度均值其他数据,打印到控制台
        blocks = re_spots_blocks(spots_temp, minn, maxx, num_labels_fil_blocks)
    # spots_temp = np.array(spots) # 用来解决 '>' not supported between instances of 'list' and 'float'
    # element,blocks = re_spots_blocks(m, spots_temp)
    # print("blocks ",m,":", blocks)
    # return blocks  # 返回列表
    return spots, blocks, num_labels_fil_blocks


# 筛选高P
def P_mask_test(img, img2, P_stats, P_num_labels, m, P_labels):
    global P_res, P_temp, P_temp_s
    spots = list()  # 点密度列表
    blocks = list()  # 相同密度块列表

    for i in range(1, P_num_labels):  # 这里不从0开始遍历是因为0应该是背景图
        # i = 10
        maker = P_labels == i
        mask = np.zeros(img.shape, dtype=np.uint8)
        # mask[stats[i][0]:stats[i][0] + stats[i][2], stats[i][1]:stats[i][1] + stats[i][3]] = 255  # 生成掩码
        mask[:, :, :][maker] = 255  # 生成掩码
        P_res = cv2.bitwise_and(img2, mask)  # 这里的img2是元素图，res是掩码后的图
        P_temp = P_stats[i][4]  # 提取连通区域面积
        area = ostu(P_res)
        P_temp_s = area / P_temp  # 调用ostu求点的数量，数量除面积得点的密度
        spots.append(P_temp_s)  # 存入列表
        # cv2.imwrite('E:/re/re26/working24/result3/output2.jpg', res)
        # print('spots=', spots)
    spots_temp = np.array(spots)  # 用来解决 '>' not supported between instances of 'list' and 'float'
    element, blocks = P_re_spots_blocks(m, spots_temp)
    # print(len(spots_temp))
    return element, spots, blocks


# 一个两类
def c_el1(count_list, blocks):
    s_elment = [[0 for col in range(1)] for row in range(2 ** (len(count_list)))]
    k_count = 0  # 分几类矿物
    b_c = 0
    for el1 in range(2):
        b_c += 1
        s_elment[b_c - 1] = [k for k in blocks[count_list[0]][el1]]
        k_count += 1
    return k_count, s_elment


# 两个两类
def c_el2(count_list, blocks):
    s_elment = [[0 for col in range(1)] for row in range(2 ** (len(count_list)))]
    k_count = 0  # 分几类矿物
    b_c = 0
    for el1 in range(2):
        for el2 in range(2):
            b_c += 1
            s_elment[b_c - 1] = [k for k in blocks[count_list[0]][el1] if (k in blocks[count_list[1]][el2])]
            k_count += 1
    return k_count, s_elment


# 三个两类
# 一般来说啊，这个以下都用不到的，只是写出这些可能性，
# 换句话来说，如果能用到这个以下的分类方法，可以考虑换合并方法了,需要的话请微调参数~具体微调详见函数re_spots_blocks，代码245行左右
# 即便展示结果，这里两类划分也只写到七，已经有2^7种类了，太多，没有实际意义，接下来都不考虑了
def c_el3(count_list, blocks):
    s_elment = [[0 for col in range(1)] for row in range(2 ** (len(count_list)))]
    k_count = 0  # 分几类矿物
    b_c = 0
    for el1 in range(2):
        for el2 in range(2):
            for el3 in range(2):
                b_c += 1
                s_elment[b_c - 1] = [k for k in blocks[count_list[0]][el1] if
                                     (k in blocks[count_list[1]][el2] and k in blocks[count_list[2]][el3])]
                k_count += 1
    return k_count, s_elment


# 四个两类
def c_el4(count_list, blocks):
    s_elment = [[0 for col in range(1)] for row in range(2 ** (len(count_list)))]
    k_count = 0  # 分几类矿物
    b_c = 0
    for el1 in range(2):
        for el2 in range(2):
            for el3 in range(2):
                for el4 in range(2):
                    b_c += 1
                    s_elment[b_c - 1] = [k for k in blocks[count_list[0]][el1] if (
                                k in blocks[count_list[1]][el2] and k in blocks[count_list[2]][el3] and k in
                                blocks[count_list[3]][el4])]
                    k_count += 1
    return k_count, s_elment


# 五个两类
def c_el5(count_list, blocks):
    s_elment = [[0 for col in range(1)] for row in range(2 ** (len(count_list)))]
    k_count = 0  # 分几类矿物
    b_c = 0
    for el1 in range(2):
        for el2 in range(2):
            for el3 in range(2):
                for el4 in range(2):
                    for el5 in range(2):
                        b_c += 1
                        s_elment[b_c - 1] = [k for k in blocks[count_list[0]][el1] if (
                                    k in blocks[count_list[1]][el2] and k in blocks[count_list[2]][el3] and k in
                                    blocks[count_list[3]][el4] and k in blocks[count_list[4]][el5])]
                        k_count += 1
    return k_count, s_elment


# 六个两类
def c_el6(count_list, blocks):
    s_elment = [[0 for col in range(1)] for row in range(2 ** (len(count_list)))]
    k_count = 0  # 分几类矿物
    b_c = 0
    for el1 in range(2):
        for el2 in range(2):
            for el3 in range(2):
                for el4 in range(2):
                    for el5 in range(2):
                        for el6 in range(2):
                            b_c += 1
                            s_elment[b_c - 1] = [k for k in blocks[count_list[0]][el1] if (
                                        k in blocks[count_list[1]][el2] and k in blocks[count_list[2]][el3] and k in
                                        blocks[count_list[3]][el4] and k in blocks[count_list[4]][el5] and k in
                                        blocks[count_list[5]][el6])]
                            k_count += 1
    return k_count, s_elment


# 七个两类
def c_el7(count_list, blocks):
    s_elment = [[0 for col in range(1)] for row in range(2 ** (len(count_list)))]
    k_count = 0  # 分几类矿物
    b_c = 0
    for el1 in range(2):
        for el2 in range(2):
            for el3 in range(2):
                for el4 in range(2):
                    for el5 in range(2):
                        for el6 in range(2):
                            for el7 in range(2):
                                b_c += 1
                                s_elment[b_c - 1] = [k for k in blocks[count_list[0]][el1] if (
                                            k in blocks[count_list[1]][el2] and k in blocks[count_list[2]][el3] and k in
                                            blocks[count_list[3]][el4] and k in blocks[count_list[4]][el5] and k in
                                            blocks[count_list[5]][el6] and k in blocks[count_list[6]][el7])]
                                k_count += 1
    return k_count, s_elment


# 合并相同元素密度块号，好上色绘图
def dsame_elements(blocks):
    count = 0
    count_list = list()
    for i in range(elem_count):
        if (blocks[i][1] != 0):
            count_list.append(count)
        count += 1
    s_elment = [[0 for col in range(1)] for row in range(2 ** (len(count_list)))]
    print("\n两类个数：", (len(count_list)))
    k_count = 0  # 分几类矿物
    if ((len(count_list)) == 0):
        k_count = 1
        s_elment = [k for k in blocks[0]]
    if ((len(count_list)) == 1):
        k_count, s_elment = c_el1(count_list, blocks)
    if ((len(count_list)) == 2):
        k_count, s_elment = c_el2(count_list, blocks)
    if ((len(count_list)) == 3):
        k_count, s_elment = c_el3(count_list, blocks)
    if ((len(count_list)) == 4):
        k_count, s_elment = c_el4(count_list, blocks)
    if ((len(count_list)) == 5):
        k_count, s_elment = c_el5(count_list, blocks)
    if ((len(count_list)) == 6):
        k_count, s_elment = c_el6(count_list, blocks)
    if ((len(count_list)) == 7):
        k_count, s_elment = c_el7(count_list, blocks)
    return k_count, s_elment


# 面积占比
def are_pic_same_element(k_count, same_elements, stats):
    are_pic = list()
    for i in range(0, k_count):
        area = 0
        # for j in range(len(same_elements[i])):
        # #IndexError: only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) and integer or boolean arrays are valid indices
        print("块号：", same_elements[i])
        for j in same_elements[i]:
            area += stats[j][-1]
        area_zhanbi = round((area / area_zong), 4)
        are_pic.append(area_zhanbi)
    return are_pic


# 这个函数主要是用来存密度数据的，自动输出表格的话需要用到
def AVE_same_element(k_count, same_elements, num_labels_fil_blocks, spots, GREY):
    # df = pd.DataFrame(data=None, columns=['name', 'number'])
    midu_avg = list()
    midu_rag = list()
    midu = list()
    midu_avg_k_count=0
    # huidu="灰度"+str(GREY)
    for i in range(0, k_count):
        # grey = str(GREY-1) + "_" + str(k_count)
        for j in same_elements[i]:
            midu.append(spots[num_labels_fil_blocks.index(j)])
        if len(midu)>1:
           midu_r = str(round(min(midu), 2)) + "--" + str(round(max(midu), 2))
           print("我倒要看看有没有nan！：",midu)
           midu_avg.append(round(np.nanmean(midu), 2))  # 用来记录平均密度数值，这里记录平均密度而不是范围是因为平均密度更好绘图，需要范围表可以再打一张
           midu_avg_k_count+=round(np.nanmean(midu), 2)
           print("midu_avg,均值：", round(np.nanmean(midu), 2))
           # lei_pic.append((grey)) #用来记录那个类别图的名字啊
           # gre_pic.append((huidu))  # 用来记录灰度
           midu_rag.append(midu_r)  # 用来记录密度范围数值
           # midu_rag.append(huidu)  # 用来记录灰度
        elif len(midu)==1:
            midu_avg.append(round(min(midu), 2))
            midu_rag.append(round(min(midu), 2))
            midu_avg_k_count +=round(min(midu), 2)
        elif len(midu)==0:
            midu_avg.append(0)
            midu_rag.append(0)
            midu_avg_k_count += 0
        else:
            midu_avg.append(0)
            midu_rag.append(0)
            midu_avg_k_count +=0
    # return midu_avg,lei_pic,gre_pic,midu_rag
    if midu_avg_k_count>0:
        midu_avg_k_count/=k_count
    midu.clear()
    return midu_avg, midu_rag,midu_avg_k_count


# 输出块类编号和灰度
def bianhao_huidu(k_count, GREY,xx,midu_avg,midu_avg_k_count,midu_avg_si,midu_avg_ti,taitiekuang):
    global ttk
    gre_pic = list()
    lei_pic = list()
    kuangwu = list()
    gre_pic.clear()
    lei_pic.clear()
    kuangwu.clear()
    print("我想传进去的峰值：",xx)
    print("我想传进去的峰值的数据类型：",type(xx))
    x=int(xx)
    x1=x*0.9
    x2=x*1.1
    for i in range(0, k_count):
        print(i)
        huidu = "灰度" + str(GREY)
        print("灰度:", huidu)
        gre_pic.append((huidu))  # 用来记录灰度
        grey = str(GREY - 1) + "_" + str(i)
        print("编号:", grey)
        lei_pic.append((grey))  # 用来记录那个类别图的名字啊
        if (x <= 79 and x1<=79)or((x1<= 79 and x2<=79))or((x<= 79 and x2<=79)):
            kuangwu.append("石英");
        elif (midu_avg_k_count[10]<midu_avg_si) and(midu_avg_k_count[11]>midu_avg_ti):
            kuangwu.append("钛铁矿");
            ttk = GREY - 1
        elif (79 < x <= 104 and 79<x1<=104)or(79 < x1 <= 104 and 79<x2<=104)or(79 < x <= 104 and 79<x2<=104):
            kuangwu.append("斜长石");
        elif (104 < x <= 157 and 104 < x1 <= 157) or (104 < x2 <= 157 and 104 < x1 <= 157) or (
                        104 < x <= 157 and 104 < x2 <= 157):
            if ttk != 8:
                if midu_avg[1][i] > taitiekuang[1]:
                    kuangwu.append("辉石");
                else:
                    kuangwu.append("橄榄石");
            else:
                if (max(midu_avg[1]) - min(midu_avg[1])) < 0.05 and (max(midu_avg[5]) > 0.25):
                    kuangwu.append("橄榄石");
                elif (max(midu_avg[1]) - min(midu_avg[1])) < 0.05 or (
                        max(midu_avg[5]) - min(midu_avg[5])) < 0.05:
                    kuangwu.append("辉石");
                else:
                    # print("fe/mg比值：",midu_avg[3][i]/midu_avg[5][i],"fe/mg灰度整体下比值：",midu_avg_k_count[3]/midu_avg_k_count[5])
                    print("ca均值：", midu_avg[1][i], "ca灰度下整体均值：", midu_avg_k_count[1])
                    # print("fe均值：",midu_avg[3][i],"fe灰度下整体均值：",midu_avg_k_count[3])
                    print("mg均值：", midu_avg[5][i], "mg灰度下整体均值：", midu_avg_k_count[5])
                    # if ((midu_avg[1][i]<=midu_avg_k_count[1])and(midu_avg[5][i]>=midu_avg_k_count[5]))or((midu_avg[3][i]/midu_avg[5][i])<midu_avg_k_count[3]/midu_avg_k_count[5]): #3：Fe,5:Mg,都大于均值取橄榄石,fe/mg mg多，整体小，即橄榄石fe/mg比辉石小
                    if ((midu_avg[1][i] <= midu_avg_k_count[1]) and (midu_avg[5][i] >= midu_avg_k_count[5])):
                        kuangwu.append("橄榄石");
                    else:
                        kuangwu.append("辉石");
        elif (157 < x <= 185 and 157<x1<=185)or(157 < x2 <= 185 and 157<x1<=185)or(157 < x <= 185 and 157<x2<=185):
            kuangwu.append("钛铁矿");
        elif (x > 185 and x1>185)or(x2 > 185 and x1>185)or(x > 185 and x2>185):
            kuangwu.append("陨硫铁");
    return gre_pic, lei_pic,kuangwu


# 总图上色
def Seco_re_mark(img, labels, same_elements, count, GREY, ALL_output):  # 着色
    output1 = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    count = count
    c = 0
    for m in range(count):
        # 放在这随机是有原因的！
        painting1 = np.random.randint(0, 255)
        painting2 = np.random.randint(0, 255)
        painting3 = np.random.randint(0, 255)
        if same_elements[c] is not null:
            output = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
            inter = same_elements[c][:]
            print("第", c + 1, "次遍历的inter ：", inter)
            for i in range(len(inter)):
                # print(i)
                mask = labels == inter[i]
                output[:, :, 0][mask] = painting1
                output[:, :, 1][mask] = painting2
                output[:, :, 2][mask] = painting3
                # 这个是灰度全图
                output1[:, :, 0][mask] = painting1
                output1[:, :, 1][mask] = painting2
                output1[:, :, 2][mask] = painting3
                # 这个是总全图
                ALL_output[:, :, 0][mask] = painting1
                ALL_output[:, :, 1][mask] = painting2
                ALL_output[:, :, 2][mask] = painting3
                cv2.imwrite(r'same_element/G{}/{}.png'.format(GREY, c), output)
        print("painting {} sucessfully~".format(c))  # 用来看进度的
        cv2.imwrite(r'same_element/G{}/all.png'.format(GREY), output1)
        c += 1
        inter.clear()
    return ALL_output


if __name__ == '__main__':
    img_folder_duibi = r'duibi'
    img_path_db = sorted([os.path.join(img_folder_duibi, name) for name in os.listdir(img_folder_duibi) if
                          name.endswith('.tiff')])  ###这里的'.tif'可以换成任意的文件后缀
    elem_count = len(img_path_db)
    bianhao = list()
    leihao = list()
    A_num = [[0 for col in range(1)] for row in range(6)]
    K_num = [[0 for col in range(1)] for row in range(6)]
    # 但其实我感觉哈，如果是存储文件的话，这个要不要其实无所谓
    spots = [[[0 for col in range(1)] for row in range(elem_count)] for l in range(6)]  # 初始化spots列表 1*elem_count*6为0，不然会报错
    # blocks = [[[[0 for col in range(1)] for row in range(4)] for l in range(elem_count)]for n in range(6)]  # 初始化blocks列表 1*4*elem_count*6为0，不然会报错
    blocks = [[[0 for col in range(1)] for row in range(elem_count)] for l in range(6)]  # 初始化blocks列表 1*elem_count*6为0，不然会报错
    midu_avg = [[[0 for col in range(1)] for row in range(elem_count)] for l in range(6)]  # 初始化midu_avg列表 1*elem_count*6为0，不然会报错
    midu_avg_k_count=[[[0 for col in range(1)] for row in range(elem_count)] for l in range(6)]  # 初始化midu_avg_k_count列表 1*elem_count*6为0，不然会报错
    gre_pic = [[0 for col in range(1)] for l in range(6)]  # 初始化gre_pic列表 1*6为0，不然会报错
    kuangwu = [[0 for col in range(1)] for l in range(6)]  # 初始化gre_pic列表 1*6为0，不然会报错
    midu_rag = [[[0 for col in range(1)] for row in range(elem_count)] for l in range(6)]  # 初始化midu_rag列表 1*6为0，不然会报错
    lei_pic = [[0 for col in range(1)] for l in range(6)]  # 初始化lei_pic列表 1*elem_count*6为0，不然会报错
    num_labels_fil_blocks = [[[0 for col in range(1)] for row in range(elem_count)] for l in range(6)]  # 初始化num_labels_fil_blocks列表 1*elem_count*6为0，不然会报错
    are_pic = [[0 for col in range(1)] for row in range(6)]
    same_elements_list = [[0 for col in range(1)] for row in range(6)]
    stats_list = [[0 for col in range(1)] for row in range(6)]
    grey_all= [[0 for col in range(1)] for row in range(6)]
    midu_avg_si = 0
    midu_avg_ti = 0
    # 读入图片
    GREY = 0  # 灰度遍历计数
    folder_create()

    org_img_folder_org = r'orgpic'
    img_path = sorted([os.path.join(org_img_folder_org, name) for name in os.listdir(org_img_folder_org) if
                       name.endswith('.tiff')])  ###这里的'.tif'可以换成任意的文件后缀
    print('本次执行检索到 ' + str(len(img_path)) + ' 张图像\n')
    for i in range(len(img_path)):
        img_name_org = os.path.split(img_path[i])[-1]
        img_name_org = img_name_org.split('.')[0]
        img_org = io.imread(img_path[i])  ##img_path[i]就是完整的单个指定文件路径了
        tiffread(img_path[i], r'yuan')
    print(img_name_org)
    all_img0 = cv2.imread(r'yuan/{}.png'.format(img_name_org))
    size_img0 = all_img0.shape
    w_img0 = size_img0[1] # 宽度
    h_img0 = size_img0[0]  # 高度
    All_output = np.zeros((all_img0.shape[0], all_img0.shape[1], 3), np.uint8)
    cv2.imwrite(r'yuan/{}.jpg'.format(img_name_org), all_img0)
    read_path(r'yuan/{}.jpg'.format(img_name_org))  # 改透明背景，暂时用不到，目前方法改了反而不好

    org_img_folder_duibi = r'duibi'
    img_path_duibi = sorted([os.path.join(org_img_folder_duibi, name) for name in os.listdir(org_img_folder_duibi) if name.endswith('.tiff')])  ###这里的'.tif'可以换成任意的文件后缀
    for i in range(len(img_path_duibi)):
        img_name_duibi = os.path.split(img_path_duibi[i])[-1]
        img_name_duibi = img_name_duibi.split('.')[0]
        img_duibi = io.imread(img_path_duibi[i])  ##img_path[i]就是完整的单个指定文件路径了
        tiffread(img_path_duibi[i], r'duibi1')
    contrasten(r'duibi1', r'duibi2')

    path1 = r"orgpic"
    extract(r"orgpic")  # path1
    piclist = multyfiel(r"blackbg_mdf")
    print(piclist)
    for picname in piclist:
        #x,peaks  = findpeaks(r'' + "blackbg_mdf/" + picname)
        x, peaks,peaks_less= find_g(r'' + "blackbg_mdf/" + picname)
        print(x)
        # 文件名和扩展名
        splitfile = os.path.splitext(picname)
        name = splitfile[0]
        exist = os.path.exists(name)
        if not exist:
            os.makedirs(name)
        grey_all=step1_threshold(r'' + "blackbg_mdf/" + picname, name, (x+1)) #暂时先不用，好像可以不用这么分
        #step2_threshold(r'yuan/{}.jpg'.format(name), name, peaks_less)

    org_img_folder_G = r'{}'.format(name)  # 这是多张提取，该地址为灰度图所在文件夹
    imglist_G = getFileList(org_img_folder_G, [], 'png')  # 调用getFileList()函数
    print('本次执行检索到 ' + str(len(imglist_G)) + ' 张图像\n')  # 测试用，看看图片遍历有没有问题
    for imgpath_G in imglist_G:
        imgname_G = os.path.splitext(os.path.basename(imgpath_G))[0]
        img_G = cv2.imread(imgpath_G, cv2.IMREAD_GRAYSCALE)  # 对每幅图像执行相关操作
        img_G = reverse_huidu(img_G)
        cv2.imwrite(r'Grayscale/{}.png'.format(imgname_G), img_G)

    org_img_folder1 = r'Grayscale'  # 这是多张提取，该地址为灰度图所在文件夹
    imglist1 = getFileList(org_img_folder1, [], 'png')  # 调用getFileList()函数
    print('本次执行检索到 ' + str(len(imglist1)) + ' 张图像\n')  # 测试用，看看图片遍历有没有问题
    for imgpath1 in imglist1:
        # elements = list() #元素是否存在列表
        m = 0  # 对比图遍历计数
        GREY += 1
        # 保存.doc
        text_create(GREY, m)
        __console__ = sys.stdout
        outputfile1 = open(r'julei/G{}/'.format(GREY) + "元素密度范围".format(m) + '.doc', 'w')
        outputfile2 = open(r'same_element/G{}/'.format(GREY) + "相同元素密度块号".format(m) + '.doc', 'w')
        # outputfile2 = open(r'julei/'.format(GREY) + "ALL".format(m) + '.doc', 'w')
        sys.stdout = __console__  # 除密度范围其他数据,打印到控制台

        imgname1 = os.path.splitext(os.path.basename(imgpath1))[0]
        img = cv2.imread(imgpath1, cv2.IMREAD_COLOR)  # 对每幅图像执行相关操作
        # img = cv2.imread("E:/re/re26/working/Grayscale/q1.4.jpg") # 这是单张灰度图
        connected_test(img)
        stats_list[GREY - 1] = stats
        A_num[GREY - 1] = num_labels  # 这个是记录每个灰度总块数的，新方法好像没用到
        # print(A_num)
        # img2 = cv2.imread("E:/re/re26/duibi/f1.png")  #这是单张提取对比图
        org_img_folder2 = r'duibi2'  # 这是多张提取对比图，该地址为元素图所在文件夹
        # 检索文件
        imglist2 = getFileList(org_img_folder2, [], 'png')  # 调用getFileList()函数
        print('本次执行检索到 ' + str(len(imglist2)) + ' 张图像\n')  # 测试用，看看图片遍历有没有问题
        for imgpath2 in imglist2:
            imgname2 = os.path.splitext(os.path.basename(imgpath2))[0]
            img2 = cv2.imread(imgpath2, cv2.IMREAD_COLOR)  # 对每幅图像执行相关操作
            img2 = cv2.resize(img2, (w_img0, h_img0), interpolation=cv2.INTER_CUBIC)
            spots[(GREY - 1)][m], blocks[(GREY - 1)][m], num_labels_fil_blocks[(GREY - 1)][m] = mask_test(img, img2,
                                                                                                          stats,
                                                                                                          num_labels, m,
                                                                                                          labels, GREY)
            re_mark(img, labels, blocks, m, GREY, num_labels)
            m += 1
        k_count, same_elements = dsame_elements(blocks[(GREY - 1)])
        same_elements_list[GREY - 1] = same_elements
        K_num[GREY - 1] = k_count
        for i in range(elem_count):  # 遍历密度，计算同一灰度下每种密度的均值
            # midu_avg[(GREY-1)][i],lei_pic[(GREY-1)][i],gre_pic[(GREY-1)][i],midu_rag[(GREY-1)][i]=AVE_same_element(k_count, same_elements, num_labels_fil_blocks[(GREY-1)][i], spots[(GREY-1)][i],GREY) #实际上midu_avg的大小是6*elem_count*k_count
            midu_avg[(GREY - 1)][i], midu_rag[(GREY - 1)][i],midu_avg_k_count[(GREY - 1)][i] = AVE_same_element(k_count, same_elements,num_labels_fil_blocks[(GREY - 1)][i],spots[(GREY - 1)][i],GREY)  # 实际上midu_avg的大小是6*elem_count*k_count
        print(grey_all[GREY-1])
        print(midu_avg_k_count)
        #x=(peaks_less(GREY-1)+peaks_less(GREY))/2
        #gre_pic[(GREY - 1)], lei_pic[(GREY - 1)] ,kuangwu[(GREY-1)]= bianhao_huidu(k_count, GREY,grey_all[GREY-1],midu_avg[(GREY - 1)],midu_avg_k_count[(GREY - 1)])
        print(midu_avg_k_count[(GREY - 1)])
        midu_avg_si += midu_avg_k_count[(GREY - 1)][10]  # 计算每个灰度下si的密度
        midu_avg_ti += midu_avg_k_count[(GREY - 1)][11]  # 计算每个灰度下ti的密度
        #print("gre_pic[(GREY-1)]:", gre_pic[(GREY - 1)])
        #print("lei_pic[(GREY-1)]:", lei_pic[(GREY - 1)])
        area_zong += area_all
        print("共分矿物类数：", k_count, "\n")
        print("same_elements_blocks:", same_elements, file=outputfile2)
        All_output = Seco_re_mark(img, labels, same_elements, k_count, GREY, All_output)
        outputfile1.close()  # close后才能看到写入的数据
        outputfile2.close()  # close后才能看到写入的数据
    cv2.imwrite(r'same_element/all.png', All_output)
    print("saving all same_elements of all Grayscale picture sucessfully~")
    midu_avg_si /= GREY
    midu_avg_ti /= GREY
    for gi in range(0, GREY):
        if ttk == 8:
            gre_pic[gi], lei_pic[gi], kuangwu[gi] = bianhao_huidu(K_num[gi], (gi + 1), grey_all[gi],
                                                                  midu_avg[gi],
                                                                  midu_avg_k_count[gi], midu_avg_si, midu_avg_ti,
                                                                  [0, 0])
        else:
            gre_pic[gi], lei_pic[gi], kuangwu[gi] = bianhao_huidu(K_num[gi], (gi + 1), grey_all[gi],
                                                                  midu_avg[gi],
                                                                  midu_avg_k_count[gi], midu_avg_si, midu_avg_ti,
                                                                  midu_avg_k_count[ttk])
        print("gre_pic[(GREY-1)]:", gre_pic[(GREY - 1)])
        print("lei_pic[(GREY-1)]:", lei_pic[(GREY - 1)])
    for i in range(0, GREY):
        are_pic[(i)] = are_pic_same_element(K_num[i], same_elements_list[i], stats_list[i])
        print("are_pic[(GREY-1)]:", are_pic[i])
    # 下面是写入到表格，懒得再写函数了，直接在主函数写了
    n_columns = {0: "Al", 1: "Ca",  2: "F", 3: "Fe", 4: "K", 5: "Mg", 6: "Na", 7: "O",
                 8: "P", 9: "S", 10: "Si", 11: "Ti", 12: "面积占比", 13: "灰度",14:"矿物"}
    for i in range(0, GREY):
        leihao.clear()
        for j in range(K_num[i]):  # 记录类别数量的列表
            leihao.append(j)
        print(leihao)
        bianhao = dict(zip(leihao, lei_pic[i]))  # 转字典，做index
        print(bianhao)
        midu_avg[(i)].append(are_pic[(i)])
        midu_avg[(i)].append(gre_pic[(i)])
        midu_rag[(i)].append(are_pic[(i)])
        midu_rag[(i)].append(gre_pic[(i)])
        midu_avg[(i)].append(kuangwu[(i)])
        midu_rag[(i)].append(kuangwu[(i)])
        if i == 0:  # 灰度1
            if K_num[i] == 1:
                df1_1 = pd.DataFrame(midu_avg[(i)])
                print(df1_1)
                df1_1 = df1_1.T
                print(df1_1)
                df1_1.rename(columns=n_columns, index=bianhao, inplace=True)
                print(df1_1)

                df1_2 = pd.DataFrame(midu_rag[(i)])
                print(df1_2)
                df1_2 = df1_2.T
                print(df1_2)
                df1_2.rename(columns=n_columns, index=bianhao, inplace=True)
                print(df1_2)
                # df1_1=pd.DataFrame(columns=n_columns,index=bianhao,data=(np.transpose(midu_avg[(i)]))) #zip为转置，为什么需要？想知道自己想，不想改就别管它
                # df1_2 = pd.DataFrame(columns=n_columns, index=bianhao, data=(np.transpose(midu_rag[(i)])))
            else:
                df1_1 = pd.DataFrame(midu_avg[(i)])
                print(df1_1)
                df1_1 = df1_1.T
                print(df1_1)
                df1_1.rename(columns=n_columns, index=bianhao, inplace=True)
                print(df1_1)

                df1_2 = pd.DataFrame(midu_rag[(i)])
                print(df1_2)
                df1_2 = df1_2.T
                print(df1_2)
                df1_2.rename(columns=n_columns, index=bianhao, inplace=True)
                print(df1_2)
                # df1_1 = pd.DataFrame(columns=n_columns, index=bianhao,data=(np.transpose(midu_avg[(i)])))  # 多类绘图用
                # df1_2 = pd.DataFrame(columns=n_columns, index=bianhao, data=(np.transpose(midu_rag[(i)])))
                df = pd.DataFrame(df1_1,
                                  columns=["Al", "Ca","F", "Fe", "K", "Mg", "Na","O", "P", "S","Si", "Ti"])
                df_T = df.T
                ax1_2 = df_T.plot(xticks=range(0, elem_count), linestyle=(0, ()), grid=True,
                                  color=plt.get_cmap('tab20')(range(K_num[i])), figsize=(30, 15)).legend(
                    bbox_to_anchor=(1, 1), fontsize=15)
                ax1_1 = df.plot(xticks=range(0, K_num[i]), linestyle=(0, ()), grid=True,
                                color=plt.get_cmap('tab20')(range(elem_count)), figsize=(30, 15)).legend(
                    bbox_to_anchor=(1, 1), fontsize=15)
                # ax1_2.set_prop_cycle('color', colors)
                # ax1_1.set_prop_cycle('color', colors)
                fig1_1 = ax1_1.get_figure()
                fig1_1.savefig(r'biaoge/fig_{}.png'.format(i + 1))
                fig1_2 = ax1_2.get_figure()
                fig1_2.savefig(r'biaoge/fig_{}_T.png'.format(i + 1))
        if i == 1:  # 灰度2
            if K_num[i] == 1:
                df2_1 = pd.DataFrame(midu_avg[(i)])
                print(df2_1)
                df2_1 = df2_1.T
                print(df2_1)
                df2_1.rename(columns=n_columns, index=bianhao, inplace=True)
                print(df2_1)

                df2_2 = pd.DataFrame(midu_rag[(i)])
                print(df2_2)
                df2_2 = df2_2.T
                print(df2_2)
                df2_2.rename(columns=n_columns, index=bianhao, inplace=True)
                print(df2_2)
            # df2_1=pd.DataFrame(columns=n_columns,index=bianhao,data=(np.transpose(midu_avg[(i)])))
            # df2_2 = pd.DataFrame(columns=n_columns, index=bianhao, data=(np.transpose(midu_rag[(i)])))
            else:
                df2_1 = pd.DataFrame(midu_avg[(i)])
                print(df2_1)
                df2_1 = df2_1.T
                print(df2_1)
                df2_1.rename(columns=n_columns, index=bianhao, inplace=True)
                print(df2_1)

                df2_2 = pd.DataFrame(midu_rag[(i)])
                print(df2_2)
                df2_2 = df2_2.T
                print(df2_2)
                df2_2.rename(columns=n_columns, index=bianhao, inplace=True)
                print(df2_2)
                # df2_1 = pd.DataFrame(columns=n_columns, index=bianhao, data=(np.transpose(midu_avg[(i)])))
                # df2_2 = pd.DataFrame(columns=n_columns, index=bianhao, data=(np.transpose(midu_rag[(i)])))
                df = pd.DataFrame(df2_1,
                                  columns=["Al", "Ca","F", "Fe", "K", "Mg", "Na","O", "P", "S","Si", "Ti"])
                df_T = df.T
                ax2_2 = df_T.plot(xticks=range(0, elem_count), linestyle=(0, ()), grid=True,
                                  color=plt.get_cmap('tab20')(range(K_num[i])), figsize=(30, 15)).legend(
                    bbox_to_anchor=(1, 1), fontsize=15)
                ax2_1 = df.plot(xticks=range(0, K_num[i]), linestyle=(0, ()), grid=True,
                                color=plt.get_cmap('tab20')(range(elem_count)), figsize=(30, 15)).legend(
                    bbox_to_anchor=(1, 1), fontsize=15)
                # ax2_2.set_prop_cycle('color', colors)
                # ax2_1.set_prop_cycle('color', colors)
                fig2_1 = ax2_1.get_figure()
                fig2_1.savefig(r'biaoge/fig_{}.png'.format(i + 1))
                fig2_2 = ax2_2.get_figure()
                fig2_2.savefig(r'biaoge/fig_{}_T.png'.format(i + 1))

        if i == 2:  # 灰度3
            if K_num[i] == 1:
                df3_1 = pd.DataFrame(midu_avg[(i)])
                print(df3_1)
                df3_1 = df3_1.T
                print(df3_1)
                df3_1.rename(columns=n_columns, index=bianhao, inplace=True)
                print(df3_1)

                df3_2 = pd.DataFrame(midu_rag[(i)])
                print(df3_2)
                df3_2 = df3_2.T
                print(df3_2)
                df3_2.rename(columns=n_columns, index=bianhao, inplace=True)
                print(df3_2)
                # df3_1=pd.DataFrame(columns=n_columns,index=bianhao,data=(np.transpose(midu_avg[(i)])))
                # df3_2 = pd.DataFrame(columns=n_columns, index=bianhao, data=(np.transpose(midu_rag[(i)])))
            else:
                df3_1 = pd.DataFrame(midu_avg[(i)])
                print(df3_1)
                df3_1 = df3_1.T
                print(df3_1)
                df3_1.rename(columns=n_columns, index=bianhao, inplace=True)
                print(df3_1)

                df3_2 = pd.DataFrame(midu_rag[(i)])
                print(df3_2)
                df3_2 = df3_2.T
                print(df3_2)
                df3_2.rename(columns=n_columns, index=bianhao, inplace=True)
                print(df3_2)
                # df3_1 = pd.DataFrame(columns=n_columns, index=bianhao, data=(np.transpose(midu_avg[(i)])))
                # df3_2 = pd.DataFrame(columns=n_columns, index=bianhao, data=(np.transpose(midu_rag[(i)])))
                df = pd.DataFrame(df3_1,
                                  columns=["Al", "Ca","F", "Fe", "K", "Mg", "Na","O", "P", "S","Si", "Ti"])
                df_T = df.T
                ax3_2 = df_T.plot(xticks=range(0, elem_count), linestyle=(0, ()), grid=True,
                                  color=plt.get_cmap('tab20')(range(K_num[i])), figsize=(30, 15)).legend(
                    bbox_to_anchor=(1, 1), fontsize=15)
                ax3_1 = df.plot(xticks=range(0, K_num[i]), linestyle=(0, ()), grid=True,
                                color=plt.get_cmap('tab20')(range(elem_count)), figsize=(30, 15)).legend(
                    bbox_to_anchor=(1, 1), fontsize=15)
                # ax3_2.set_prop_cycle('color', colors)
                # ax3_1.set_prop_cycle('color', colors)
                fig3_1 = ax3_1.get_figure()
                fig3_1.savefig(r'biaoge/fig_{}.png'.format(i + 1))
                fig3_2 = ax3_2.get_figure()
                fig3_2.savefig(r'biaoge/fig_{}_T.png'.format(i + 1))
        if i == 3:  # 灰度4
            if K_num[i] == 1:
                df4_1 = pd.DataFrame(midu_avg[(i)])
                print(df4_1)
                df4_1 = df4_1.T
                print(df4_1)
                df4_1.rename(columns=n_columns, index=bianhao, inplace=True)
                print(df4_1)

                df4_2 = pd.DataFrame(midu_rag[(i)])
                print(df4_2)
                df4_2 = df4_2.T
                print(df4_2)
                df4_2.rename(columns=n_columns, index=bianhao, inplace=True)
                print(df4_2)
                # df4_1=pd.DataFrame(columns=n_columns,index=bianhao,data=(np.transpose(midu_avg[(i)])))
                # df4_2 = pd.DataFrame(columns=n_columns, index=bianhao, data=(np.transpose(midu_rag[(i)])))
            else:
                df4_1 = pd.DataFrame(midu_avg[(i)])
                print(df4_1)
                df4_1 = df4_1.T
                print(df4_1)
                df4_1.rename(columns=n_columns, index=bianhao, inplace=True)
                print(df4_1)

                df4_2 = pd.DataFrame(midu_rag[(i)])
                print(df4_2)
                df4_2 = df4_2.T
                print(df4_2)
                df4_2.rename(columns=n_columns, index=bianhao, inplace=True)
                print(df4_2)
                # df4_1 = pd.DataFrame(columns=n_columns, index=bianhao, data=(np.transpose(midu_avg[(i)])))
                # df4_2 = pd.DataFrame(columns=n_columns, index=bianhao, data=(np.transpose(midu_rag[(i)])))
                df = pd.DataFrame(df4_1,
                                  columns=["Al", "Ca","F", "Fe", "K", "Mg", "Na","O", "P", "S","Si", "Ti"])
                df_T = df.T
                ax4_2 = df_T.plot(xticks=range(0, elem_count), linestyle=(0, ()), grid=True,
                                  color=plt.get_cmap('tab20')(range(K_num[i])), figsize=(30, 15)).legend(
                    bbox_to_anchor=(1, 1), fontsize=15)
                ax4_1 = df.plot(xticks=range(0, K_num[i]), linestyle=(0, ()), grid=True,
                                color=plt.get_cmap('tab20')(range(elem_count)), figsize=(30, 15)).legend(
                    bbox_to_anchor=(1, 1), fontsize=15)
                # ax4_2.set_prop_cycle('color', colors)
                # ax4_1.set_prop_cycle('color', colors)
                fig4_1 = ax4_1.get_figure()
                fig4_1.savefig(r'biaoge/fig_{}.png'.format(i + 1))
                fig4_2 = ax4_2.get_figure()
                fig4_2.savefig(r'biaoge/fig_{}_T.png'.format(i + 1))
        if i == 4:  # 灰度5
            if K_num[i] == 1:
                df5_1 = pd.DataFrame(midu_avg[(i)])
                print(df5_1)
                df5_1 = df5_1.T
                print(df5_1)
                df5_1.rename(columns=n_columns, index=bianhao, inplace=True)
                print(df5_1)

                df5_2 = pd.DataFrame(midu_rag[(i)])
                print(df5_2)
                df5_2 = df5_2.T
                print(df5_2)
                df5_2.rename(columns=n_columns, index=bianhao, inplace=True)
                print(df5_2)
                # df5_1=pd.DataFrame(columns=n_columns,index=bianhao,data=(np.transpose(midu_avg[(i)])))
                # df5_2 = pd.DataFrame(columns=n_columns, index=bianhao, data=(np.transpose(midu_rag[(i)])))
            else:
                df5_1 = pd.DataFrame(midu_avg[(i)])
                print(df5_1)
                df5_1 = df5_1.T
                print(df5_1)
                df5_1.rename(columns=n_columns, index=bianhao, inplace=True)
                print(df5_1)

                df5_2 = pd.DataFrame(midu_rag[(i)])
                print(df5_2)
                df5_2 = df5_2.T
                print(df5_2)
                df5_2.rename(columns=n_columns, index=bianhao, inplace=True)
                print(df5_2)
                # df5_1 = pd.DataFrame(columns=n_columns, index=bianhao, data=(np.transpose(midu_avg[(i)])))
                # df5_2 = pd.DataFrame(columns=n_columns, index=bianhao, data=(np.transpose(midu_rag[(i)])))
                df = pd.DataFrame(df5_1,
                                  columns=["Al", "Ca","F", "Fe", "K", "Mg", "Na","O", "P", "S","Si", "Ti"])
                df_T = df.T
                ax5_2 = df_T.plot(xticks=range(0, elem_count), linestyle=(0, ()), grid=True,
                                  color=plt.get_cmap('tab20')(range(K_num[i])), figsize=(30, 15)).legend(
                    bbox_to_anchor=(1, 1), fontsize=15)
                ax5_1 = df.plot(xticks=range(0, K_num[i]), linestyle=(0, ()), grid=True,
                                color=plt.get_cmap('tab20')(range(elem_count)), figsize=(30, 15)).legend(
                    bbox_to_anchor=(1, 1), fontsize=15)
                # ax5_2.set_prop_cycle('color', colors)
                # ax5_1.set_prop_cycle('color', colors)
                fig5_1 = ax5_1.get_figure()
                fig5_1.savefig(r'biaoge/fig_{}.png'.format(i + 1))
                fig5_2 = ax5_2.get_figure()
                fig5_2.savefig(r'biaoge/fig_{}_T.png'.format(i + 1))
        if i == 5:  # 灰度6
            if K_num[i] == 1:
                df6_1 = pd.DataFrame(midu_avg[(i)])
                print(df6_1)
                df6_1 = df6_1.T
                print(df6_1)
                df6_1.rename(columns=n_columns, index=bianhao, inplace=True)
                print(df6_1)

                df6_2 = pd.DataFrame(midu_rag[(i)])
                print(df6_2)
                df6_2 = df6_2.T
                print(df6_2)
                df6_2.rename(columns=n_columns, index=bianhao, inplace=True)
                print(df6_2)
                # df6_1=pd.DataFrame(columns=n_columns,index=bianhao,data=(np.transpose(midu_avg[(i)])))
                # df6_2 = pd.DataFrame(columns=n_columns, index=bianhao, data=(np.transpose(midu_rag[(i)])))
            else:
                df6_1 = pd.DataFrame(midu_avg[(i)])
                print(df6_1)
                df6_1 = df6_1.T
                print(df6_1)
                df6_1.rename(columns=n_columns, index=bianhao, inplace=True)
                print(df6_1)

                df6_2 = pd.DataFrame(midu_rag[(i)])
                print(df6_2)
                df6_2 = df6_2.T
                print(df6_2)
                df6_2.rename(columns=n_columns, index=bianhao, inplace=True)
                print(df6_2)
                # df6_1 = pd.DataFrame(columns=n_columns, index=bianhao, data=(np.transpose(midu_avg[(i)])))
                # df6_2 = pd.DataFrame(columns=n_columns, index=bianhao, data=(np.transpose(midu_rag[(i)])))
                df = pd.DataFrame(df6_1,
                                  columns=["Al", "Ca","F", "Fe", "K", "Mg", "Na","O", "P", "S","Si", "Ti"])
                df_T = df.T
                ax6_2 = df_T.plot(xticks=range(0, elem_count), linestyle=(0, ()), grid=True,
                                  color=plt.get_cmap('tab20')(range(K_num[i])), figsize=(30, 15)).legend(
                    bbox_to_anchor=(1, 1), fontsize=15)
                ax6_1 = df.plot(xticks=range(0, K_num[i]), linestyle=(0, ()), grid=True,
                                color=plt.get_cmap('tab20')(range(elem_count)), figsize=(30, 15)).legend(
                    bbox_to_anchor=(1, 1), fontsize=15)
                # ax6_2.set_prop_cycle('color', colors)
                # ax6_1.set_prop_cycle('color', colors)
                fig6_1 = ax6_1.get_figure()
                fig6_1.savefig(r'biaoge/fig_{}.png'.format(i + 1))
                fig6_2 = ax6_2.get_figure()
                fig6_2.savefig(r'biaoge/fig_{}_T.png'.format(i + 1))
        bianhao.clear()
    if (GREY == 3):  # 3灰度
        df1 = pd.concat([df1_1, df2_1, df3_1], axis=0, join='outer')  # axis=0时合并结果为相同列名的数据
        df2 = pd.concat([df1_2, df2_2, df3_2], axis=0, join='outer')  # axis=0时合并结果为相同列名的数据
    if (GREY == 4):  # 4灰度
        df1 = pd.concat([df1_1, df2_1, df3_1, df4_1], axis=0, join='outer')  # axis=0时合并结果为相同列名的数据
        df2 = pd.concat([df1_2, df2_2, df3_2, df4_2], axis=0, join='outer')  # axis=0时合并结果为相同列名的数据
    if (GREY == 5):  # 5灰度
        df1 = pd.concat([df1_1, df2_1, df3_1, df4_1, df5_1], axis=0, join='outer')  # axis=0时合并结果为相同列名的数据
        df2 = pd.concat([df1_2, df2_2, df3_2, df4_2, df5_2], axis=0, join='outer')  # axis=0时合并结果为相同列名的数据
    if (GREY == 6):  # 6灰度
        df1 = pd.concat([df1_1, df2_1, df3_1, df4_1, df5_1, df6_1], axis=0, join='outer')  # axis=0时合并结果为相同列名的数据
        df2 = pd.concat([df1_2, df2_2, df3_2, df4_2, df5_2, df6_2], axis=0, join='outer')  # axis=0时合并结果为相同列名的数据
    df1.index.name = '编号'
    df2.index.name = '编号'
    df1.columns.name = '元素'
    df2.columns.name = '元素'
    df1.to_csv(r'biaoge/汇总表—-密度均值.csv', encoding='gbk')  # 方便画图用的，方不方便看我不知道，还没运行过
    df2.to_csv(r'biaoge/汇总表—-密度范围.csv', encoding='gbk')

    P_spots = [[0 for col in range(1)] for row in range(6)]
    P_blocks = [[0 for col in range(1)] for row in range(6)]
    P_elements = list()
    P_img0 = cv2.imread(r'yuan/{}.png'.format(img_name_org))
    P_output1 = np.zeros((P_img0.shape[0], P_img0.shape[1], 3), np.uint8)
    # 读入图片
    P_img2 = cv2.imread(r'duibi2/P.png')  # 这是P元素图
    P_img2 = cv2.resize(P_img2, (w_img0, h_img0), interpolation=cv2.INTER_CUBIC)
    P_org_img_folder = r'Grayscale'  # 这是多张提取，灰度图所在文件夹
    # 检索文件
    P_imglist = getFileList(P_org_img_folder, [], 'png')  # 调用getFileList()函数
    # print('本次执行检索到 ' + str(len(imglist)) + ' 张图像\n') # 测试用，看看图片遍历有没有问题
    for P_imgpath in P_imglist:
        P_imgname = os.path.splitext(os.path.basename(P_imgpath))[0]
        P_img = cv2.imread(P_imgpath, cv2.IMREAD_COLOR)  # 对每幅图像执行相关操作
        P_connected_test(P_img)
        P_element, P_spots[P_m], P_blocks[P_m] = P_mask_test(P_img, P_img2, P_stats, P_num_labels, P_m, P_labels)
        P_elements.append(P_element)
        # del (spots[P_m][0]) # 这是删除m元素下初始化的数
        del (P_blocks[P_m][0])  # 这是删除初始化的数
        P_output1 = P_re_mark(P_img, P_labels, P_blocks, P_m, P_output1)
        # print(P_blocks[m])
        P_m += 1
    cv2.imwrite(r'same_element/P/all.png', P_output1)
    print("saving all same_elements of P picture sucessfully~")



# Copyright (c) 2025 UUI-D
# Licensed under the MIT License. See LICENSE for details.
