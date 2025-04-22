import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
from scipy.io import loadmat,savemat
from util import crop_center,crop_custom, Visualize3D, minmax_normalize, rand_crop,BandMinMaxQuantileStateful
import os
import numpy as np
import h5py
from os.path import join, exists
from scipy.io import loadmat, savemat

from util import crop_center,crop_custom, Visualize3D, minmax_normalize, rand_crop,BandMinMaxQuantileStateful
from PIL import Image
import torch
import spectral
from skimage import io
import cv2 
from PIL import Image, ImageDraw

# 由 mat文件开始处理， 通过rgb.三通道 生成原始图，根目录 数据集/噪声类型/方法/.mat
# 选那种图，09 ，传统方法也要生成图 
# 一个放大细节 ， 左上，右下

def normalize_list(input_list):
    min_val = min(input_list)
    max_val = max(input_list)
    normalized_list = [(x - min_val) / (max_val - min_val) for x in input_list]
    return normalized_list

def get_india_rgb():

    save_path = "/home/jiahua/liuy/hsi_pipeline/draw_pic/india/"
    test_path = '/data1/jiahua/ly/india_all_method_mat/'
     
    rect_list = os.listdir(test_path)

    for rect in rect_list:

        img = loadmat(os.path.join(test_path,rect))  
        print(img.keys())
        data = img['output'][...]
        result_gt = np.zeros([128,128,3], dtype=np.float64)
        # for i in range(31):
            # result_gt[:,:,1] = data[:,:,i]
        result_gt[:,:,0] = data[:,:,2]
        result_gt[:,:,1] = data[:,:,24]
        result_gt[:,:,2] = data[:,:,127]
        result_gt = minmax_normalize(result_gt)
        result_gt = result_gt * 255
        cv2.imwrite(save_path+rect[:-4]+'.png',result_gt)
 
def make_erf():
    
    test_path = "/data3/jiahua/ly/test_data/kaist_1024_complex/1024_deadline/scene27_reflectanceNW.mat"
    save_path = '/data3/jiahua/ly/test_data/kaist_1024_complex/erf_test/test'
    
    img = loadmat(test_path)
    gt = img['gt']
    input = img['input']
    print(gt.shape)
    gt = gt[:256,:256,:]
    input = input[:256,:256,:]
    savemat(save_path + '.mat', {'gt':gt,'input': input})   

def getchannels():
    img = loadmat('/cvlabdata1/ly/hsi_data/kaist/ori_dataset/kaist_2048_complex/512_mixture/scene09_reflectance.mat')
    print(img.keys())
    data = img['input'][...]
    result_gt = np.zeros([2048,2048,3], dtype=np.float64)
    result_gt[:,:,0] = data[:,:,9]
    result_gt[:,:,1] = data[:,:,19]
    result_gt[:,:,2] = data[:,:,29]
    result_gt = result_gt * 255
    cv2.imwrite('/home/liuy/projects/hsi_pipeline/montage_imgs/final_input//scene09_mixture_g.png',result_gt[:,:,1])

def getnoise_only():
    gt = cv2.imread("/home/liuy/projects/hsi_pipeline/montage_imgs/final_input/scene09_11121_sp1.png")
    input = cv2.imread("/home/liuy/projects/hsi_pipeline/montage_imgs/final_input/scene09_mixture1123.png")
    noise = input - gt 
    cv2.imwrite("/home/liuy/projects/hsi_pipeline/montage_imgs/noisy_only_mixture/scene09_1123_mixtureonly.png",noise)

def get_india_rgb():

    save_path = "/home/jiahua/liuy/hsi_pipeline/draw_pic/india/"
    test_path = '/data1/jiahua/ly/india_all_method_mat/'
     
    rect_list = os.listdir(test_path)

    for rect in rect_list:

        img = loadmat(os.path.join(test_path,rect))  
        print(img.keys())
        data = img['output'][...]
        result_gt = np.zeros([128,128,3], dtype=np.float64)
        # for i in range(31):
            # result_gt[:,:,1] = data[:,:,i]
        result_gt[:,:,0] = data[:,:,2]
        result_gt[:,:,1] = data[:,:,24]
        result_gt[:,:,2] = data[:,:,127]
        result_gt = minmax_normalize(result_gt)
        result_gt = result_gt * 255
        cv2.imwrite(save_path+rect[:-4]+'.png',result_gt)
        

def get_urban_rgb():

    save_path = "/home/jiahua/liuy/hsi_pipeline/draw_pic/urban/"
    test_path = '/data1/jiahua/ly/urban/'

    # test_path = test_path+method
     
    rect_list = os.listdir(test_path)

    for rect in rect_list:

        img = loadmat(os.path.join(test_path,rect))  
        print(img.keys())
        data = img['data'][...]
        # data = data.transpose((2,1,0))
        # data = data.transpose((1,0,2))

        result_gt = np.zeros([256,256,3], dtype=np.float64)

        tmp = data[:,:,0:31]
        tmp = minmax_normalize(tmp)
        result_gt[:,:,0] = tmp[:,:,15]
        tmp = data[:,:,93:124]
        tmp = minmax_normalize(tmp)
        result_gt[:,:,1] = tmp[:,:,12]
        tmp = data[:,:,179:210]
        tmp = minmax_normalize(tmp)
        result_gt[:,:,2] = tmp[:,:,16]

        result_gt = result_gt * 255
        # cv2.imwrite(save_path+rect[:-4]+'.png',result_gt)
        cv2.imwrite(save_path+'gt.png',result_gt)


# Spectral curve compare // 光谱曲线对比图
def draw_2():
    import random
    img = loadmat("/data1/jiahua/ly/test_data/wdc_complex/512_mixture/wdc.mat")
    print(img.keys())
    data = img['gt'][...]
    # for i in range(10):
    #     spectral.save_rgb("/home/jiahua/HSI-CVPR/hsi_pipeline/result/wdc/sequential/"+str(i)+".png",data,bands=(19*i,19*i+9,19*i+18))
    normalize_img = minmax_normalize(data)
    # normalize_img = data * 255
    point_1 = normalize_img[162,87,:150]
    
    point_2_1 = normalize_img[12,150,:150]
    point_2_2 = normalize_img[25,158,:150] 
    
    point_7_1 = normalize_img[194,38,:150]
    point_7_2 = normalize_img[222,58,:150] 
    
    point_3 = normalize_img[34,118,:150]
    point_4 = normalize_img[115,34,:150]
    
    point_5 = normalize_img[116,1,:150]
    point_6 = normalize_img[150,152,:150]
    
    # max_index = np.unravel_index(np.argmax(normalize_img), normalize_img.shape)
    # print(max_index)
    
    Bands = np.arange(150)
    # 创建一个折线图
    plt.figure(figsize=(8, 2),dpi=400)
    plt.rcParams['font.sans-serif'] = 'times new roman'
    
    plt.plot(Bands, point_7_1 , label='Position C' , color='#FF0000' , marker = 'o' ,markersize=1)
    plt.plot(Bands, point_7_2 , label='Position D' , color='#0070C0' , marker = 'o' ,markersize=1)
    from matplotlib.font_manager import FontProperties
    font = FontProperties()
    font.set_weight('bold')

    # 添加图例，并设置字体大小和加粗
    # plt.legend()
    # plt.plot(Bands, point_3 , label='point_3')

    ax = plt.gca()
    x1_label = ax.get_xticklabels() 
    [x1_label_temp.set_fontname('Times New Roman') for x1_label_temp in x1_label]
    y1_label = ax.get_yticklabels() 
    [y1_label_temp.set_fontname('Times New Roman') for y1_label_temp in y1_label]

    
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    
    # xticks = [10, 20, 30]
    # plt.xticks(xticks)
    
    # plt.ylim(0, 1)
    # 添加标题和标签
    # plt.xlabel(r'$\mathbf{Band}$')
    plt.ylabel(r'$\mathbf{Normalize\ pixel}$')
    plt.ylabel(r'$\mathbf{Normalize\ pixel}$')
    
    plt.legend(loc='upper right',fontsize='32', prop=font)

    # 保存图片为PNG文件
    plt.savefig('/home/jiahua/HSI-CVPR/hsi_pipeline/result/wdc/point_bands/agg_2.png')
    
    return 0
    
def get_real_rgb():

    # img = loadmat("")
    # print(img.keys())
    # data = img['data'][...]
    # print(data.shape)
    # spectral.save_rgb("/home/jiahua/HSI-CVPR/hsi_pipeline/result/urban/s2s.png",data,bands=(102, 138, 202))

    image_A = cv2.imread('/home/jiahua/HSI-CVPR/hsi_pipeline/result/urban/noisy.png')
    image_B = cv2.imread('/home/jiahua/HSI-CVPR/hsi_pipeline/result/urban/s2s.png')

    print(image_A.shape)
    w, h ,c = image_A.shape

    result_gt = np.zeros([224,224,3], dtype=np.float64)

    for i in range(w):
        result_gt[i,0:w-i] = image_A[i,0:w-i]
        result_gt[i,w-i:w] = image_B[i,w-i:w]

    cv2.imwrite('/home/jiahua/HSI-CVPR/hsi_pipeline/result/urban/fake.png',result_gt)


if __name__ == '__main__':

    draw_2()

    # img = loadmat("/data1/jiahua/result/urban/hsdt_s.mat")
    # print(img.keys())
    # data = img['output'][...]
    # data = data.transpose((2,1,0))
    # data = data.transpose((1,0,2))
    # print(data.shape)
    # spectral.save_rgb("/home/jiahua/HSI-CVPR/hsi_pipeline/figure/urban/hsdt.png",data,bands=(102,138 , 20))


    # img = loadmat('/data1/jiahua/result/wdc/wdc_LLRGTV.mat')
    # print(img.keys())
    # data = img['output_image'][...]
    # savemat('/data1/jiahua/result/wdc/wdc_LLRGTV.mat', {'data': data})
    # test_path= "/home/jiahua/HSI-CVPR/hsi_pipeline/figure/wdc_res/"
     
    # rect_list = os.listdir(test_path)

    # for rect in rect_list:
    #     data = cv2.imread(test_path+rect)
    #     data = cv2.resize(data,(256,256))
    #     cv2.imwrite(test_path+rect,data)
        # print(data.shape)
    
    # savemat('/data1/jiahua/result/urban/LRTDTV.mat', {'data': data})
    
    # data = img['gt'][...]
    # result_gt = np.zeros([2048,2048,3], dtype=np.float64)
    # result_gt[:,:,0] = data[:,:,0]
    # # result_gt[:,:,0] = data[:,:,9]
    # # result_gt[:,:,1] = data[:,:,19]
    # # result_gt[:,:,2] = data[:,:,29]
    # result_gt = result_gt * 255
    # cv2.imwrite('/home/liuy/projects/hsi_pipeline/montage_imgs/09_test/scene09_reflectance_gt_0.png',result_gt[:,:,0])
    # cv2.imwrite('/home/liuy/projects/hsi_pipeline/montage_imgs/09_test/scene09_reflectance_mixture_9.png',result_gt[:,:,0])
    # cv2.imwrite('/home/liuy/projects/hsi_pipeline/montage_imgs/09_test/scene09_reflectance_mixture_19.png',result_gt[:,:,1])
    # cv2.imwrite('/home/liuy/projects/hsi_pipeline/montage_imgs/09_test/scene09_reflectance_mixture_29.png',result_gt[:,:,2])

    # img = loadmat('/cvlabdata1/ly/hsi_data/kaist/ori_dataset/kaist_2048_complex/512_stripe/scene09_reflectance.mat')
    # print(img.keys())
    # data = img['input'][...]
    # result_gt = np.zeros([2048,2048,3], dtype=np.float64)
    # for i in range(31):
    #     result_gt[:,:,1] = data[:,:,i]
    # # result_gt[:,:,0] = data[:,:,5]
    # # result_gt[:,:,1] = data[:,:,15]
    # # result_gt[:,:,2] = data[:,:,20]
    #     result_gt = result_gt * 255
    #     cv2.imwrite('/home/liuy/projects/hsi_pipeline/montage_imgs/noisy_sp/scene09_noise_'+str(i)+'.png',result_gt)

    
    # img = loadmat('/data1/jiahua/test_data/indian_mat/indian_pines.mat')
    # print(img.keys())
    # data = img['pavia'][...]
    # data = data[0:128,0:128,:]
    # savemat("/data1/jiahua/ly/india/india1.mat",{'data':data})
    # # data = data.transpose((2,1,0))
    # result_gt = np.zeros([128,128,3], dtype=np.float64)
    # # for i in range(31):
    # #     result_gt[:,:,1] = data[:,:,i]
    # test1 = data[:,:,0:30]
    # test2 = data[:,:,124:155]
    # test1 = minmax_normalize(test1)
    # test2 = minmax_normalize(test2)
    # result_gt[:,:,0] = test1[:,:,2]
    # result_gt[:,:,1] = test1[:,:,24]
    # result_gt[:,:,2] = test2[:,:,3]
    # result_gt = result_gt * 255
    # # result_gt = np.rot90(result_gt)
    # # result_gt = result_gt[::-1]
    # cv2.imwrite('/home/jiahua/liuy/hsi_pipeline/draw_pic/india/india.png',result_gt)


    # data = loadmat("/data1/jiahua/ly/cave_test_complex/512_mixture/chart_and_stuffed_toy_ms.mat")
    # data = data['input']
    # result_gt = np.zeros([512,512,3], dtype=np.float64)
    # # for i in range(31):
    # #     result_gt[:,:,1] = inputs[:,:,i]
    # result_gt[:,:,0] = data[:,:,9]
    # result_gt[:,:,1] = data[:,:,19]
    # result_gt[:,:,2] = data[:,:,29]
    # result_gt = result_gt * 255
    # cv2.imwrite(os.path.join('/home/jiahua/liuy/hsi_pipeline/draw_pic/cave','noisy.png'),result_gt)