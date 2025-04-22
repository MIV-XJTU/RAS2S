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
from indexes import cal_bwpsnr
from PIL import Image
import torch
import spectral
from skimage import io
import cv2 

# get different noise picture from direct image name of CAVE datasers
def save_noise(name):
    noise_folder = ['/data3/jiahua/ly/test_data/cave_test_complex/512_stripe',
                    '/data3/jiahua/ly/test_data/cave_test_complex/512_noniid',
                    '/data3/jiahua/ly/test_data/cave_test_complex/512_mixture',
                    '/data3/jiahua/ly/test_data/cave_test_complex/512_impulse',
                    '/data3/jiahua/ly/test_data/cave_test_complex/512_deadline',
                    '/data3/jiahua/ly/test_data/cave_added_noise/512_speckle',
                    '/data3/jiahua/ly/test_data/cave_added_noise/512_poisson'
                    ]
    for noise in noise_folder:
        
        img = loadmat(os.path.join(noise,name))
        # print(img.keys())
        gt = img['gt'][...]
        noisy = img['input'][...]
        gt = gt[217:377,258:418,:]
        noisy = noisy[217:377,258:418,:]
        noise_type = noise.split('_')[-1]
        _name = name.split('.')[0]
        spectral.save_rgb("/home/jiahua/HSID/result/noise_exp/"+_name+'_gt.png',gt,bands=(29,19,9))
        spectral.save_rgb("/home/jiahua/HSID/result/noise_exp/"+_name+'_'+noise_type+".png",noisy,bands=(29,19,9))

def get_rgb():

    band = 29
    datasets = 'wdc'

    save_path = "/home/jiahua/HSI-CVPR/hsi_pipeline/figure/" + datasets + '_test'
    test_path = '/data1/jiahua/result/' + datasets
     
    rect_list = os.listdir(test_path)

    for rect in rect_list:

        img = loadmat(os.path.join(test_path,rect))  
        print(rect)
        if datasets == 'urban/':
            if rect == 'urban_LLRGTV.mat':
                # print(img.keys())
                data = img['output_image'][...]
            else:
                # print(img.keys())
                data = img['data'][...]
            if 'gru' in rect or 's2s' in rect:
                data = data.transpose((2,1,0))
            if 's2s' in rect:
                data = data.transpose((1,0,2))
            spectral.save_rgb(save_path + rect[:-4] +".png",data,bands=(102,138, 202))
        if datasets == 'real_40/':
            data = img['data'][...]
            if 'real' not in rect:
                data = data.transpose((2,1,0))
                data = data.transpose((1,0,2))
            data = np.rot90(data)
            # print(data.shape)
            # spectral.save_rgb(save_path + rect[:-4] +".png",data,bands=(5,11,16))
            result_gt = np.zeros([512,512,1], dtype=np.float64)
            result_gt[:,:,0] = data[:,:,band]
            if 'real_40_NGMeet.mat' == rect or 'real_40_LLRT.mat' == rect:
                result_gt =result_gt
            else:
                result_gt = result_gt * 255
            cv2.imwrite(save_path + rect[:-4] +".png",result_gt)
        if datasets == 'wdc':
            data = img['data'][...]
        
        result_gt = np.zeros([256,256,3], dtype=np.float64)
        result_gt[:,:,0] = data[:,:,9]
        result_gt[:,:,1] = data[:,:,19]
        result_gt[:,:,2] = data[:,:,29]
        result_gt = minmax_normalize(result_gt)
        result_gt = result_gt * 255
        cv2.imwrite(save_path + rect[:-4] +".png",result_gt)

def get_res():
    save_path = "/home/jiahua/HSI-CVPR/hsi_pipeline/figure/wdc_res/"
    test_path = '/data1/jiahua/result/wdc/'

    noise = loadmat('/data1/jiahua/result/wdc/noisy')
    gt = loadmat('/data1/jiahua/result/wdc/gt') 
    noise = noise['data'][...]
    gt = gt['data'][...]
    print(gt.shape)

    result = np.zeros([256,256,3], dtype=np.float64)
    rect_list = os.listdir(test_path)

    for rect in rect_list:

        img = loadmat(os.path.join(test_path,rect))

        data = img['data'][...]
        result[:,:,0] = -(data[:,:,76] - gt[:,:,76])
        result[:,:,1] = -(data[:,:,43] - gt[:,:,43])
        result[:,:,2] = -(data[:,:,10] - gt[:,:,10])

        # result = np.abs(result)
        # result = minmax_normalize(result)
        # result = result * 255
        import seaborn as sns
        plt.figure(figsize=(6, 6))
        sns.heatmap(result[:,:,1], annot=False, fmt='f', cmap='viridis',cbar=False,xticklabels=False, yticklabels=False)    
        plt.gca().set_aspect('equal', adjustable='box')
        fig = plt.gcf() 
        # fig.colorbar = False
        fig.savefig(save_path+rect[:-4]+".png",bbox_inches='tight', pad_inches=0,dpi=300)
        # fig.close()
        # cv2.imwrite(save_path+rect[:-4]+".png", result)
        print(rect)

def drawRect(dataname,area):

    test_path= "/home/jiahua/liuy/hsi_pipeline/draw_pic/" + str(dataname) + "/"

    save_path= '/home/jiahua/liuy/hsi_pipeline/draw_pic/' + str(dataname) + "_big/"
     
    rect_list = os.listdir(test_path)

    for rect in rect_list:
        
        i=j=lengh=lengw=0
        rate = 0
        
        if dataname == "kaist":
            lengh= 256
            lengw = 256
            i = 315 #要放大区域的左上角的 x 坐标（竖轴坐标）
            j = 1788 #要放大区域的左上角的 y 坐标（横轴坐标）
            rate = 8

        elif dataname == "cave":
            lengh= 64
            lengw = 64
            i = 265 #要放大区域的左上角的 x 坐标（竖轴坐标）
            j = 374 #要放大区域的左上角的 y 坐标（横轴坐标）
            rate = 8
            
        elif dataname == "icvl" or dataname == "icvl_img": # 276,53
            lengh= 64
            lengw = 64
            i = 73 #要放大区域的左上角的 x 坐标（竖轴坐标）
            j = 95 #要放大区域的左上角的 y 坐标（横轴坐标）
            rate = 8
        
        elif dataname == "real_15":
            lengh= 64
            lengw = 64
            i = 118 #要放大区域的左上角的 x 坐标（竖轴坐标）
            j = 323 #要放大区域的左上角的 y 坐标（横轴坐标）
            rate = 4

        elif dataname == "real_40":
            lengh= 64
            lengw = 64
            i = 122 #要放大区域的左上角的 x 坐标（竖轴坐标）
            j = 321 #要放大区域的左上角的 y 坐标（横轴坐标）
            rate = 8
        
        elif dataname == "india":
            lengh= 32
            lengw = 32
            i = 2 #要放大区域的左上角的 x 坐标（竖轴坐标）
            j = 24 #要放大区域的左上角的 y 坐标（横轴坐标）
            rate = 2

        elif dataname == "urban":
            lengh= 28
            lengw = 28
            i = 93 #要放大区域的左上角的 x 坐标（竖轴坐标）90 97
            j = 71 #要放大区域的左上角的 y 坐标（横轴坐标）66 73
            rate = 8

        elif dataname == "wdc" or dataname == "wdc_res":
            lengh= 32
            lengw = 32
            i = 160 #要放大区域的左上角的 x 坐标（竖轴坐标）97
            j = 77 #要放大区域的左上角的 y 坐标（横轴坐标）73
            rate = 8

        # rate = rate * 2
        img = cv2.imread(os.path.join(test_path,rect), -1)  #在cv2.imread参数中加入-1，表示不改变读取图像的类型直接读取
        print(img.shape)

        if len(img.shape) == 2:
            H,W = img.shape
        else:
            H,W,C = img.shape

        # # 复制灰度图像以便不影响原始图像
        # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        shift =1
        h = i+lengh  #要放大区域的高
        w = j+lengw  #要放大区域的宽
        h_2 = rate*lengh
        w_2 = rate*lengw

        pt1_1 = (j, i)  # 长方形框左上角坐标
        pt2_1 = (w, h)  # 长方形框右下角坐标 

        patch = img[ i:h,j:w]  # numpy 里先x，后y，x轴沿垂直方向向下，y轴沿水平方向向右
        # plt.imshow(patch)
        patch = cv2.resize(patch, (w_2, h_2))  #(w, h)
        
        pt1 = (0,0)
        pt2 = (h_2,w_2)

        if area == 1:     # 左上
            img[0:h_2,0:w_2] = patch
        elif area == 2:    # 右上
            img[0:h_2,0:w_2] = patch
        elif area == 3:    # 左下
            img[H-h_2:H,0:w_2] = patch
            pt1 = (0,H-h_2)
            pt2 = (w_2,H)
        elif area == 4:    # 右下
            img[H-h_2:H,W-w_2:W] = patch
            pt1 = (w_2,h_2)
            pt2 = (W,H)

        # # 256 3 , 512 6 , 1024 , 24
        # lineWidth = 6 # int (lengh/32/rate *12)
        # cv2.rectangle(img, pt1_1, pt2_1, (0,0,255), lineWidth)  # cv2 里也是先x，后y，x轴沿水平方向向右，y轴沿垂直方向向下
        # cv2.rectangle(img, pt1, pt2, (0,0,255), lineWidth)  # cv2 里也是先x，后y，x轴沿水平方向向右，y轴沿垂直方向向下

        # 定义红色（BGR 格式）的颜色
        red_color = (0, 0, 255)

        save_path_custom = save_path + rect
        img = cv2.resize(img,(256,256))
        cv2.imwrite(save_path_custom, img)

def residual_noise(test_path):

    noise = Image.open(test_path + "/noise.png")
    gt = Image.open(test_path + "/gt.png")

    save_path= '/home/jiahua/HSI-CVPR/hsi_pipeline/figure/icvl_img_big_res/'

    result_gt = np.zeros([128,128,3], dtype=np.float64)
     
    rect_list = os.listdir(test_path)

    for rect in rect_list:

        # 打开两个图像文件

        image = Image.open(test_path+rect)

        # 确保两个图像具有相同的大小
        if noise.size != image.size:
            raise ValueError("两个图像的大小不一致")

        # 将图像转换为 NumPy 数组
        array1 = np.array(noise)
        array2 = np.array(image)
        array3 = np.array(gt)

        # 进行像素相减
        result_array = np.abs(array3 - array2)

        # 创建一个 PIL 图像对象
        result_image = Image.fromarray(result_array)

        # 保存结果图像为 PNG
        result_image.save(save_path+rect)

def result_resize():

    new_path = '/data1/jiahua/result/supplyment_data/icvl_result/'
    rect_list = os.listdir(new_path)

    for rect in rect_list:
       
        img = cv2.imread(os.path.join(new_path,rect))
        img = cv2.resize(img,(256,256))
        cv2.imwrite(os.path.join(new_path,rect),img)

def get_band_psnr():

    test_path = '/data1/jiahua/result/wdc/'

    gt = loadmat('/data1/jiahua/result/wdc/gt') 
    gt = gt['data'][...]
    # gt = gt.transpose((2,1,0))
    # print(gt.shape)
    rect_list = os.listdir(test_path)

    for rect in rect_list:
        print(rect)
        img = loadmat(os.path.join(test_path,rect)) 
        img = img['data'][...]
        print(img.shape)
        print(img[1,1,1])

        if 'LLRT' in rect :
            img = img / 2000
        elif  'NGMeet' in rect:
            img = img / 255
        # img = img.transpose((2,1,0))
        psnr = []
        c,h,w=gt.shape
                
        for k in range(c):
            psnr.append(10*np.log10((h*w)/sum(sum((gt[k]-img[k])**2))))
        # gt=torch.tensor(gt)
        # img=torch.tensor(img)
        # psnr = np.mean(cal_bwpsnr(gt, img))

        print(sum(psnr)/len(psnr))
        # print(psnr)

def rename_folds(path):
    rect_list = os.listdir(path)

    for rect in rect_list:
        # print(os.path.join(path,rect.split("_")[0]+'.png'))
        # os.rename(os.path.join(path,rect),os.path.join(path,rect.split("_")[0]+'.png'))
        os.rename(os.path.join(path,rect),os.path.join(path,rect.split(".png")[0]+'.png'))

def splitImage(path):

    save_path = '/home/jiahua/HSID/result/split_img'
    
    img = cv2.imread(path)
    print(img.shape)
    H,W,_ = img.shape
    split_scale = 3
    
    for i in range(split_scale):
        for j in range(split_scale):
            tmp = img[i*(H//split_scale):(i+1)*(H//split_scale),j*(W//split_scale):(j+1)*(W//split_scale),:]
            print(tmp.shape)
            cv2.imwrite(os.path.join(save_path,str(i)+'_'+str(j)+".png"),tmp)
        
    
if __name__ == '__main__':
    

    drawRect('cave',1)

    # img = loadmat('/data1/jiahua/ly/test_data/cave_test_complex/cvpr/fake_and_real_food_ms.mat')
    # input = img['input'][...]
    # img = img.transpose((2,1,0))
    # img = img.transpose((1,0,2))
    # gt = img['gt']
    
    # spectral.save_rgb("/home/jiahua/HSI-CVPR/hsi_pipeline/result/cave/Noisy.png",img,bands=(9,19,29))
    # savemat('/data1/jiahua/result/indian/s2s.mat',{'data':img})
    # splitImage("/home/jiahua/HSID/result/demo/urban_fidnet.png")