# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 13:49:08 2024

@author: 13621
"""

# -*- coding: utf-8 -*-
"""
Created on Sat May 18 10:56:01 2024

@author: 13621
"""

import os
import numpy as np
import cv2
from time import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets #手写数据集要用到
from sklearn.manifold import TSNE








Input_path = r'E:\dataset\3rd_paper\images'   # 图片文件夹，所有图片放在一起

size = 2            ## 归一化大小，如果是255则没有进行归一化




Image_names=os.listdir(Input_path) #获取目录下所有图片名称列表
print(len(Image_names))
data=np.zeros((len(Image_names),size*size)) #初始化一个np.array数组用于存数据
label=np.zeros((len(Image_names),)) #初始化一个np.array数组用于存数据



#读取并存储图片数据，原图为rgb三通道，而且大小不一，先灰度化，再resize成200x200固定大小
for i in range(len(Image_names)):
    image_path=os.path.join(Input_path,Image_names[i])
    basename = os.path.basename(image_path)

    if 'iCT' in basename:
        label[i] = 0
        
    elif 'Revolution' in basename:
        label[i] = 1

    elif 'Optima' in basename:
        label[i] = 2    

    elif 'SOMATOM' in basename:
        label[i] = 3

        
    
    img=cv2.imread(image_path)
    img_gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    img=cv2.resize(img_gray,(size,size))
    img=img.reshape(1,size*size)
    data[i]=img
tsne_2D = TSNE(n_components=2, init='pca',perplexity= 50,random_state=0) #调用TSNE
result_2D = tsne_2D.fit_transform(data)
# tsne_3D = TSNE(n_components=3, init='pca', random_state=0)
#result_3D = tsne_3D.fit_transform(data)
# print('Finished......')



labels = label
data = result_2D
x_min, x_max = np.min(data, 0), np.max(data, 0)
embedding = (data - x_min) / (x_max - x_min)
fig = plt.figure()

colors = ['r', 'g',  'c'， 'b']
plt.figure(figsize=(10, 10))

# 遍历每个数据点,根据标签绘制不同颜色的点
for i in range(embedding.shape[0]):
    plt.scatter(embedding[i, 0], embedding[i, 1], c=colors[int(labels[i])], s=80, marker='.')



legend_colors = [plt.Line2D([0], [0], color=c, marker='.', linestyle='') for c in colors]
plt.legend(legend_colors, ['iCT', 'Optima' ,'Revolution', 'SOMATOM' ], loc='upper right')





# for i in range(data.shape[0]):
#     plt.text(data[i, 0], data[i, 1], str(label[i]),
#              color=plt.cm.Set1(label[i]),
#              fontdict={'weight': 'bold', 'size': 10})
plt.xticks([])
plt.yticks([])
plt.title('t-SNE Visualization')



# plt.show(fig1)
plt.savefig('t_sne_OD.png', dpi=2000)  # 保存图像到当前路径
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    