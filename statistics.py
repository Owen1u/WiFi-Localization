'''
统计数据集 幅度和相位 两个维度 的均值和标准差
由于房间0、1均为无人数据，因此仅以房间3作统计
注意:调整参数时,取 WINDOW_SIZE==STRIDE
'''
import os
import torch
from torch.utils.data import ConcatDataset
from dataset.ap import SingleDevice_AP
WINDOW_SIZE = 2
STRIDE = 2
N_SAMPLE_SECOND= 50
DIM = 64*4
dataset_list = []
data_files=['csi_2023_09_09_20_55.txt',
            'csi_2023_09_09_21_12.txt',
            'csi_2023_09_09_21_20.txt',
            'csi_2023_09_09_21_43.txt',
            'csi_2023_09_09_21_51.txt',
            'csi_2023_09_09_22_06.txt',
            ]
gt_dir = '/server19/lmj/github/wifi_localization/data/room3-gt'
data_dir = [
            # ['/server19/lmj/github/wifi_localization/data/room0',None],
            # ['/server19/lmj/github/wifi_localization/data/room1',None],
            ['/server19/lmj/github/wifi_localization/data/room3',gt_dir]]
for file in data_files:
    for dirname in data_dir:
        dataset_list.append(SingleDevice_AP(data_file=os.path.join(dirname[0],file),
                                        gt_file = os.path.join(dirname[1],file) if dirname[1] else dirname[1],
                                        window_size=WINDOW_SIZE,stride=STRIDE,n_sample_persecond=N_SAMPLE_SECOND))
dataset = ConcatDataset(dataset_list)

imgs = torch.zeros([N_SAMPLE_SECOND*WINDOW_SIZE,2,DIM,1])
means = []
stdevs = []
for i in range(len(dataset)):
    timestamp,img,label = dataset[i]
    imgs = torch.concat((imgs,img.unsqueeze(-1)),dim=3)
for i in range(2):
    pixels = imgs[:,i,:,:].ravel()
    means.append(torch.mean(pixels).item())
    stdevs.append(torch.std(pixels).item())
print(means)
print(stdevs)