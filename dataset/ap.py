'''
Descripttion: 
version: 
Contributor: Minjun Lu
Source: Original
LastEditTime: 2023-09-24 03:37:47
'''
import cmath
from typing import Any
import torch
from torch.utils.data import Dataset
import numpy as np
from collections.abc import Iterable
from scipy.interpolate import interp1d
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

class Unit_AP(Dataset):
    def __init__(self,timestamp,rssi,mcs,gain,rx) -> None:
        super().__init__()
        self.timestamp = timestamp
        self.rssi = rssi
        self.mcs = mcs
        self.gain = gain
        self.rx = rx
        self.abs = np.array([abs(i) for i in rx],dtype=np.float32)
        self.phase = np.array([cmath.phase(i) for i in rx],dtype=np.float32)

class Single_AP(Dataset):
    def __init__(self, data_file:str,gt_file=None, **kw) -> None:
        super().__init__()
        self.data:Iterable[Unit_AP]=[]
        if gt_file:
            with open(gt_file) as gt:
                self.gt = gt.readline().strip().split()
                gt.close()
        with open(data_file,'r') as file:
            for sample in file.readlines():
                sample_list = sample.strip().split()
                timestamp = np.array(sample_list[:3],dtype=np.float32)
                timestamp = 360 * timestamp[0] + 60 * timestamp[1] + timestamp[2]
                rssi = np.array(sample_list[3:7],dtype=np.float32)
                mcs = np.array(sample_list[7],dtype=np.float32)
                gain = np.array(sample_list[8:12],dtype=np.float32)
                rx = np.array(sample_list[12:],dtype=np.complex128)
                self.data.append(Unit_AP(timestamp,rssi,mcs,gain,rx))
            file.close()
        self.data = sorted(self.data,key=lambda x:x.timestamp)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index: Any) -> Unit_AP:
        return self.data[index]
    
class ShiftWindow(Single_AP):
    def __init__(self, data_file: str, window_size:float=2,stride:float = 2, n_sample:int=25, **kw) -> None:
        super().__init__(data_file)
        self.window_size = window_size
        self.stride = stride
        self.n_sample = n_sample
    def __len__(self):
        return int((self.data[-1].timestamp-self.window_size)/self.stride)+1
    def __getitem__(self, index):
        starttime = self.stride*index
        endtime = starttime + self.window_size
        a_p = []
        timestamps = []
        for data in self.data:
            if data.abs[7]>1.6:
                continue
            if data.timestamp<starttime:
                timestamps = [data.timestamp]
                a_p = [np.array([data.abs,data.phase])]
            if starttime<=data.timestamp<endtime:
                timestamps.append(data.timestamp)
                a_p.append(np.array([data.abs,data.phase]))
            elif data.timestamp>=endtime:
                timestamps.append(data.timestamp)
                a_p.append(np.array([data.abs,data.phase]))
                break
        a_p = np.stack(a_p,axis=0)
        
        new_timestamps = np.linspace(starttime, endtime, num=self.n_sample, endpoint=False)
        fc = interp1d(timestamps, a_p, kind='slinear',axis=0)
        new_ap = np.array(fc(new_timestamps))
        return (starttime, endtime), new_timestamps, new_ap

class SingleDevice_AP(Dataset):
    MEAN = torch.tensor([1.4712990522384644, 0.1000966802239418])
    STD = torch.tensor([0.4386167526245117, 1.7101144790649414])
    def __init__(self,data_file:str, gt_file:str, window_size:float=2, stride:float=2, n_sample:int=25, buf:int=10) -> None:
        super().__init__()
        self.data_file = data_file
        self.gt_file = gt_file
        self.window_size = window_size
        self.stride = stride
        self.data = ShiftWindow(data_file,window_size,stride,n_sample)
        if gt_file:
            with open(gt_file) as gt:
                self.gt = np.array(gt.readline().strip().split(), dtype=np.float32)
                gt.close()
            self.gt_window=2
            # assert self.gt_window%window_size==0,'window_size必须能被gt_window整除'
            ratio = int((self.gt_window-window_size)/(stride))+1
            inter0 = np.linspace(0, 100, len(self.gt), endpoint=True)
            inter = np.linspace(0, 100, len(self.data), endpoint=True)
            fc = interp1d(inter0, self.gt, kind='nearest',axis=0)
            self.gt = self.gt_origin = fc(inter)
            self.gt = self.gt_diff = np.diff(self.gt,prepend=self.gt[0:1])
            aftershock = np.zeros_like(self.gt)
            for i,gt in enumerate(self.gt):
                new = np.zeros_like(self.gt)
                if gt==0:
                    continue
                elif gt>0:
                    kernel = min(buf//window_size,len(self.gt)-i-1)
                    new[i+1:i+kernel]=gt
                elif gt<0:
                    kernel = min(buf//window_size,i)
                    new[i-kernel:i]=gt
                aftershock=aftershock+new
            self.gt = self.gt_wavelet =self.gt + aftershock
            print('GT:',self.gt)
            # _gt=[]
            # for gt in self.gt:
            #     if gt == 0:
            #         _gt.append(0)
            #     else:
            #         _gt.append(1)
            self.gt = np.array(self.gt,dtype=np.float32).flatten()
        else:
            self.gt = np.array([0]*len(self.data), dtype=np.float32)
        self.n_sample = n_sample
        self.normal= transforms.Compose([
                                         transforms.Normalize(mean=self.MEAN,
                                                              std=self.STD)])
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index: Any) -> torch.Tensor:
        timestamp,timestamps, a_p = self.data[index]
        # T,L,C
        # a_p = a_p[:,:,0:64]
        a_p = torch.from_numpy(a_p).float()
        a_p = self.normal(a_p.permute(1,2,0)).permute(2,0,1)
        return timestamp, a_p, torch.from_numpy(np.array(self.gt[index]+2)).long()
    
    def show(self,index):
        timestamp,timestamps, a_p = self.data[index]
        return timestamps, a_p[:,0,7], a_p[:,1,7], self.gt_origin[index],self.gt_diff[index], self.gt_wavelet[index]

if __name__ == '__main__':
    import os
    from torch.utils.data import ConcatDataset
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
                                            window_size=2,stride=2,n_sample=100))
    dataset = ConcatDataset(dataset_list)
    # print(dataset[100][1].size())
    # fig_e, ax_e = plt.subplots(figsize=(24,8), facecolor='white')
    # ap = []
    # x = []
    # phase = []
    # cankao = []
    # for d in dataset:
    #     x.append(d.timestamp)
    #     ap.append(d.abs[7])
    #     phase.append(d.phase[7])
    #     cankao.append(1.6)
    # ax_e.plot(x, cankao, color='r')
    # ax_e.plot(x, ap, color='b')
    # ax_e.plot(x, phase, color='g')
    # fig_e.savefig('/server19/lmj/github/wifi_localization/a_p.jpg')

    imgs = torch.zeros([100,2,256,1])
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
        