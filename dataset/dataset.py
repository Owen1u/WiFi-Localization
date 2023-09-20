from typing import Any
import torch
from torch.utils.data import Dataset
import numpy as np
from collections.abc import Iterable
from scipy.interpolate import interp1d
import torchvision.transforms as transforms

class Unit():
    def __init__(self,timestamp,rssi,mcs,gain,rx) -> None:
        self.timestamp = timestamp
        self.rssi = rssi
        self.mcs = mcs
        self.gain = gain
        self.rx = rx

class Single(Dataset):
    def __init__(self,data_file:str) -> None:
        super().__init__()
        self.data:Iterable[Unit]=[]
        with open(data_file,'r') as file:
            for sample in file.readlines():
                sample_list = sample.strip().split()
                timestamp = np.array(sample_list[:3],dtype=np.float32)
                timestamp = 360 * timestamp[0] + 60 * timestamp[1] + timestamp[2]
                rssi = np.array(sample_list[3:7],dtype=np.float32)
                mcs = np.array(sample_list[7],dtype=np.float32)
                gain = np.array(sample_list[8:12],dtype=np.float32)
                rx = np.array(sample_list[12:],dtype=np.complex128).reshape((4,-1))
                self.data.append(Unit(timestamp,rssi,mcs,gain,rx))
            file.close()
        self.data = sorted(self.data,key=lambda x:x.timestamp)
        # self.data.sort(key=lambda x:x[0])
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index: Any) -> Any:
        print(self.data[index].timestamp)
        print(self.data[index].rx)
        return self.data[index]

class ShiftWindow(Single):
    def __init__(self, data_file: str, window_size:float=1,stride:float = 0.5, n_sample:int=25) -> None:
        super().__init__(data_file)
        self.window_size = window_size
        self.stride = stride
        self.n_sample = n_sample
    def __len__(self):
        return int((self.data[-1].timestamp-self.window_size)/self.stride)+1

    def __getitem__(self, index):
        starttime = self.stride*index
        endtime = starttime + self.window_size
        # print(starttime,endtime)
        rx = []
        timestamps = []
        for data in self.data:
            if data.timestamp<starttime:
                timestamps = [data.timestamp]
                rx = [data.rx]
            if starttime<=data.timestamp<endtime:
                timestamps.append(data.timestamp)
                rx.append(data.rx)
            elif data.timestamp>=endtime:
                timestamps.append(data.timestamp)
                rx.append(data.rx)
                break
        # print(timestamps)
        new_timestamps = np.linspace(starttime, endtime, num=self.n_sample, endpoint=False)
        # print(new_timestamps)
        rx = np.stack(rx,axis=0)
        # print(rx)
        fc = interp1d(timestamps, rx, kind='slinear',axis=0)
        new_rx = np.array(fc(new_timestamps),dtype=np.complex128)
        return (starttime, endtime), new_rx

class SingleDevice(Dataset):
    def __init__(self,data_file:str, gt_file:str, window_size:float=1,stride:float=0.5, n_sample:int=25) -> None:
        super().__init__()
        self.data = ShiftWindow(data_file,window_size,stride,n_sample)
        if gt_file:
            with open(gt_file) as gt:
                self.gt = np.array(gt.readline().strip().split(), dtype=np.float32)
                gt.close()
            self.gt_window=2
            assert self.gt_window%window_size==0,'window_size必须能被gt_window整除'
            ratio = int((self.gt_window-window_size)/(stride))+1
            self.gt = np.array([[gt]*(ratio+1) for gt in self.gt],dtype=np.float32).flatten()
        else:
            self.gt = np.array([0]*len(self.data), dtype=np.float32)
            
        self.normal= transforms.Compose([
                                        transforms.Normalize(std=(0.5)*n_sample,
                                                            mean=(0.5)*n_sample)])
        self.n_sample = n_sample
    def __len__(self):
        return len(self.data)
    
    def totensor(self,t:np.ndarray):
        t = t.reshape((self.n_sample,-1))
        res=[]
        for a in t:
            min_value = min(a)
            max_value = max(a)
            res.append((a-min_value)/(max_value-min_value))
        res = np.array(np.stack(res,axis=0),dtype=np.float32)
        res = res.reshape((self.n_sample,4,-1))
        return torch.from_numpy(res)
    
    def __getitem__(self, index: Any) -> Any:
        timestamp, rx = self.data[index]
        # T,L,C
        real = rx.real
        # real = self.totensor(real)
        real = torch.from_numpy(real)
        imag = rx.imag
        imag = torch.from_numpy(imag)
        # imag = self.totensor(imag)
        rx = torch.stack([real,imag],axis=-1)
        # T,L,C,2
        return timestamp, rx, self.gt[index]
    
class MultiDevice(Dataset):
    def __init__(self,data_files:list, gt_file:str, window_size:float=1,stride:float=0.5) -> None:
        super().__init__()
        self.data=[ShiftWindow(data_file,window_size,stride) for data_file in data_files]
        with open(gt_file) as gt:
            self.gt = np.array(gt.readline().strip().split(), dtype=np.float32)
            gt.close()
        self.gt_window=2
        assert self.gt_window%window_size==0,'window_size必须能被gt_window整除'
        ratio = int((self.gt_window-window_size)/(stride))+1
        self.gt = np.array([[gt]*(ratio+1) for gt in self.gt],dtype=np.float32).flatten()

    def __len__(self):
        length = [len(_data) for _data in self.data]
        return min(length)
    def get(self,index):
        timestamp = [d[index][0] for d in self.data]
        rx = [d[index][1] for d in self.data]
        return timestamp,rx
    def __getitem__(self, index: Any) -> Any:
        # print(index)
        timestamp, rx = self.get(index)
        assert len(set(timestamp))==1,'时间戳错误:{}'.format(timestamp)
        rx = np.stack(rx,axis=1)
        real = rx.real
        imag = rx.imag
        rx = np.stack([real,imag],axis=-1)
        # T,ROOM,L,C,V
        return timestamp[0], rx, self.gt[index]
        
        
if __name__ == '__main__':
    data_files=['/server19/lmj/github/wifi_localization/data/room0/csi_2023_09_09_22_06.txt',
                '/server19/lmj/github/wifi_localization/data/room1/csi_2023_09_09_22_06.txt',
                '/server19/lmj/github/wifi_localization/data/room3/csi_2023_09_09_22_06.txt',
                ]
    # gt_file = '/server19/lmj/github/wifi_localization/data/room3-gt/csi_2023_09_09_20_55.txt'
    # d = MultiDevice(data_files, gt_file, window_size=1)
    
    d = Single(data_file=data_files[2])
    
    print(d[500])
    
                
        