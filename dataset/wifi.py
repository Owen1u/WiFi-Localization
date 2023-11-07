'''
Descripttion: 
version: 
Contributor: Minjun Lu
Source: Original
LastEditTime: 2023-11-08 04:09:19
'''
import os
import pywt
import math
import cmath
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections.abc import Iterable
from scipy.interpolate import interp1d
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from torch.utils.data import Dataset
from copy import copy,deepcopy
# np.set_printoptions(threshold=np.inf)
PI = 3.1415926
def sgn(num):
    if num > 0.0:
        return 1.0
    elif num == 0.0:
        return 0.0
    else:
        return -1.0

class Frame():
    def __init__(self,timestamp,rx,rssi=None,mcs=None,gain=None) -> None:
        super().__init__()
        self.timestamp = timestamp
        self.rssi = rssi
        self.mcs = mcs
        self.gain = gain
        self.decline = np.sum(gain)-np.sum(rssi)
        self.rx = rx
        if isinstance(rx,Iterable):
            self.amplitude = np.array([abs(i) for i in rx],dtype=np.float32)
            self.phase = np.array([cmath.phase(i) for i in rx],dtype=np.float32)
            # print(self.phase)
            # self.phase = np.angle(rx)
            for i in range(0,len(self.phase),64):
                self.phase[i:i+64] = np.unwrap(self.phase[i:i+64])
        else:
            self.amplitude = abs(rx)
            self.phase = cmath.phase(rx)
        
class WiFi(Dataset):
    def __init__(self,data_file,gt_file=None,sampling_f=48,duration=300,buf=8,
                 window_size=72,stride=1,phase_diff=[0,64],plotting=False,
                 _time=3,_rssi=4,_mcs=1,_gain=4,_H=64,_num_H=4,
                 subcarrier=(24,32),is_amp=True,is_phase_diff=True) -> None:
        super().__init__()
        
        self.sampling_f = sampling_f
        self.duration = duration
        self.window_size = window_size
        self.stride = stride
        self.is_amp=is_amp
        self.is_phase_diff = is_phase_diff
        if plotting:
            self.fig, self.ax = plt.subplots(nrows=4,ncols=1,figsize=(18,24), facecolor='white')
            self.ax[0].set_title(os.path.basename(data_file))
        
        # fp = open('/server19/lmj/github/wifi_localization/data/20230918/1.txt','w+')
        self.data:Iterable[Frame]=[]
        with open(data_file,'r') as file:
            for sample in file.readlines():
                sample_list = sample.strip().split()
                for sample_ele in range(0,len(sample_list),sum([_time,_rssi,_mcs,_gain,_H*_num_H])):
                    # fp.write(' '.join(sample_list[sample_ele:sample_ele+12+256])+'\n')
                    # print(sample_list[sample_ele+12:sample_ele+12+256])
                    timestamp = np.array(sample_list[sample_ele:sample_ele+sum([_time])],dtype=np.float32)
                    timestamp = 3600 * timestamp[0] + 60 * timestamp[1] + timestamp[2]
                    rssi = np.array(sample_list[sample_ele+sum([_time]):sample_ele+sum([_time,_rssi])],dtype=np.float32)
                    mcs = np.array(sample_list[sample_ele+sum([_time,_rssi]):sample_ele+sum([_time,_rssi,_mcs])],dtype=np.float32)
                    gain = np.array(sample_list[sample_ele+sum([_time,_rssi,_mcs]):sample_ele+sum([_time,_rssi,_mcs,_gain])],dtype=np.float32)
                    rx = [i.replace('i','j') for i in sample_list[sample_ele+sum([_time,_rssi,_mcs,_gain]):sample_ele+sum([_time,_rssi,_mcs,_gain,_H*_num_H])]]
                    rx = np.array(rx,dtype=np.complex128)
                    self.data.append(Frame(timestamp,rx,rssi,mcs,gain))
            file.close()
        self.data = sorted(self.data,key=lambda x:x.timestamp)
        
        self.amps = None
        amps = []
        for k in range(subcarrier[0],subcarrier[1],1):
            # amp = self.filt([i.amplitude[k] for i in self.data])
            amp = [i.amplitude[k] for i in self.data]
            timestamps,amp = self.sampling([i.timestamp for i in self.data],amp,sampling_f,duration)
            amp = self.hampel(amp,7)
            amp = amp-np.mean(amp)
            amp = amp/np.std(amp)
            amps.append(amp)
        self.amps = np.stack(amps,axis=0)

        self.phases = None
        phases = []
        for k in range(subcarrier[0],subcarrier[1],1):
            phase = np.array([i.phase[k+phase_diff[0]]-i.phase[k+phase_diff[1]] for i in self.data])
            _phase=[]
            for i in range(0,len(phase)):
                if phase[i] >np.pi:
                    while phase[i]>np.pi:
                        phase[i]-=2*np.pi
                elif phase[i] <= -np.pi:
                    while phase[i] <= -np.pi:
                        phase[i]+=2*np.pi
                if phase[i]<0:
                    phase[i]+=2*np.pi
                _phase.append(phase[i])                  
            phase = np.array(_phase)
            
            # phase = signal.savgol_filter(phase,7,3)
            timestamps,phase = self.sampling([i.timestamp for i in self.data],phase,sampling_f,duration)
            phase = self.hampel(phase,7)
            phase = phase-np.mean(phase)
            phase = phase/np.std(phase)
            phases.append(phase)
        self.phases = np.stack(phases,axis=0)
        
        self.mcs = None
        # mcs=[i.mcs[0] for i in self.data]
        # timestamps,mcs = self.sampling([i.timestamp for i in self.data],mcs,sampling_f,duration)
        # mcs = mcs-np.mean(mcs)
        # mcs = mcs/np.std(mcs)
        # self.mcs = np.stack([mcs],axis=0)
        
        self.gains = None
        # gains=[]
        # for k in range(_gain):
        #     gain = np.array([i.gain[k]-i.rssi[k] for i in self.data])
        #     timestamps,gain = self.sampling([i.timestamp for i in self.data],gain,sampling_f,duration)
        #     gain = gain-np.mean(gain)
        #     gain = gain/np.std(gain)
        #     gains.append(gain)
        # self.gains = np.stack(gains,axis=0)

        
        if plotting:
            if self.amps is not None:
                self.ax[0].plot(timestamps, self.amps[-1], color='b',linestyle='-',label='amplitude')
            if self.phases is not None:
                self.ax[1].plot(timestamps, self.phases[-1], color='b',linestyle='-',label='phase')
            if self.mcs is not None:
                self.ax[2].plot(timestamps, self.mcs[-1], color='b',linestyle='-',label='MCS')
            if self.gains is not None:
                self.ax[3].plot(timestamps, self.gain[-1], color='b',linestyle='-',label='gain')

            self.ax[0].set_title('amplitude')
            self.ax[1].set_title('phase_diff')
            self.ax[2].set_title('MCS')
            self.ax[3].set_title('gain')
        
        self.gt_bin = self.gt = None
        
        if gt_file:
            with open(gt_file) as gt:
                self.gt = np.array(gt.readline().strip().split(), dtype=np.float32)
                gt.close()
            self.gt_window=2
            self.gt = self.gt[:duration//self.gt_window]
            inter0 = np.linspace(0, duration, duration//self.gt_window)
            fc = interp1d(inter0, self.gt, kind='nearest',axis=0)
            inter = np.linspace(0, duration, sampling_f*duration)
            self.gt = fc(inter)
            # self.gt_diff = self.diff(self.gt)
            # self.gt_wavelet = self.wavelet(self.gt_diff,buf=buf*sampling_f)
            # self.gt_gauss=gaussian_filter1d(self.gt_wavelet, sigma=sampling_f*1.5)
            self.gt_bin=np.array([int(i!=0) for i in self.gt])
            
            if plotting:
                self.ax[0].plot(inter, self.gt, color='r',linestyle='-',label='gt')
                self.ax[1].plot(inter, self.gt, color='r',linestyle='-',label='gt')
                self.ax[0].plot(inter, self.gt_bin, color='r',linestyle='--',label='gt_bin')
                self.ax[1].plot(inter, self.gt_bin, color='r',linestyle='--',label='gt_bin')
                # self.ax.plot(inter, self.gt_diff, color='m',linestyle='-',label='diff')
                # self.ax.plot(inter, self.gt_wavelet, color='c',linestyle='-.',label='wavelet')
                # self.ax.plot(inter, self.gt_gauss, color='g',linestyle='-.',label='gauss')
        if plotting:
            # plt.xlim(0,300)
            # plt.ylim(-1, 1)
            plt.xlabel('Time')
            plt.ylabel('Value')
            self.fig.legend(loc='upper right')
            self.fig.savefig('dataset.jpg')
    
    def unwrap(self,l:np.ndarray):
        _l = deepcopy(l)
        new_l = np.zeros_like(_l)
        for i in range(len(_l)):
            if i ==0:
                new_l[i]=_l[i]
            elif _l[i]-_l[i-1]>PI:
                _l[i:]=_l[i:]-2*PI
                new_l[i]=_l[i]
            elif _l[i]-_l[i-1]<-PI:
                _l[i:]=_l[i:]+2*PI
                new_l[i]=_l[i]
            else:
                new_l[i]=_l[i]
        print(l[5:],new_l[5:])
        return new_l
    
                
    def hampel(self,X,k):
        length = X.shape[0] - 1
        nsigma = 3
        iLo = np.array([i - k for i in range(0, length + 1)])
        iHi = np.array([i + k for i in range(0, length + 1)])
        iLo[iLo < 0] = 0
        iHi[iHi > length] = length
        xmad = []
        xmedian = []
        for i in range(length + 1):
            w = X[iLo[i]:iHi[i] + 1]
            medj = np.median(w)
            mad = np.median(np.abs(w - medj))
            xmad.append(mad)
            xmedian.append(medj)
        xmad = np.array(xmad)
        xmedian = np.array(xmedian)
        scale = 1.25  # 缩放
        xsigma = scale * xmad
        xi = ~(np.abs(X - xmedian) <= nsigma * xsigma)  # 找出离群点（即超过nsigma个标准差）
    
        # 将离群点替换为中为数值
        xf = X.copy()
        xf[xi] = xmedian[xi]
        return xf

    
    def diff(self,sig):
        return np.diff(sig,prepend=sig[0:1])
    def wavelet(self,sig,buf,clip=None):
        aftershock = np.zeros_like(sig)
        for i,gt in enumerate(sig):
            new = np.zeros_like(sig)
            if gt==0:
                continue
            elif gt>0:
                kernel = min(int(buf),len(sig)-i-1)
                new[i+1:i+kernel+1]=gt
                new[i-kernel//3:i]=gt
            elif gt<0:
                kernel = min(int(buf),i)
                new[i+1:i+kernel//3+1]=gt
                new[i-kernel:i]=gt
            # else:
            #     kernel = min(buf//2,i)
            #     new[i-kernel:i]=gt
            #     new[i+1:i+kernel+1]=gt
            aftershock=aftershock+new
        if clip:
            return np.clip(sig+aftershock,-clip,clip)
        else:
            return sig+aftershock
        
    def medfilt(self,iter,kernel_size=5):
        padding=(kernel_size-1)//2
        iter = [iter[0]]*padding+iter+[iter[-1]]*padding
        res = []
        for i in range(len(iter)-2*padding):
            _res = iter[i:i+kernel_size]
            _res.sort(key=lambda x:abs(x))
            res.append(_res[padding])
        return res
    def filt(self,iter):
        mean = np.mean(iter)
        std = np.std(iter)
        res = []
        for i in iter:
            if i>mean+5*std or i<mean-5*std:
                res.append(mean)
            else:
                res.append(i)
        return res

    def sampling(self,timestamp,data,sampling_f,duration):
        fc = interp1d(timestamp, data, kind='slinear',axis=0)
        timestamps = np.linspace(0, duration, num=sampling_f*duration, endpoint=False)
        new_data = np.array(fc(timestamps))
        return timestamps,new_data
        
    def bandpass(self,x:np.ndarray):
        b, a = signal.butter(8,0.05,'highpass')
        x_filter = signal.filtfilt(b,a,x)
        b, a = signal.butter(8,0.6,'lowpass')
        x_filter = signal.filtfilt(b,a,x_filter)
        return x_filter
        
    def __len__(self):
        return int((self.duration-self.window_size)/self.stride)+1
    def __getitem__(self, index):
        start = int(index*self.stride)
        end = int(index*self.stride+self.window_size)
        timestamp = [start,end]
        data = []
        if self.is_amp:
            amp = self.amps[:,int(index*self.stride*self.sampling_f):int((index*self.stride+self.window_size)*self.sampling_f)]
            amp = torch.from_numpy(amp).float()
            data.append(amp)
        # decline = self.declines[:,int(index*self.stride*self.sampling_f):int((index*self.stride+self.window_size)*self.sampling_f)]
        # decline = torch.from_numpy(decline).float()
        if self.is_phase_diff:
            phase = self.phases[:,int(index*self.stride*self.sampling_f):int((index*self.stride+self.window_size)*self.sampling_f)]
            phase = torch.from_numpy(phase).float()
            data.append(phase)
        # mcs = self.mcs[:,int(index*self.stride*self.sampling_f):int((index*self.stride+self.window_size)*self.sampling_f)]
        # mcs = torch.from_numpy(mcs).float()
        # gain = self.gains[:,int(index*self.stride*self.sampling_f):int((index*self.stride+self.window_size)*self.sampling_f)]
        # gain = torch.from_numpy(gain).float()
        data = torch.concat(data,dim=0)
        if self.gt is not None and self.gt_bin is not None:
            label2 = self.gt[int(index*self.stride*self.sampling_f):int((index*self.stride+self.window_size)*self.sampling_f):24]
            label1 = self.gt_bin[int(index*self.stride*self.sampling_f):int((index*self.stride+self.window_size)*self.sampling_f):24]
            return timestamp,data,torch.from_numpy(label1),torch.from_numpy(label2)
        else:
            return timestamp,data,-1,-1
    
class MultiWiFi(Dataset):
    def __init__(self,dir_list,filename,gt_file) -> None:
        super().__init__()
        self.datalist = [WiFi(os.path.join(dir,filename),gt_file) for dir in dir_list]
    def __len__(self):
        return len(self.datalist[0])
    def __getitem__(self, index):
        setlist=[]
        for subset in self.datalist:
            timestamp,data,label = subset[index]
            setlist.append(data)
        return timestamp,torch.concat(setlist,dim=0),label
            
        

if __name__=='__main__':
    dataset = WiFi(data_file = '/server19/lmj/github/wifi_localization/data/test/task2-handle/signal/left_3.txt',
                   gt_file='/server19/lmj/github/wifi_localization/data/test/task2-handle/gt/left_3.txt',
                   duration=300,phase_diff=[0,64],subcarrier=(24,32),
                   _rssi=4,_gain=4,_H=64,_num_H=4,
                   plotting=True)
    print(dataset[0])