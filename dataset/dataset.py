from typing import Any
import torch
from torch.utils.data import Dataset
import numpy as np
from collections.abc import Iterable
from scipy.interpolate import interp1d
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


MEAN = [1.5241853706992233e-05, 1.5241853706992233e-05, 1.5241853706992233e-05, 1.5241853706992233e-05, 1.5241853706992233e-05, 1.5241853706992233e-05, 1.5241853706992233e-05, 1.5241853706992233e-05, 0.02633012955711858, -0.03126869082404951, -0.01080645699816693, -0.04894640959914193, -0.04462377790262817, -0.037435416873202815, -0.06402226468964713, 0.006454407526589369, -0.04192976373548586, 0.05196851711788414, 0.006161802469542482, 0.07057442934386919, 0.06336285917594366, 0.052457688354534245, 0.08722274710094306, -0.008096161936362041, 0.06075464678802843, -0.08204751611060067, -0.01538760390108787, -0.10084195927272822, -0.08269470928100862, -0.06619542365673065, -0.10873723922232473, 0.01474967290761209, -0.06932171228228853, 0.09224164852375824, 0.01941360349459982, 0.12063920629797288, 0.10382989900204431, 0.06541745416762117, 0.1329608120208795, -0.024992309733228982, 0.08381705051356313, -0.12183658378460502, -0.0302652156894199, -0.153932248122521, -0.13727936831018936, -0.08299742345675283, -0.15075367920484845, 0.046906301952646424, -0.06968379557062436, 0.13525978307839298, 0.048846927398865356, 0.1493820426881731, 0.14451097292208473, 0.07703071277974827, 0.17644925488538094, -0.06340338676966806, 0.08599160687608205, -0.1542875114456489, -0.062210072817908336, -0.1828219740508201, -0.18814919391095655, -0.07734887670548857, -0.1866057789844269, 0.0843454514240272, -0.0812448538915274, 0.18112126235497897, 0.09049175219552401, 0.1948342761690767, 0.18455578329985778, 0.07724835366410025, 0.20189546457841243, -0.07356425591643881, 0.07144856089519354, -0.20359528862821388, -0.09273137936675663, -0.2053004531527787, -0.21537174930264724, -0.04638734382118482, -0.1846919597778214, 0.1238847673175556, -0.060786642798043265, 0.21563384915348474, 0.11116441344629893, 0.2090689346730336, 0.23017044057229574, 0.05199264016653831, 0.18986384909028634, -0.1275896775644368, 0.04955391841980129, -0.21985247242678865, -0.11979650673963457, -0.19730977955991938, -0.2397638822350141, -0.054557153991476134, -0.20081644268932217, 0.14910721554484854, -0.025027887624174842, 0.24478478428802347, 0.14927810237725292, 0.1866897834117868, 0.23599041127281903, 0.030627388912467586, 0.18667018553265144, -0.14430728635019155, 0.013831309468594875, -0.24026249613097017, -0.16394879749293306, -0.1763567622910869, -0.22576559907804036, -0.0012240349578485766, -0.17329348250626733, 0.1541625060445165, -0.0076507783308991896, 0.24072844425864787, 0.1778764965158466, 0.16400134488347495, 0.21798094091433204, -0.019675153042487226, 0.11391143993885315, -0.15207438785395136, -0.015604484280319629, -0.17769129460547, -0.0008401446849875862, -0.0005099147974123895, 0.00034021981018998547, -0.0008283337009869576, -1.0049175382365186e-05, -0.0003347556653864045]
STD = [5.080646096893921e-07, 5.080646096893921e-07, 5.080646096893921e-07, 5.080646096893921e-07, 5.080646096893921e-07, 5.080646096893921e-07, 5.080646096893921e-07, 5.080646096893921e-07, 0.6914086816678499, 0.6646412156567577, 0.7599693043575239, 0.7553766777147984, 0.7843620474806535, 0.8322464806534613, 0.8497607898866177, 0.821520851619222, 0.8764848233894221, 0.8208760523076156, 0.840465605611297, 0.8583565693219776, 0.8010989192938531, 0.885325696698193, 0.8442936676210243, 0.8297598506730395, 0.8802634872049374, 0.7898307816182581, 0.8061628304764086, 0.8707811994697391, 0.7948172006449051, 0.8958573032529076, 0.8910514245313318, 0.8146407004871489, 0.8846535450320061, 0.8228621255445716, 0.8427705994905103, 0.8681523723380073, 0.7787948719894697, 0.9286157596331099, 0.8892090189963716, 0.8148483391659322, 0.9087893151443961, 0.7747141776929939, 0.7996931977067346, 0.8776036199188032, 0.7834850997471942, 0.8857000826728842, 0.8788255165294196, 0.7918909554157612, 0.889469624836667, 0.7802402677763819, 0.7803557367639097, 0.8925915571686852, 0.7747074432070442, 0.8990059551119158, 0.8809187803342421, 0.7804208489273468, 0.8876038425764943, 0.7651151067808981, 0.7685728130427159, 0.8751133347998817, 0.7323870048350656, 0.8950910701509547, 0.8745747116920441, 0.7514542249870793, 0.7442782431608796, 0.5703890045656065, 0.752963031801194, 0.8656310992209635, 0.7304480508131029, 0.8919965921265702, 0.8729161858049437, 0.7537409304304161, 0.8894492801135443, 0.7393044517554076, 0.7398580634326684, 0.8917641780000438, 0.7453369072266348, 0.8896592396647933, 0.8963058418595111, 0.7388912306462765, 0.8881801172682621, 0.7457429695926692, 0.7368758043633203, 0.8884687450468317, 0.7292759897890203, 0.8912971952034558, 0.9121539510860609, 0.7176333586818554, 0.8778843525494882, 0.7678346147920883, 0.7370887945255682, 0.9075172378411068, 0.7657608706130328, 0.8886144562286464, 0.9176446794049719, 0.7327538856219833, 0.8837228685834237, 0.7646671389755847, 0.7125646850240578, 0.9201468514615603, 0.7781625539860899, 0.8535427596760893, 0.8960326029498774, 0.7170050120611728, 0.8361233931463865, 0.7759783531808222, 0.7212171761324171, 0.8889106929442787, 0.7935658967336054, 0.835161338130895, 0.8979597588683677, 0.7335813049671236, 0.8255896259811755, 0.8122716735711505, 0.7343848253940989, 0.8767781198615148, 0.7884092941007261, 0.7731888386906676, 0.7841378577411535, 0.6783881553863577, 0.6440258450582681, 0.6632865395054004, 0.005793139565210342, 0.005853083338190228, 0.005251837021252897, 0.005209970668743814, 0.0034135332606742915, 0.0033351558426944626]


class Unit():
    def __init__(self,timestamp,rssi,mcs,gain,rx,gt) -> None:
        self.timestamp = timestamp
        self.rssi = rssi
        self.mcs = mcs
        self.gain = gain
        self.rx = rx
        self.gt=gt

class Single(Dataset):
    def __init__(self,data_file:str,gt_file=None, **kw) -> None:
        super().__init__()
        self.data:Iterable[Unit]=[]
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
                rx = np.array(sample_list[12:],dtype=np.complex128).reshape((4,-1))
                gt=self.gt[int(timestamp//2)] if self.gt and timestamp//2<len(self.gt) else 0
                self.data.append(Unit(timestamp,rssi,mcs,gain,rx,gt))
            file.close()
        self.data = sorted(self.data,key=lambda x:x.timestamp)
        if gt_file:
            with open(gt_file) as gt:
                self.gt = np.array(gt.readline().strip().split(), dtype=np.float32)
                gt.close()
                
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index: Any) -> Any:
        print(self.data[index].timestamp)
        print(self.data[index].rx)
        return self.data[index]

class ShiftWindow(Single):
    def __init__(self, data_file: str,gt_file, window_size:float=2,stride:float = 2, n_sample:int=25) -> None:
        super().__init__(data_file,gt_file)
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
    def __init__(self,data_file:str, gt_file:str, window_size:float=2,stride:float=2, n_sample:int=25) -> None:
        super().__init__()
        self.data = ShiftWindow(data_file,gt_file,window_size,stride,n_sample)
        if gt_file:
            with open(gt_file) as gt:
                self.gt = np.array(gt.readline().strip().split(), dtype=np.float32)
                gt.close()
            self.gt_window=2
            # assert self.gt_window%window_size==0,'window_size必须能被gt_window整除'
            ratio = int((self.gt_window-window_size)/(stride))+1
            inter0 = np.linspace(0, 100, len(self.gt), endpoint=True)
            inter = np.linspace(0, 100, len(self.gt)*ratio, endpoint=True)
            fc = interp1d(inter0, self.gt, kind='nearest',axis=0)
            self.gt = fc(inter)
            # _gt=[]
            # for gt in self.gt:
            #     if gt == 0:
            #         _gt.append(0)
            #     else:
            #         _gt.append(1)
            self.gt = np.array(self.gt,dtype=np.float32).flatten()
        else:
            self.gt = np.array([0]*len(self.data), dtype=np.float32)
            
        self.normal= transforms.Compose([
                                         transforms.Normalize(mean=MEAN,
                                                              std=STD)])
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
        imag = rx.imag
        fig, ax = plt.subplots(figsize=(10,10), facecolor='white')
        scatter = ax.scatter(real[0][0], imag[0][0], c='r', alpha=0.8)
        scatter = ax.scatter(real[0][1], imag[0][1], c='g', alpha=0.8)
        scatter = ax.scatter(real[0][2], imag[0][2], c='b', alpha=0.8)
        scatter = ax.scatter(real[0][3], imag[0][3], c='y', alpha=0.8)
        fig.savefig('/server19/lmj/github/wifi_localization/look.jpg')
        # real = self.totensor(real)
        real = torch.from_numpy(real)
        
        imag = torch.from_numpy(imag)
        # imag = self.totensor(imag)
        rx = torch.stack([real,imag],axis=-1)
        # T,L,C,2
        rx = rx.reshape(self.n_sample,4,-1).permute(2,0,1)
        # rx = self.normal(rx)
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
    data_files=['/server19/lmj/github/wifi_localization/data/room0/csi_2023_09_09_21_43.txt',
                '/server19/lmj/github/wifi_localization/data/room1/csi_2023_09_09_21_43.txt',
                '/server19/lmj/github/wifi_localization/data/room3/csi_2023_09_09_21_43.txt',
                ]
    gt_file = '/server19/lmj/github/wifi_localization/data/room3-gt/csi_2023_09_09_21_43.txt'
    # d = MultiDevice(data_files, gt_file, window_size=1)
    
    data = SingleDevice(data_file=data_files[2],gt_file=gt_file,window_size=2,stride=2)
        # SingleDevice(data_file=data_files[1],gt_file=gt_file,window_size=1,stride=1)+\
        # SingleDevice(data_file=data_files[2],gt_file=gt_file,window_size=1,stride=1)
    print(data[180])
    fig_e, ax_e = plt.subplots(figsize=(24,8), facecolor='white')
    x = []
    y = []
    gt = []
    for d in data:
        x.append(d.timestamp)
        rx = d.rx
        y.append(np.linalg.norm(rx,))
        gt.append(d.gt)
    ax_e.plot(x, gt, color='r')
    ax_e.plot(x, y, color='b')
    fig_e.savefig('/server19/lmj/github/wifi_localization/energy4.jpg')
    
    
    # imgs = torch.zeros([128,25,4,1])
    # means = []
    # stdevs = []
    # for i in range(len(d)):
    #     timestamp,img,label = d[i]
    #     imgs = torch.concat((imgs,img.unsqueeze(-1)),dim=3)
    # for i in range(128):
    #     pixels = imgs[i,:,:,:].ravel()
    #     means.append(torch.mean(pixels).item())
    #     stdevs.append(torch.std(pixels).item())
    # print(means)
    # print(stdevs)
                
        