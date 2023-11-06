'''
Descripttion: 
version: 
Contributor: Minjun Lu
Source: Original
LastEditTime: 2023-11-06 02:23:02
'''
import os
import sys
import glob
import numpy as np
from dataset.wifi import WiFi
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

TASK = 'task2'

results = open('/server19/lmj/github/wifi_localization/predict/2_office_42.txt','r')
results = results.readlines()
sum0=0
n_samples=0
for res in results:
    res = res.strip().split(' ')
    res = np.array([int(r) for r in res])
    n_samples+=res.shape[0]
    sum0+=np.sum(res==0)

print('标签0的占比：',sum0/float(n_samples))

for test_file,pred in zip(sorted(glob.glob(os.path.join('lmj/github/wifi_localization/data/test',TASK,'*/data/*.txt'))),results):
    room = test_file.split('/')[6]
    print(test_file)
    basename = os.path.basename(test_file).split('.')[0]
    data = WiFi(data_file=test_file)
    timestamps = np.linspace(0, data.duration, num=data.sampling_f * data.duration, endpoint=False)
    phase = data.phases[-1]
    fig, ax = plt.subplots(nrows=3,ncols=1,figsize=(18,18), facecolor='white')
    ax[0].plot(timestamps, phase, color='b',linestyle='-',label='phase_diff')
    
    pred = pred.strip().split(' ')
    pred = np.array([int(p) for p in pred])
    inter0 = np.linspace(0, data.duration, data.duration//2)
    fc = interp1d(inter0, pred, kind='nearest',axis=0)
    inter = np.linspace(0, data.duration, data.sampling_f * data.duration)
    pred = fc(inter)
    ax[0].plot(timestamps, pred, color='r',linestyle='-',label='prediction')
    fig.savefig(os.path.join(os.path.join('/server19/lmj/github/wifi_localization/overviews/',TASK,room+'_'+basename+'.jpg')))
    plt.close()
    
