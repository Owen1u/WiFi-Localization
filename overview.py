'''
Descripttion: 
version: 
Contributor: Minjun Lu
Source: Original
LastEditTime: 2023-11-05 16:26:22
'''
import os
import glob
import numpy as np
from dataset.wifi import WiFi
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

results = open('/server19/lmj/github/wifi_localization/predict/1_office.txt','r')
results = results.readlines()
for test_file,pred in zip(sorted(glob.glob('lmj/github/wifi_localization/data/test/task1/*/data/*.txt')),results):
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
    fig.savefig(os.path.join('/server19/lmj/github/wifi_localization/overviews',room+'_'+basename+'.jpg'))
    plt.close()
    
