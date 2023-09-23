from dataset.ap import SingleDevice_AP
import matplotlib.pyplot as plt
import numpy as np

dataset = SingleDevice_AP(data_file='/server19/lmj/github/wifi_localization/data/room3/csi_2023_09_09_22_06.txt',
                          gt_file='/server19/lmj/github/wifi_localization/data/room3-gt/csi_2023_09_09_22_06.txt',
                          window_size=2,stride=2,n_sample=100,buf=10)
fig_ap, ax_ap = plt.subplots(figsize=(12,6), facecolor='white')

x = np.array([])
amplitude = np.array([])
phase = np.array([])
gt_origin = np.array([])
gt_diff = np.array([])
gt_wavelet = np.array([])
cankao = np.array([])
for i in range(len(dataset)):
    timestamps,a,p,origin,diff,wavelet = dataset.show(i)
    x = np.append(x,timestamps)
    amplitude = np.append(amplitude,a)
    phase = np.append(phase,p)
    gt_origin = np.append(gt_origin,[origin]*len(timestamps))
    gt_diff = np.append(gt_diff,[diff]*len(timestamps))
    gt_wavelet = np.append(gt_wavelet,[wavelet]*len(timestamps))
    cankao = np.append(cankao,[1.6]*len(timestamps))
ax_ap.plot(x, cankao, color='y',linestyle='--',label='Maximum')
ax_ap.plot(x, amplitude, color='b',label='Amplitude')
# ax_ap.plot(x, phase, color='g',label='相位')
ax_ap.plot(x, gt_origin, color='r',label='GT')
ax_ap.plot(x, gt_diff, color='m',label='GT_diff')
ax_ap.plot(x, gt_wavelet, color='c',linestyle='-.',label='GT_wavelet')

plt.title('CSI Amplitude and Ground Truth')
plt.xlabel('time')
plt.ylabel('value')
fig_ap.legend(loc='upper right')
fig_ap.savefig('amplitude.jpg')

