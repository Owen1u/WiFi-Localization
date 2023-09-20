'''
Descripttion: 
version: 
Contributor: Minjun Lu
Source: https://github.com/seominseok0429/pytorch-warmup-cosine-lr/blob/master/warmup_scheduler/scheduler.py
LastEditTime: 2022-10-26 21:04:08
'''
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import matplotlib.pyplot as plt

class GradualWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]


    def step(self, epoch=None, metrics=None):
        if self.finished and self.after_scheduler:
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.total_epoch)
        else:
            return super(GradualWarmupScheduler, self).step(epoch)

if __name__ == '__main__':
    v = torch.zeros(10)
    optim = torch.optim.Adam([v], lr=0.0001)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, 200, eta_min=0, last_epoch=-1)
    scheduler = GradualWarmupScheduler(optim, multiplier=10, total_epoch=10, after_scheduler=cosine_scheduler)
    a = []
    b = []
    c = []
    for epoch in range(1, 200):
        a.append(epoch)
        b.append(optim.param_groups[0]['lr'])
        scheduler.step(epoch)
        

    plt.plot(a,b)
    plt.savefig('/nvme0n1/lmj/disorder_selfsup/tools/gradualwarmup.jpg',bbox_inches='tight', pad_inches=0)
    # plt.show()