'''
Descripttion: 
version: 
Contributor: Minjun Lu
Source: Original
LastEditTime: 2023-09-24 03:36:37
'''
import os
import random
import argparse
import builtins
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, ConcatDataset, random_split

from config.cfg import Config
# from dataset.dataset import MultiDevice, SingleDevice
from dataset.ap import SingleDevice_AP
from model import Basemodel
from utils.logger import Log
from utils.meter import AverageMeter
from utils.gradualwarmup import GradualWarmupScheduler

cudnn.benchmark = True
cudnn.deterministic = True

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

local_rank = int(os.getenv('LOCAL_RANK',-1))
parser = argparse.ArgumentParser()
parser.add_argument("--cfg", type=str, help='path to config', default='/server19/lmj/github/wifi_localization/config/config.yaml')
args = parser.parse_args()
print(local_rank)
dist.init_process_group(backend='nccl', init_method='env://')
torch.cuda.set_device(local_rank)

cfg = Config(args.cfg)
config = cfg()

if local_rank!=0:
    def print_pass(*args):
        pass
    builtins.print = print_pass
else:
    if not os.path.isdir(config['model_save_dir']):
        os.mkdir(config['model_save_dir'])
    if not os.path.isdir(os.path.join(config['model_save_dir'],config['model_name'])):
        os.mkdir(os.path.join(config['model_save_dir'],config['model_name']))
    logger = Log(os.path.join(config['model_save_dir'],config['model_name'],'logging.log'),['epoch','learning rate','train loss','val loss','score','best_score'],dec=6)
    logger.head(config)

set_seed(config['seed'])

dataset_list = []
data_files=['csi_2023_09_09_20_55.txt',
            'csi_2023_09_09_21_12.txt',
            'csi_2023_09_09_21_20.txt',
            'csi_2023_09_09_21_43.txt',
            'csi_2023_09_09_21_51.txt',
            'csi_2023_09_09_22_06.txt',
            ]
data_dir = ['/server19/lmj/github/wifi_localization/data/room0',
            '/server19/lmj/github/wifi_localization/data/room1',
            '/server19/lmj/github/wifi_localization/data/room3']
gt_dir = '/server19/lmj/github/wifi_localization/data/room3-gt'
# for file in data_files:
#     dataset_list.append(MultiDevice(data_files=[os.path.join(dirname,file) for dirname in data_dir],
#                                     gt_file = os.path.join(gt_dir,file),
#                                     window_size=2,stride=2))
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
                                         window_size=2,stride=2,n_sample=100,buf=10))


full_dataset = ConcatDataset(dataset_list)

train_size = int(0.9 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_data, test_data = random_split(full_dataset,[train_size, test_size])

train_sampler = torch.utils.data.distributed.DistributedSampler(train_data,shuffle=True)
test_sampler = torch.utils.data.distributed.DistributedSampler(test_data,shuffle=False)

train_loader = DataLoader(train_data,
                          batch_size=8,
                          num_workers=4,
                          pin_memory=True,
                          drop_last = True,
                          sampler=train_sampler
                          )

test_loader = DataLoader(test_data,
                         batch_size=8,
                         num_workers=4,
                         pin_memory=True,
                         sampler=test_sampler
                         )

# loss = nn.MSELoss(reduce=True,size_average=True)
loss = torch.nn.CrossEntropyLoss()

device = torch.device('cuda',local_rank)
model = Basemodel(T=100).to(device)
model=torch.nn.parallel.DistributedDataParallel(model,device_ids=[local_rank],find_unused_parameters =True,output_device=local_rank)

filtered_parameters = []
params_num = []
for p in filter(lambda p: p.requires_grad, model.parameters()):
    filtered_parameters.append(p)
    params_num.append(np.prod(p.size()))
if local_rank==0:
    print('Trainable params num : ', sum(params_num))
    logger.log.info('Trainable params num : '+ str(sum(params_num)))
    
optimizer = optim.Adam(filtered_parameters, lr=config['lr_f'],betas=(0.9, 0.999))
cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config['epoch'], eta_min=config['lr_min'], last_epoch=-1)
scheduler = GradualWarmupScheduler(optimizer, multiplier=config['multiplier'], total_epoch=config['warmup_epoch'], after_scheduler=cosine_scheduler)

best_score=-1
def val(model,loss):
    val_loss_counter = AverageMeter('Loss', ':.4e')
    n_sample=0
    global best_score
    n_correct=0
    model.eval()
    with torch.no_grad():
        val_loss_counter.reset()
        with tqdm(total=len(test_loader),desc='val',ncols=100) as valbar:
            for batch_idx,data in enumerate(test_loader):
                timestamp,csi,gt = data
                csi = csi.cuda()
                gt = gt.cuda()
                
                batchsize = gt.size()[0]
                n_sample+=batchsize
                
                preds = model(csi)
                # print(preds,gt)
                cost = loss(preds,gt.long())
                val_loss_counter.update(cost.item())
                _,preds = preds.max(1)
                # print(preds)
                for pred,label in zip(preds,gt.long()):
                    if pred == label:
                        n_correct+=1

                dist.barrier()
                valbar.update(1)
        acc = n_correct/float(n_sample)
        if acc > best_score:
            best_score = acc
            torch.save(model.state_dict(), os.path.join(config['model_save_dir'],config['model_name'],'best.pth'))
        print('eval Loss:{0}'.format(val_loss_counter))
    return val_loss_counter,acc
        
print('start training...')
train_loss_counter = AverageMeter('Loss', ':.4e')
for epoch in range(config['epoch']):
    train_loss_counter.reset()
    model.train()
    with tqdm(total=len(train_loader),desc='epoch:{0}/{1}'.format(epoch+1, config['epoch']),ncols=100) as trainbar:
        for batch_idx,data in enumerate(train_loader):
            timestamp,csi,gt = data
            csi = csi.cuda()
            gt = gt.cuda()
            # print(gt)
            preds = model(csi)
            cost = loss(preds,gt.long())
            optimizer.zero_grad()
            cost.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
            optimizer.step()
            train_loss_counter.update(cost.item())
            dist.barrier()
            trainbar.update(1)
            
    print('[epoch:{0}/{1}] Loss:{2}'.format(epoch+1,config['epoch'],train_loss_counter))
    val_loss,score=val(model,loss)
    if local_rank==0:
        logger.print([[str(epoch+1),str(optimizer.param_groups[0]['lr']),str(train_loss_counter),str(val_loss),str(score),str(best_score)]])
        torch.save(model.state_dict(), os.path.join(config['model_save_dir'],config['model_name'],'lastest.pth'))
            
    scheduler.step() 
    dist.barrier()
        
            
            
            