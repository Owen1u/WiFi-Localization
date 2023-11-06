'''
Descripttion: 
version: 
Contributor: Minjun Lu
Source: Original
LastEditTime: 2023-11-06 14:14:35
'''
'''
Descripttion: 
version: 
Contributor: Minjun Lu
Source: Original
LastEditTime: 2023-10-09 19:13:04
'''
import os
import glob
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
from torch.autograd import Variable

from config.cfg import Config
from dataset.wifi import WiFi,MultiWiFi
from model.model import Basemodel
from model.net import Model
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
    logger = Log(os.path.join(config['model_save_dir'],config['model_name'],'logging.log'),['epoch','learning rate','train loss','val loss','score_manned','score_numhuman','score','best_score'],dec=6)
    logger.head(config)

set_seed(config['seed'])

dataset_list = []
for gt_file in glob.glob(os.path.join('/server19/lmj/github/wifi_localization/data/1020/train/gt','*.txt')):
    data_file = gt_file.replace('gt','signal')
    dataset_list.append(WiFi(data_file=data_file,
                                gt_file=gt_file,
                                stride=config['stride'],
                                subcarrier=config['subcarrier'],
                                window_size=config['window_size']))
# for gt_file in glob.glob(os.path.join('/server19/lmj/github/wifi_localization/data/0909/gt','*.txt')):
#     data_file = gt_file.replace('gt','signal')
#     dataset_list.append(WiFi(data_file=data_file,
#                                 gt_file=gt_file,
#                                 stride=config['stride'],
#                                 window_size=config['window_size']))

train_data = ConcatDataset(dataset_list)

dataset_list = []
for gt_file in glob.glob(os.path.join('/server19/lmj/github/wifi_localization/data/1020/train/gt','*.txt')):
    data_file = gt_file.replace('gt','signal')
    dataset_list.append(WiFi(data_file=data_file,
                                gt_file=gt_file,
                                stride=2,
                                subcarrier=config['subcarrier'],
                                window_size=config['window_size']))
test_data = ConcatDataset(dataset_list)

# full_data = ConcatDataset(dataset_list)
# train_size = int(0.8 * len(full_data))
# test_size = len(full_data) - train_size
# train_data, test_data = random_split(full_data,[train_size, test_size])

train_sampler = torch.utils.data.distributed.DistributedSampler(train_data,shuffle=True)
test_sampler = torch.utils.data.distributed.DistributedSampler(test_data,shuffle=False)

train_loader = DataLoader(train_data,
                          batch_size=16,
                          num_workers=1,
                          pin_memory=True,
                        #   drop_last = True,
                          sampler=train_sampler
                          )

test_loader = DataLoader(test_data,
                         batch_size=16,
                         num_workers=1,
                         pin_memory=True,
                         sampler=test_sampler
                         )

device = torch.device('cuda',local_rank)
model = Model().to(device)
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

lossMSE = nn.MSELoss(reduce=True,size_average=True)
lossCE = torch.nn.CrossEntropyLoss()
# loss = nn.CTCLoss()

best_score=-1
def val(model,loss):
    val_loss_counter = AverageMeter('Loss', ':.4e')
    n_sample_manned=0
    n_sample_numhuman=0
    global best_score
    n_correct_manned=0
    n_correct_numhuman=0
    model.eval()
    with torch.no_grad():
        val_loss_counter.reset()
        with tqdm(total=len(test_loader),desc='val',ncols=100) as valbar:
            for batch_idx,data in enumerate(test_loader):
                timestamp,csi,gt_manned,gt_numhuman = data
                csi = csi.cuda()
                gt_manned = gt_manned.cuda()
                gt_numhuman = gt_numhuman.cuda()
                
                batch_size = csi.size()[0]
                # n_sample+=batch_size
                
                # print(preds,gt)
                preds_manned, preds_numhuman = model(csi)
                cost = lossMSE(preds_numhuman,gt_numhuman)+lossCE(preds_manned,gt_manned)
                val_loss_counter.update(cost.item())
                _,preds = preds_manned.max(1)
                for pred,label in zip(preds.cpu().numpy().tolist(),gt_manned.cpu().numpy().tolist()):
                    n_sample_manned+=len(pred)
                    for p,l in zip(pred,label):
                        if p==l:
                            n_correct_manned+=1
                    
                for pred,label in zip(preds_numhuman.cpu().numpy().tolist(),gt_numhuman.cpu().numpy().tolist()):
                    n_sample_numhuman+=len(pred)
                    for p,l in zip(pred,label):
                        if round(p) == round(l):
                            n_correct_numhuman+=1

                dist.barrier()
                valbar.update(1)
        acc_manned = n_correct_manned/float(n_sample_manned)
        acc_numhuman = n_correct_numhuman/float(n_sample_numhuman)
        acc = 0.8 * acc_manned + 0.2 * acc_numhuman
        if acc > best_score:
            best_score = acc
            torch.save(model.state_dict(), os.path.join(config['model_save_dir'],config['model_name'],'best.pth'))
        print('eval Loss:{0}'.format(val_loss_counter))
    return val_loss_counter,acc_manned,acc_numhuman,acc
        
print('start training...')
train_loss_counter = AverageMeter('Loss', ':.4e')
for epoch in range(config['epoch']):
    train_loss_counter.reset()
    model.train()
    with tqdm(total=len(train_loader),desc='epoch:{0}/{1}'.format(epoch+1, config['epoch']),ncols=100) as trainbar:
        for batch_idx,data in enumerate(train_loader):
            timestamp,csi,gt_manned,gt_numhuman = data
            csi = csi.cuda()
            gt_manned = gt_manned.cuda()
            gt_numhuman = gt_numhuman.cuda()
            batch_size = csi.size()[0]
            # print(gt)
            preds_manned, preds_numhuman = model(csi)
            cost = lossMSE(preds_numhuman,gt_numhuman)+lossCE(preds_manned,gt_manned)
            optimizer.zero_grad()
            cost.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
            optimizer.step()
            train_loss_counter.update(cost.item())
            dist.barrier()
            trainbar.update(1)
            
    print('[epoch:{0}/{1}] Loss:{2}'.format(epoch+1,config['epoch'],train_loss_counter))
    val_loss,acc_manned,acc_numhuman,score=val(model,lossMSE)
    if local_rank==0:
        logger.print([[str(epoch+1),str(optimizer.param_groups[0]['lr']),str(train_loss_counter),str(val_loss),str(acc_manned),str(acc_numhuman),str(score),str(best_score)]])
        torch.save(model.state_dict(), os.path.join(config['model_save_dir'],config['model_name'],'lastest.pth'))
            
    scheduler.step() 
    dist.barrier()
        