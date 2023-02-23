# 22-9-16
# kli
# Combine all train chr and test chr

import torch.nn as nn
import torch as t
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np

deal_fd = "gm-64-noscn/"
fn = 'gm64_no_scn'
trainlist = ['2' ,'3' ,'5' ,'6' ,'7' ,'8' ,'9' ,'10' ,'11' ,'12' 
             ,'13' ,'14' ,'15' ,'17' ,'18' ,'19' ,'20' ,'21' ,"X" ]
testlist = ['1','4','16','22']

lrtraind = []
lrtestd = []
hrtraind = []
hrtestd = []

for c in trainlist:
    datashr = np.load(deal_fd + 'chr-few-' + c + '.npz')['hr_sample']
    dataslr = np.load(deal_fd + 'chr-few-' + c + '.npz')['lr_sample']

    bnum = dataslr.shape[0]
    idx = np.random.choice(np.arange(bnum),int(bnum/2),replace=False)
    lrout = dataslr[idx,...]
    hrout = datashr[idx,...]

    lrtraind.append(lrout)
    hrtraind.append(hrout)

lrtraind = np.concatenate(lrtraind,axis=0)
hrtraind = np.concatenate(hrtraind,axis=0)

np.savez(fn + "_train.npz",hr_sample = hrtraind,lr_sample = lrtraind)

for c in testlist:
    datashr = np.load(deal_fd + 'chr-few-' + c + '.npz')['hr_sample']
    dataslr = np.load(deal_fd + 'chr-few-' + c + '.npz')['lr_sample']

    bnum = dataslr.shape[0]
    idx = np.random.choice(np.arange(bnum),int(bnum/2),replace=False)
    lrout = dataslr[idx,...]
    hrout = datashr[idx,...]

    lrtestd.append(lrout)
    hrtestd.append(hrout)

lrtestd = np.concatenate(lrtestd,axis=0)
hrtestd = np.concatenate(hrtestd,axis=0)
np.savez(fn + "_test.npz",hr_sample = hrtestd,lr_sample = lrtestd)