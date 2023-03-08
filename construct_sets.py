import numpy as np

deal_fd = "data/"
fn = 'gm64'
trainlist = ['2']
testlist = ['1']

lrtraind = []
lrtestd = []
hrtraind = []
hrtestd = []

for c in trainlist:
    datashr = np.load(deal_fd + 'chr-' + c + '.npz')['hr_sample']
    dataslr = np.load(deal_fd + 'chr-' + c + '.npz')['lr_sample']

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
    datashr = np.load(deal_fd + 'chr-' + c + '.npz')['hr_sample']
    dataslr = np.load(deal_fd + 'chr-' + c + '.npz')['lr_sample']

    bnum = dataslr.shape[0]
    idx = np.random.choice(np.arange(bnum),int(bnum/2),replace=False)
    lrout = dataslr[idx,...]
    hrout = datashr[idx,...]

    lrtestd.append(lrout)
    hrtestd.append(hrout)

lrtestd = np.concatenate(lrtestd,axis=0)
hrtestd = np.concatenate(hrtestd,axis=0)
np.savez(fn + "_test.npz",hr_sample = hrtestd,lr_sample = lrtestd)