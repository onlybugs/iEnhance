import numpy as np
import pandas as pd
import time
import cooler
import multiprocessing

fn = "./data/Rao2014-GM12878-MboI-allreps-filtered.10kb.cool"
scale = 8
out_path = "./divide-path/"

bssum = [600,4000,600]
log_name = "log-divide.txt"

def remove_zeros(matrix):
    idxy = ~np.all(np.isnan(matrix), axis=0)
    M = matrix[idxy, :]
    M = M[:, idxy]
    M = np.asarray(M)
    idxy = np.asarray(idxy)
    return M, idxy

def DownSampling(rmat,ratio = 2):
    sampling_ratio = ratio
    m = np.matrix(rmat)

    all_sum = m.sum(dtype='float')
    m = m.astype(np.float64)
    idx_prob = np.divide(m, all_sum,out=np.zeros_like(m), where=all_sum != 0)
    idx_prob = np.asarray(idx_prob.reshape(
        (idx_prob.shape[0]*idx_prob.shape[1],)))
    idx_prob = np.squeeze(idx_prob)

    sample_number_counts = int(all_sum/(2*sampling_ratio))
    # 0 1 2 ... 8
    id_range = np.arange(m.shape[0]*m.shape[1])
    id_x = np.random.choice(
        id_range, size=sample_number_counts, replace=True, p=idx_prob)

    sample_m = np.zeros_like(m)
    for i in np.arange(sample_number_counts):
        x = int(id_x[i]/m.shape[0])
        y = int(id_x[i] % m.shape[0])
        sample_m[x, y] += 1.0
    sample_m = np.transpose(sample_m) + sample_m
    # print("after sample :",sample_m.sum())

    return np.array(sample_m)

def divide_hicm(hic_m,d_size,jump):
    '''
    
    '''
    lens = hic_m.shape[0] # 6
    out = [] # np.zeros(d_size ** 2).reshape(d_size,d_size)
    all_sum = 0

    for l in range(0,lens,jump):
        lifb = False
        if(l + d_size >= lens):
            l = lens - d_size
            lifb = True

        for c in range(l,lens,jump):
            cifb = False
            if(c + d_size >= lens):
                temp_m = hic_m[l:l+d_size,lens - d_size:lens]
                cifb = True
            else:
                temp_m = hic_m[l:l+d_size,c:c+d_size]
            result = np.triu(temp_m,k=1).T + np.triu(temp_m)
            all_sum += 1
            out.append(result)
            if(cifb):
                break
        if(lifb): break
    return all_sum,np.array(out)

def wlog(fname,w):
    with open(fname,'a+') as f:
        f.write(w)
        f.close()

chrs_list = ['1' ,'2' ,'3' ,'4' ,'5' ,'6' ,'7' ,'8' ,'9' ,'10' ,'11' ,'12' 
             ,'13' ,'14' ,'15' ,'16' ,'17' ,'18' ,'19' ,'20' ,'21' ,'22']


def divide(c):
    start = time.time()
    # Step1 ReadMat
    rdata = cooler.Cooler(fn)
    rmat = rdata.matrix(balance=True).fetch('chr' + c)
    rmat, _ = remove_zeros(rmat)

    # Step2 Downsampling and Norm
    logw = 'chr ' + c + " rmat sum :" + str(rmat.sum())
    lrmat = DownSampling(rmat,scale ** 2)
    logw = logw + "\n" + 'chr ' + c + " lrmat sum:" + str(lrmat.sum()) + "\n"
    wlog(log_name,logw)

    # Step3 divide
    hrn,piece_hr = divide_hicm(rmat,150,30)
    lrn,piece_lr = divide_hicm(lrmat,150,30)

    # Step5 Sampling
    block_sum = piece_hr.sum(axis=2).sum(axis = 1)
    bgood = np.percentile(block_sum,80)
    bmedian = np.percentile(block_sum,60)

    layeridx = [np.array(block_sum <= bmedian),
                np.array(block_sum >= bgood),
                np.array(block_sum > bmedian) & np.array(block_sum < bgood)]

    hrout = []
    lrout = []

    for i in range(len(layeridx)):
        tempi = layeridx[i].sum()
        if(i == 0):
            if(tempi <= bssum[i]):
                lrout.append(piece_lr[layeridx[i]])
                hrout.append(piece_hr[layeridx[i]])
            else:
                ridx = np.random.choice(np.arange(tempi),bssum[i],replace=False)
                lrout.append(piece_lr[layeridx[i]][ridx])
                hrout.append(piece_hr[layeridx[i]][ridx])
        elif(i == 1):
            if(tempi <= bssum[i]):
                lrout.append(piece_lr[layeridx[i]])
                hrout.append(piece_hr[layeridx[i]])
            else:
                ridx = np.random.choice(np.arange(tempi),bssum[i],replace=False)
                lrout.append(piece_lr[layeridx[i]][ridx])
                hrout.append(piece_hr[layeridx[i]][ridx])
        else:
            if(tempi <= bssum[i]):
                lrout.append(piece_lr[layeridx[i]])
                hrout.append(piece_hr[layeridx[i]])
            else:
                ridx = np.random.choice(np.arange(tempi),bssum[i],replace=False)
                lrout.append(piece_lr[layeridx[i]][ridx])
                hrout.append(piece_hr[layeridx[i]][ridx])

    # Step6 Save
    piece_hr = np.concatenate(hrout,axis=0)
    piece_lr = np.concatenate(lrout,axis=0)
    np.savez(out_path + "chr-" + c + ".npz",hr_sample = piece_hr,lr_sample = piece_lr)

    logw = 'chr ' + c + " hr sample sum :" + str(piece_hr.shape)
    logw = logw + "\n" + 'chr ' + c + " lr sample sum:" + str(piece_lr.shape) + "\n"
    wlog(log_name,logw)

    # Step7 time
    cost_time = time.time() - start
    wlog(log_name,"time" + str(cost_time) + "\n\n")

if __name__ == '__main__':
    pool_num = len(chrs_list) if multiprocessing.cpu_count() > len(chrs_list) else multiprocessing.cpu_count()

    start = time.time()
    print(f'Start a multiprocess pool with process_num = {pool_num}')
    pool = multiprocessing.Pool(pool_num)
    for chr in chrs_list:
        pool.apply_async(func = divide,args=(chr,))
    pool.close()
    pool.join()
    print(f'All downsampling processes done. Running cost is {(time.time()-start)/60:.1f} min.')
    print("All is done ! ^_^ ")
