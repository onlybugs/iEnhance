# 22-5-30
# c1:22-8-24
# kli
# test


import numpy as np
import pandas as pd
import time
import cooler
import multiprocessing

fn = "/store/kli/workdir/compareABandTAD/data/Rao2014-GM12878-MboI-allreps-filtered.10kb.cool"
scale = 8
bssum = [600,4000,600]
log_name = "gm-64-noscn/log-divide-no.txt"

sample_stat = {"chr":[],"sam_hrn":[],"sam_lrn":[],'hr_sum':[],'lr_sum':[],'time':[]}

def remove_zeros(matrix):
    idxy = ~np.all(np.isnan(matrix), axis=0)
    M = matrix[idxy, :]
    M = M[:, idxy]
    M = np.asarray(M)
    idxy = np.asarray(idxy)
    return M, idxy

def DownSampling(rmat,ratio = 2):
    sampling_ratio = ratio
    # testm = rmat # np.arange(9).reshape(3,3)
    m = np.matrix(rmat)

    # 采样概率
    # all ele sum = 60
    all_sum = m.sum(dtype='float')
    # print("before sample :",all_sum)
    # !!!!必须要加
    m = m.astype(np.float64)
    # 其实就是除法
    idx_prob = np.divide(m, all_sum,out=np.zeros_like(m), where=all_sum != 0)
    # reshape 1,9
    idx_prob = np.asarray(idx_prob.reshape(
        (idx_prob.shape[0]*idx_prob.shape[1],)))
    # 这是为了设计抽样概率
    idx_prob = np.squeeze(idx_prob)

    # 采样索引
    # 60 / 4
    sample_number_counts = int(all_sum/(2*sampling_ratio))
    # 0 1 2 ... 8
    id_range = np.arange(m.shape[0]*m.shape[1])
    # np.random.seed(0)

    # choice 15 from id_range
    # p: idx_prob
    # 放回抽样 
    id_x = np.random.choice(
        id_range, size=sample_number_counts, replace=True, p=idx_prob)

    # 输出矩阵
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
    # 拆
    lens = hic_m.shape[0] # 6
    # divide 4*4
    # d_size = 3
    # jump = 2
    
    # 输出的结果 先定0 最后删了
    out = [] # np.zeros(d_size ** 2).reshape(d_size,d_size)
    # 计数 除了用来统计总数还需要最后reshape的时候用
    all_sum = 0
    # print(out)
    # col
    for l in range(0,lens,jump):
        lifb = False
        if(l + d_size >= lens):
            l = lens - d_size
            lifb = True
        # row
        for c in range(l,lens,jump):
            cifb = False
            if(c + d_size >= lens):
                temp_m = hic_m[l:l+d_size,lens - d_size:lens]
                cifb = True
            # 取对称阵
            else:
                temp_m = hic_m[l:l+d_size,c:c+d_size]
            # 对称和保存结果
            result = np.triu(temp_m,k=1).T + np.triu(temp_m)
            all_sum += 1
            out.append(result)
            # out = np.concatenate((out,result))
            # print(result)
            if(cifb):
                break
        if(lifb): break
    # print(all_sum)
    # 返回最终结果
    return all_sum,np.array(out)

def wlog(fname,w):
    with open(fname,'a+') as f:
        f.write(w)
        f.close()

chrs_list = ['1' ,'2' ,'3' ,'4' ,'5' ,'6' ,'7' ,'8' ,'9' ,'10' ,'11' ,'12' 
             ,'13' ,'14' ,'15' ,'16' ,'17' ,'18' ,'19' ,'20' ,'21' ,'22']


def divide(c):
    start = time.time()
    sample_stat['chr'].append('chr' + c)
    # Step1 ReadMat
    rdata = cooler.Cooler(fn)
    rmat = rdata.matrix(balance=True).fetch('chr' + c)
    rmat, _ = remove_zeros(rmat)

    # Step2 Downsampling and Norm
    logw = 'chr ' + c + " rmat sum :" + str(rmat.sum())
    lrmat = DownSampling(rmat,scale ** 2)
    logw = logw + "\n" + 'chr ' + c + " lrmat sum:" + str(lrmat.sum()) + "\n"
    sample_stat['hr_sum'].append(rmat.sum())
    sample_stat['lr_sum'].append(lrmat.sum())
    wlog(log_name,logw)
    # SCN
    # normMh,Dh = scn_normalization(rmat)
    # normMl,Dl = scn_normalization(lrmat)

    # Step3 divide
    hrn,piece_hr = divide_hicm(rmat,150,30)
    lrn,piece_lr = divide_hicm(lrmat,150,30)

    # Step5 Sampling
    block_sum = piece_hr.sum(axis=2).sum(axis = 1)
    bgood = np.percentile(block_sum,80)
    bmedian = np.percentile(block_sum,60)
    # if(bmean < bmedian): raise Exception("bmean lower than bmedian!")

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

    # # Step6 Save
    piece_hr = np.concatenate(hrout,axis=0)
    piece_lr = np.concatenate(lrout,axis=0)
    np.savez("gm-64-noscn/chr-few-" + c + ".npz",hr_sample = piece_hr,lr_sample = piece_lr)

    sample_stat['sam_hrn'].append(piece_hr.shape)
    sample_stat['sam_lrn'].append(piece_lr.shape)
    logw = 'chr ' + c + " hr sample sum :" + str(piece_hr.shape)
    logw = logw + "\n" + 'chr ' + c + " lr sample sum:" + str(piece_lr.shape) + "\n"
    wlog(log_name,logw)

    # Step7 time
    cost_time = time.time() - start
    sample_stat['time'].append(cost_time)
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
    out_csv = pd.DataFrame(sample_stat)
    out_csv.to_csv("gm-64-noscn/scale8_sample.csv",index=False)
    print("All is done ! ^_^ ")
