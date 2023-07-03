import numpy as np
import multiprocessing
import torch as t
import time
import cooler

# wait accept!
from normga4 import Construct
from model import iEnhance

model = t.load("pretrained/BestscHiCModule.pt",map_location = t.device('cpu'))
fn = ["./scData/hip_"+i+".cool" for i in ["1","10","20","30","50","80","100"]]
chrs_list = ['6','12','18',"16","20"]
scpool_iden = ["hip_"+i for i in ["1","10","20","30","50","80","100"]]
model.eval()


def remove_zeros11(matrix):
    idxy = ~np.all(matrix == 0, axis=0)
    M = matrix[idxy, :]
    M = M[:, idxy]
    M = np.asarray(M)
    idxy = np.asarray(idxy)
    return M, idxy

def Combine(d_size,jump,lens,hic_m):

    Hrmat = t.zeros_like(hic_m,dtype=t.float32)
    last_col_start = -1
    last_col_end = -1
    current_col_start = 0
    current_col_end = 0
    for l in range(0,lens,jump):

        lifb = False
        current_col_start = l
        current_col_end = l + d_size
        if(l + d_size >= lens):
            l = lens - d_size
            current_col_start = l
            current_col_end = l + d_size
            lifb = True

        current_col_idx = np.arange(current_col_start,current_col_end)
        last_col_idx = np.arange(last_col_start,last_col_end)
        hr_mat_col_site = np.intersect1d(current_col_idx,last_col_idx)
        small_mat_col_site = np.arange(hr_mat_col_site.shape[0])

        last_row_start = -1
        last_row_end = -1
        current_row_start = 0
        current_row_end = 0
        for c in range(l,lens,jump):
            cifb = False
            if(c + d_size >= lens):
                current_row_start = lens - d_size
                current_row_end = lens
                temp_m = hic_m[current_col_start:current_col_end,
                            current_row_start:current_row_end]
                cifb = True
            else:
                current_row_start = c
                current_row_end = c + d_size
                temp_m = hic_m[current_col_start:current_col_end,
                            current_row_start:current_row_end]
            
            result = t.triu(temp_m,diagonal=1).T + t.triu(temp_m)

            with t.no_grad():
                y = model(result.unsqueeze(0).unsqueeze(0))
            y = y.squeeze()
            enRes = t.triu(y)
            enRes = enRes.float()
            
            current_row_idx = np.arange(current_row_start,current_row_end)
            last_row_idx = np.arange(last_row_start,last_row_end)
            hr_mat_row_site = np.intersect1d(current_row_idx,last_row_idx)
            small_mat_row_site = np.arange(hr_mat_row_site.shape[0])

            if(last_row_start < 0 and last_row_end < 0 and last_col_end < 0 and last_col_start < 0):
                Hrmat[current_col_start:current_col_end,current_row_start:current_row_end] = enRes

            elif(last_col_start < 0 and last_col_end < 0 and last_row_start >= 0 and last_row_end >= 0):
                hrsub = Hrmat[:d_size,hr_mat_row_site]
                ersub = enRes[:d_size,small_mat_row_site]
                ersub[ersub == 0] = hrsub[ersub == 0]
                enRes[:d_size,small_mat_row_site] = (hrsub + ersub)/2
                Hrmat[current_col_start:current_col_end,

                    current_row_start:current_row_end] = enRes

            elif(last_row_start < 0 and last_row_end < 0 and last_col_start >= 0 and last_col_end >= 0):
                hrsub = Hrmat[hr_mat_col_site,
                            hr_mat_col_site[0]:hr_mat_col_site[0]+d_size]
                ersub = enRes[small_mat_col_site,:d_size]
                hrsub[hrsub == 0] = ersub[hrsub == 0]
                enRes[small_mat_col_site,:d_size] = (hrsub + ersub)/2
                Hrmat[current_col_start:current_col_end,
                    current_row_start:current_row_end] = enRes

            else:
                hrsub = Hrmat[hr_mat_col_site,:]
                hrsub = hrsub[:,hr_mat_row_site]
                ersub = enRes[small_mat_col_site,:]
                ersub = ersub[:,small_mat_row_site]
                ersub[ersub == 0] = hrsub[ersub == 0]
                mean_mat = (hrsub + ersub)/2 # 120 120
                enRes[:small_mat_col_site[-1]+1,
                    :small_mat_row_site[-1]+1] = mean_mat

                Hrmat[current_col_start:current_col_end,
                    current_row_start:current_row_end] = enRes

            last_row_start = current_row_start
            last_row_end = current_row_end
            if(cifb):
                break
        
        # print("\n\n\n")
        last_col_start = current_col_start
        last_col_end = current_col_end
        if(lifb): break

    Hrmat = t.triu(Hrmat,diagonal=1).T + t.triu(Hrmat)
    return Hrmat


def predict(c):
    i = 0
    for f in fn:
        rdata = cooler.Cooler(f)
        lrmat = rdata.matrix(balance=False).fetch("chr" + c)
        lrmat,_ = remove_zeros11(lrmat)

        lrmat = lrmat.astype(np.float32)
        hic_m = t.from_numpy(lrmat)
        fakemat = Combine(150,50,lrmat.shape[0],hic_m)

        np.savez('./' + scpool_iden[i] + 'scHiC-Predict-chr'+c+'.npz',fakeh = fakemat.numpy(),lhr = lrmat)
        i += 1


if __name__ == '__main__':
    pool_num = len(chrs_list) if multiprocessing.cpu_count() > len(chrs_list) else multiprocessing.cpu_count()

    start = time.time()
    print(f'Start a multiprocess pool with process_num = {pool_num}')
    pool = multiprocessing.Pool(pool_num)
    for chr in chrs_list:
        pool.apply_async(func = predict,args=(chr,))
    pool.close()
    pool.join()
    print(f'All downsampling processes done. Running cost is {(time.time()-start)/60:.1f} min.')






