# 22-9-1
# kli
# Combine mulit

import numpy as np
import multiprocessing
import torch as t
import time
import cooler

from normga4 import Construct

model = t.load("predictm/Norm-module-ga13best-50.pt",map_location = t.device('cpu'))
fn = "/store/kli/workdir/compareABandTAD/data/Rao2014-GM12878-MboI-allreps-filtered.10kb.cool"
chrs_list = ['2' ,'4' ,'6' ,'8' ,'10' ,'12','14' ,'16' ,'18','20','22']
scale = 4
model.eval()

# !!!
def remove_zeros(matrix):
    idxy = ~np.all(np.isnan(matrix), axis=0)
    M = matrix[idxy, :]
    M = M[:, idxy]
    M = np.asarray(M)
    idxy = np.asarray(idxy)
    return M, idxy

def GetOE(mat):
    '''
    Get OE
    '''
    chr_len = mat.shape[0]
    cut_off = chr_len/10
    mask = np.zeros(chr_len)
    num_mat = mat.copy()
    num_mat[num_mat > 0] = 1
    num_vector = np.sum(num_mat,axis=0)
    for i in range(chr_len):
        if(num_vector[i] >= cut_off):
            mask[i] = 1
    mask = mask == 1

    ox = np.arange(chr_len)
    oy = np.arange(chr_len)
    omask = mask.copy()
    decay = {}
    for i in range(chr_len):
        o_diag = mat[(ox,oy)]
        o_diag_mask = o_diag[omask]
        # gap
        if(o_diag_mask.shape[0] == 0):
            decay[i] = 0
        else:
            decay[i] = o_diag_mask.mean()
        ox = np.delete(ox,-1)
        oy = np.delete(oy,0)
        omask = np.delete(omask,-1)

    ex = np.arange(chr_len)
    ey = np.arange(chr_len)
    except_mat = np.ones_like(mat,dtype = np.float32)
    for i in range(chr_len):
        if(decay[i] == 0):
            ex = np.delete(ex,-1)
            ey = np.delete(ey,0)
            continue
        except_mat[(ex,ey)] = decay[i]
        except_mat[(ey,ex)] = decay[i]
        ex = np.delete(ex,-1)
        ey = np.delete(ey,0)
        
    oe = mat/except_mat

    return oe

def Combine(d_size,jump,lens,hic_m):

    Hrmat = t.zeros_like(hic_m,dtype=t.float32)
    # 处理
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
        # 获取重叠位点的生成矩阵坐标以及合并后矩阵(上一次的坐标)
        hr_mat_col_site = np.intersect1d(current_col_idx,last_col_idx)
        small_mat_col_site = np.arange(hr_mat_col_site.shape[0])

        last_row_start = -1
        last_row_end = -1
        current_row_start = 0
        current_row_end = 0
        for c in range(l,lens,jump):
            cifb = False
            # time.sleep(3)
            # 这里是if-else是为了取对称阵以及当前的位置
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
            
            # 对称
            result = t.triu(temp_m,diagonal=1).T + t.triu(temp_m)

            # .... 这里是处理矩阵的结果
            with t.no_grad():
                y = model(result.unsqueeze(0).unsqueeze(0))
            y = y.squeeze()
            # print(y.shape)
            enRes = t.triu(y)
            enRes = enRes.float()
            
            # 中间位置
            current_row_idx = np.arange(current_row_start,current_row_end)
            last_row_idx = np.arange(last_row_start,last_row_end)
            # 获取重叠位点的生成矩阵坐标以及合并后矩阵(上一次的坐标)
            hr_mat_row_site = np.intersect1d(current_row_idx,last_row_idx)
            small_mat_row_site = np.arange(hr_mat_row_site.shape[0])

            # 真正意义的第一个位置的处理
            if(last_row_start < 0 and last_row_end < 0 and last_col_end < 0 and last_col_start < 0):
                Hrmat[current_col_start:current_col_end,current_row_start:current_row_end] = enRes

            # 第一次列位置处理
            elif(last_col_start < 0 and last_col_end < 0 and last_row_start >= 0 and last_row_end >= 0):
                # 取上一次和本次的子矩阵
                hrsub = Hrmat[:d_size,hr_mat_row_site]
                ersub = enRes[:d_size,small_mat_row_site]
                # 复制HR矩阵已经有的结果 然后和子矩阵0位置复制 方便平均的时候保存数值
                ersub[ersub == 0] = hrsub[ersub == 0]
                enRes[:d_size,small_mat_row_site] = (hrsub + ersub)/2
                # 复原矩阵
                Hrmat[current_col_start:current_col_end,

                    current_row_start:current_row_end] = enRes

            # 每次换列的第一个位置的处理 包括了最后一个位置
            # 思路同上 但是这次是横着的
            elif(last_row_start < 0 and last_row_end < 0 and last_col_start >= 0 and last_col_end >= 0):
                hrsub = Hrmat[hr_mat_col_site,
                            hr_mat_col_site[0]:hr_mat_col_site[0]+d_size]
                ersub = enRes[small_mat_col_site,:d_size]
                # print(hrsub,ersub)
                # 复制HR矩阵已经有的结果 然后和子矩阵0位置复制 方便平均的时候保存数值
                hrsub[hrsub == 0] = ersub[hrsub == 0]
                enRes[small_mat_col_site,:d_size] = (hrsub + ersub)/2
                # 复原矩阵
                Hrmat[current_col_start:current_col_end,
                    current_row_start:current_row_end] = enRes
                # print("change line\n",Hrmat)

            # 其它位置
            else:
                hrsub = Hrmat[hr_mat_col_site,:]
                hrsub = hrsub[:,hr_mat_row_site]
                ersub = enRes[small_mat_col_site,:]
                ersub = ersub[:,small_mat_row_site]
                # print("hr sub\n",hrsub,'\ner sub\n',ersub)
                # 含0问题
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

def ReadFile(fn,chr):
    '''
    Read cooler
    '''
    rdata = cooler.Cooler(fn)
    rmat = rdata.matrix(balance=True).fetch(chr)
    # rmat[np.isnan(rmat)] = 0
    
    return rmat

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

def scn_normalization(X, max_iter=1000, eps=1e-6, copy=True):
    m, n = X.shape
    if m != n:
        raise ValueError
    if copy:
        X = X.copy()
    X = np.asarray(X)
    X = X.astype(float)
    D = np.ones((X.shape[0],))
    for it in np.arange(max_iter):
        # sqrt(sqrt(sum(X.^2, 1))).^(-1)
        square = np.multiply(X, X)
        # sss_row and sss_col should be equal because of sysmmetry
        sss_row = np.sqrt(np.sqrt(square.sum(axis=-1)))
        sss_row[sss_row == 0] = 1
        sss_row = sss_row**(-1)

        # sss_col = np.sqrt(np.sqrt(square.sum(axis=-2)))
        # sss_col[sss_col == 0] = 1
        # sss_col = sss_col**(-1)

        sss_col = sss_row
        # D*X*D
        # next_X = np.diag(sss_row)@X@np.diag(sss_col)
        next_X = (sss_row*(X*sss_col).T).T
        D = sss_row * D

        if np.abs(X - next_X).sum() < eps:
            print("break at iteration %d" % (it,))
            break
        X = next_X
    return X, D

def predict(c):

    HicRmat = ReadFile(fn,"chr" + c)
    HicRmat, _ = remove_zeros(HicRmat)
    lrmat = DownSampling(HicRmat,scale ** 2)

    # HicRmat = GetOE(HicRmat)
    # lrmat = GetOE(lrmat)
    # normMh,Dh = scn_normalization(HicRmat)
    # normMl,Dl = scn_normalization(lrmat)
    # HicRmat = normMl.astype(np.float32)
    lrmat = lrmat.astype(np.float32)
    print("chr",c,HicRmat.shape)
    hic_m = t.from_numpy(lrmat)
    fakemat = Combine(150,50,lrmat.shape[0],hic_m)

    np.savez('predicto/ga13/predict' + str(scale) + '-ga13-50-chr'+c+'.npz',fakeh = fakemat.numpy(),thr = HicRmat,lhr = lrmat)

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






