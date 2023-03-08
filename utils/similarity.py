from itertools import product
from time import time
import cooler
import numpy as np
import scipy.spatial.distance as dis
from scipy.stats import zscore
import pandas as pd
try:
    import multiprocessing as mp
except:
    mp=None
try:
    from datasketch import MinHash 
except:
    print("Please install datasketch if you want to use minhash method.")
from functools import partial
from multiprocessing import Pool

from tqdm import tqdm


def f0(st):
    z = zscore(st, axis=1)
    z[np.isnan(z)] = 0
    return z
def f1(i, stratum, n_bins):
    std = np.std(stratum, axis=1)
    return np.sqrt(n_bins - i) * std
def f2(i, stratum):
    mean = np.mean(stratum, axis=1)
    return stratum - mean[:, None]

def f(i, j, all_strata):
                if i<j:
                    corrs, weights = [], []
                    for stratum in all_strata:
                        s1, s2 = stratum[i, :], stratum[j, :]
                        if np.var(s1)==0 or np.var(s2)==0:
                            weights.append(0)
                            corrs.append(0)
                        else:
                            zero_pos = [k for k in range(len(s1)) if s1[k] == 0 and s2[k] == 0]
                            s1, s2 = np.delete(s1, zero_pos), np.delete(s2, zero_pos)
                            weights.append(len(s1) * np.std(s1) * np.std(s2))
                            corrs.append(np.corrcoef(s1, s2)[0, 1])
                    corrs=np.nan_to_num(corrs)
                    return np.inner(corrs, weights) / (np.sum(weights))
                    # return np.average(corrs, weights=weights)
                else:
                    return 0


def pairwise_distances(all_strata, similarity_method,
                       print_time=False, sigma=.5, window_size=10,
                       parallelize=False, n_processes=1):
    """
    
    Find the pairwise distace using different similarity method, 
    and return a distance matrix.
    

    Parameters
    ----------
    all_strata : list
        A list contained the strata of a specific chromesome for cells.
        
    similarity_method : str
        The similarity method used to calculate distance matrix.
        Now support 'innerproduct', 'hicrep', 'selfish'.
        
    print_time : bool, optional
        Whether to return the run time of similarity.
        The default is False.
        
    sigma : float, optional
        Parameter for method 'Selfish'.
        Sigma for the Gaussian kernel used in Selfish.
        The default is .5.
        
    window_size : int, optional
        Parameter for method 'Selfish'.
        Length of every window used in Selfish.
        The default is 10.
        

    Returns
    -------
    numpy.ndarray
        Distance matrix.

    """
    
    method = similarity_method.lower()
    if parallelize:
        pool = mp.Pool(n_processes)
    t0 = time()

    if method in ['inner_product', 'innerproduct']:
        # print(' Calculating z-scores...')
        zscores = []
        if parallelize:
            # pool = mp.Pool(n_processes)
            zscores = pool.starmap(f0, [(stratum,) for stratum in all_strata])
            # zscores = [pool.apply(f0, args=(stratum,)) for stratum in all_strata]
            
        else:
            for stratum in all_strata:
                z = zscore(stratum, axis=1)
                # print(np.sum(np.isnan(z)))
                z[np.isnan(z)] = 0
                zscores.append(z)
        zscores = np.concatenate(zscores, axis=1)
        t1 = time()
        # print(' Calculating inner product...')
        inner = zscores.dot(zscores.T) / zscores.shape[1]
        # print(np.max(inner), np.min(inner))
        inner[inner > 1] = 1
        inner[inner < -1] = -1
        distance_mat = np.sqrt(2 - 2 * inner)
        t2 = time()

    elif method == 'hicrep':
        # print(' Calculating means and stds...')
        n_cells, n_bins = all_strata[0].shape
        n_strata = len(all_strata)
        weighted_std = np.zeros((n_cells, n_strata))
        if parallelize:

            # pool = mp.Pool(n_processes)
            weighted_std = pool.starmap(f1, [(i,stratum,n_bins) for i, stratum in enumerate(all_strata)])
            # weighted_std = [pool.apply(f1, args=(i,stratum, n_bins)) for i, stratum in enumerate(all_strata)]
            weighted_std=np.array(weighted_std).T
            
            results2 = pool.starmap(f2, enumerate(all_strata))
            # results2 = [pool.apply(f2, args=(i,stratum)) for i, stratum in enumerate(all_strata)]
            for i in range(len(all_strata)):
                all_strata[i]=results2[i]
            
        else:
            for i, stratum in enumerate(all_strata):
                mean, std = np.mean(stratum, axis=1), np.std(stratum, axis=1)
                weighted_std[:, i] = np.sqrt(n_bins - i) * std
                all_strata[i] = all_strata[i] - mean[:, None]  # subtract a value for each row
        scores = np.concatenate(all_strata, axis=1)
        t1 = time()

        # print(' Calculating fastHiCRep score...')
        inner = scores.dot(scores.T) / (weighted_std.dot(weighted_std.T) + 1e-8)  # avoid 0 / 0
        inner[inner > 1] = 1
        inner[inner < -1] = -1
        distance_mat = np.sqrt(2 - 2 * inner)
        t2 = time()

    elif method == 'old_hicrep':
        n_cells, n_bins = all_strata[0].shape
        similarity = np.ones((n_cells, n_cells))

        if parallelize:
            # pool = mp.Pool(n_processes)
            
            results = pool.starmap(f, [(i,j,all_strata) for i,j in product(range(n_cells),range(2))])
            # results = [pool.apply(f, args=(i,j, all_strata)) for i,j in product(range(n_cells),range(2))]
            
            for i in range(n_cells):
                for j in range(i + 1, n_cells):
                    similarity[i, j] = results[i*n_cells+j]
                    similarity[j, i] = results[i*n_cells+j]
            
        else:
            for i in range(n_cells):
                for j in range(i + 1, n_cells):
                    corrs, weights = [], []
                    for stratum in all_strata:
                        s1, s2 = stratum[i, :], stratum[j, :]
                        if np.var(s1)==0 or np.var(s2)==0:
                            weights.append(0)
                            corrs.append(0)
                        else:
                            zero_pos = [k for k in range(len(s1)) if s1[k] == 0 and s2[k] == 0]
                            s1, s2 = np.delete(s1, zero_pos), np.delete(s2, zero_pos)
                            weights.append(len(s1) * np.std(s1) * np.std(s2))
                            corrs.append(np.corrcoef(s1, s2)[0, 1])
                    corrs=np.nan_to_num(corrs)
                    s = np.inner(corrs, weights) / (np.sum(weights))
                    similarity[i, j] = s
                    similarity[j, i] = s
        t1 = time()
        distance_mat = np.sqrt(2 - 2 * similarity)
        t2 = time()

    elif method == 'selfish':
        n_cells, n_bins = all_strata[0].shape
        n_strata = len(all_strata), 
        # window_size = n_bins // (n_windows + 1) * 2
        # window_size=kwargs.pop('window_size', 10)
        n_windows=n_bins//window_size
        
        # if window_size > n_strata:
        #     print('Warning: first {0} strata cannot cover the full region for calculating map similarity.'.format(n_strata),
        #           'Required: {0} strata'.format(window_size),
        #           'Use zeros to fill the missing values.')
        # print(' Calculating summation of sliding windows...')
        all_windows = np.zeros((n_cells, n_windows))
        for i, stratum in enumerate(all_strata):
            for j in range(n_windows):
                all_windows[:, j] += np.sum(stratum[:, j * window_size: (j + 1) * window_size - i],axis=1)
        t1 = time()

        # print(' Pairwisely compare the windows...')
        fingerprints = np.zeros((n_cells, n_windows * (n_windows-1)//2))
        k=0
        for i in range(n_windows):
            for j in range(n_windows-i-1):
                fingerprints[:,k]=all_windows[:,i]>all_windows[:,j]
                k+=1
        
        # for idx in range(n_cells):
        #     for i, x in enumerate(all_windows[idx]):
        #         for j, y in enumerate(all_windows[idx]):
        #             if x > y:
        #                 fingerprints[idx, i * n_windows + j] = 1
        # print(fingerprints)
        # print(np.sum(fingerprints, axis=1))
        distance = dis.pdist(fingerprints, 'euclidean')
        distance = dis.squareform(distance)
        similarity = np.exp(- sigma * distance)
        distance_mat = np.sqrt(2 - 2 * similarity)
        t2 = time()
        
    else:
        raise ValueError('Method {0} not supported. Only "inner_product", "HiCRep", "old_hicrep" and "Selfish".'.format(method))
    
    if print_time:
        print('Time 1:', t1 - t0)
        print('Time 2:', t2 - t1)
        # print(parallelize, n_processes)
        return distance_mat, t1 - t0, t2 - t1
    else:
        return distance_mat

def cal_all_strata(cell_list,chrom = "chr1",n_strata = 20):
    all_strata = [[] for i in range(n_strata)] # [ [] ,[] ]
    for cell in cell_list:
        # When resolution is 1KB, the matrix is too large to store in numpy.array. Suggest use MinHash.
        mat = cooler.Cooler(cell).matrix(balance=False).fetch(chrom)[:]
        #mat = matrix_operation(mat,['oe','randomwalk','convolution'])
        for i in range(n_strata):
            all_strata[i].append(np.diag(mat,i))
    all_strata = [np.array(strata) for strata in all_strata]
    return all_strata

def cal_minhash_chrom(cell,chrom = "chr1"):
    clr = cooler.Cooler(cell)
    pixels = clr.matrix(as_pixels=True,balance=False).fetch(chrom,chrom)
    minhash = MinHash()
    p = pixels.apply(lambda x : (str(x.bin1_id) + "," + str(x.bin2_id)).encode('utf8'),axis = 1)        
    minhash.update_batch(p)
    return minhash

def cal_minhash(cell,chroms = ["chr1"],method="intra",n_strata=None,keep_short=True,keep_all_strata=False):
    pixels = pd.DataFrame()
    clr = cooler.Cooler(cell)
    if method == 'intra':
        for i in chroms:
            px = clr.matrix(as_pixels=True,balance=False).fetch(i,i)
            pixels = pd.concat([pixels,px])
    elif method == "all":
        for i in chroms:
            px = clr.matrix(as_pixels=True,balance=False).fetch(i)
            pixels = pd.concat([pixels,px])
    else:
        print('Unrecognize minhash method on chromsomes, apply genomewide global contacts minhash.')
        pixels = clr.pixels()[:]
    if n_strata is not None:
        pixels["distance"] = pixels["bin2_id"] - pixels["bin1_id"]
        if keep_short:
            pixels = pixels[pixels["distance"]<n_strata]
            if keep_all_strata:
                gr = pixels.groupby("distance")
                p_list = []
                for i in range(n_strata):
                    minhash = MinHash()
                    try: 
                        p = gr.get_group(i).apply(lambda x : (str(x.bin1_id) + "," + str(x.bin2_id)).encode('utf8'),axis = 1)
                        minhash.update_batch(p)
                    except :
                        pass
                    p_list.append(minhash)
                return p_list
        else:
            pixels = pixels[pixels["distance"]>n_strata]
    minhash = MinHash()
    p = pixels.apply(lambda x : (str(x.bin1_id) + "," + str(x.bin2_id)).encode('utf8'),axis = 1)        
    minhash.update_batch(p)
    return minhash

def minhash_similarity(cell_list,chroms = ["chr1"],method="intra",n_strata=None,keep_short=True):
    cal_minhash1 = partial(cal_minhash,chroms=chroms,method=method,n_strata=n_strata,keep_short=keep_short)
    with Pool(processes = 20) as pool:
        minhash = list(tqdm(pool.imap(cal_minhash1,cell_list), total= len(cell_list)))
    n_cells = len(cell_list)
    sim_mat = np.zeros((n_cells,n_cells))
    col,row = np.triu_indices_from(sim_mat)
    for i,j in zip(col,row):
        sim_mat[i,j] = minhash[i].jaccard(minhash[j]) 
    sim_mat = np.triu(sim_mat,1) + np.triu(sim_mat,1).T
    return sim_mat


def cal_hicrep(cell_list,chrom = "chr1",n_strata = 20,method = 'hicrep'):
    all_strata = cal_all_strata(cell_list,chrom,n_strata)
    sim = pairwise_distances(all_strata,method)
    return sim


