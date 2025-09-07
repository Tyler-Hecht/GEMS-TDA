import numpy as np
from Metropolis import run_metro
from tqdm import tqdm

def postprocess(data, include_indicators: bool = False) -> np.ndarray:
    d_a, d_b, T_2a, T_2b = data.T
    d1 = np.where(T_2a < T_2b, d_a, d_b)
    d2 = np.where(T_2a < T_2b, d_b, d_a)
    T21 = np.minimum(T_2a, T_2b)
    T22 = np.maximum(T_2a, T_2b)
    flip_ind = np.where(T_2a < T_2b, 0, 1)
    
    if include_indicators:
        return np.vstack([d1, d2, T21, T22, flip_ind]).T
    else:
        return np.vstack([d1, d2, T21, T22]).T
    
def normalize(data: np.ndarray) -> np.ndarray:
    # normalize data to make it between 0 and 1
    p_max = np.max(data)
    p_min = np.min(data)
    return (data - p_min)/(p_max - p_min)

def run_metro_2(TI: float, n_iters: int, SNR: float = 10000, verbose: bool = False, flip: bool = True, postprocess_data: bool = False, include_indicators: bool = False) -> np.ndarray:
    if flip:
        n_2 = n_iters // 2
        data1 = run_metro(TI, n_2, verbose = verbose, flip = False, SNR = SNR)
        data2 = run_metro(TI, n_iters - n_2, verbose = verbose, flip = True, SNR = SNR)
        data = np.concatenate((data1, data2))
    else:
        data = run_metro(TI, n_iters, verbose = verbose, SNR = SNR)

    if postprocess_data:
        return postprocess(data, include_indicators = include_indicators)
    else:
        return data


def generate_data(TIs: list[float], n: int, m: int, flip: bool = True, normalize_data: bool = True, postprocess_data: bool = False, verbose: bool = True) -> np.ndarray:
    # Generates data from the Metropolis algorithm
    n_TIs = len(TIs)
    data = np.zeros((n_TIs, n, m, 4))

    
    TIs_iterable = enumerate(TIs)
    if verbose: # wrap in tqdm if verbose
        TIs_iterable = tqdm(TIs_iterable, total = n_TIs)

    for i, TI in TIs_iterable:
        for j in range(n):
            data[i, j, :, :] = run_metro_2(TI, m, flip = flip, postprocess_data=postprocess_data)
            if normalize_data:
                for param in range(4):
                    data[i, j, :, param] = normalize(data[i, j, :, param])

    return data