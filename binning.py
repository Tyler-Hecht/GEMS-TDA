import numpy as np
import matplotlib.pyplot as plt
from ripser import ripser
from plotters import *
from tqdm import trange
from sklearn.preprocessing import MinMaxScaler
import time
from metro import *
    
def determine_bin(values, bin_size):
    bins = []
    for value in values:
        bins.append((int(value//bin_size)+0.5)*bin_size)
    return tuple(bins)

def get_points(data: dict, bin_size):
    points = {}
    for i in range(0, len(data[0])):
        bin = determine_bin([data[param][i] for param in data], bin_size)
        if bin in points:
            points[bin] += 1
        else:
            points[bin] = 1
    return points
    
def bin_metro(TI_np: float, n_iters: int, params: list = [0, 1, 2, 3], thresh: int = 1, bin_size: float = 0.01, verbose: bool = True, only_ripser: bool = True):
    data = run_metro_2(TI_np, n_iters, verbose = verbose)
    norms = {}
    oldparams = params
    params = list(range(0, len(params)))
    for i in range(len(params)):
        norms[params[i]] = normalize(data[:,oldparams[i]])
    threshed = {}
    for param in params:
        threshed[param] = []
    points = get_points(norms, bin_size)
    print(points)
    for i in points:
        if points[i] >= thresh:
            for param in params:
                threshed[param].append(i[param])
    #combine the threshed data
    result = []
    for param in params:
        result.append(np.array(threshed[param]))
    result = np.array(result).T
    dgms = ripser(result, maxdim = 0)['dgms'][0]
    if only_ripser:
        return dgms
    else:
        volume = len(points) * bin_size**len(params)
        return points, norms, threshed, dgms, volume

def plot_bins(TI_np: float, n_iters: int, params: list = [0, 1, 2, 3], thresh: int = 1, s: float = 5, alpha: float = 0.1, bin_size: float = 0.01, verbose: bool = True, colorful: bool = True):
    if len(params) != 2:
        raise ValueError("params must be a list of length 2")
    param_names = [r"$d_1$", r"$d_2$", r"$T_{21}$", r"$T_{22}$"]
    x = param_names[params[0]]
    y = param_names[params[1]]
    info = bin_metro(TI_np, n_iters, params, thresh = thresh, bin_size = bin_size, verbose = verbose, only_ripser = False)
    points = info[0]
    xnorm = info[1][0]
    ynorm = info[1][1]
    threshed_x = info[2][0]
    threshed_y = info[2][1]
    dgms = info[3]
    colors = []
    binned_x = []
    binned_y = []
    for i in points:
        colors.append(points[i])
        binned_x.append(i[0])
        binned_y.append(i[1])
    colors = np.power((np.array(colors)/np.max(colors)), 0.5)
    fig, axes = plt.subplots(2, 2)
    plot_diagrams(dgms, show = False, ax = axes[1][1]) 
    axes[0][0].scatter(xnorm, ynorm, s = s, alpha = alpha)
    axes[0][0].set_xlim(0, 1)
    axes[0][0].set_ylim(0, 1)
    axes[0][0].set_aspect('equal')
    if colorful:
        axes[0][1].scatter(binned_x, binned_y, s = s, c = colors)
    else:
        axes[0][1].scatter(binned_x, binned_y, s = s)
    axes[0][1].set_xlim(0, 1)
    axes[0][1].set_ylim(0, 1)
    axes[0][1].set_aspect('equal')
    axes[1][0].scatter(threshed_x, threshed_y, s = s)
    axes[1][0].set_xlim(0, 1)
    axes[1][0].set_ylim(0, 1)
    axes[1][0].set_aspect('equal')
    #add titles and labels
    fig.suptitle(f"TI = {TI_np}, Iterations = {n_iters}, Threshhold = {thresh}, Bin Size = {bin_size}", fontsize = 13)
    axes[0][0].set_title("Original data (scaled)", fontsize = 10)
    axes[0][1].set_title("Binned data", fontsize = 10)
    axes[1][0].set_title("Thresholded data", fontsize = 10)
    axes[1][1].set_title("Persistence diagram", fontsize = 10)
    for i in range(0, 2):
        for j in range(0, 2):
            axes[i][j].set_xlabel(x)
            axes[i][j].set_ylabel(y)
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0, hspace=0.45)
    plt.show()

