import numpy as np
import matplotlib.pyplot as plt
from plotters import *
from tqdm import trange, tqdm
import time
from metro import *
from binning import *
import os

def get_dgms(TI, n_iters, params: list = [0, 1, 2, 3], bin_size: float = 0.01, remove_duplicates: bool = True, thresh: int = 1):
    if remove_duplicates:
        data = run_metro_rd(TI, n_iters, params, verbose = False)
        dictdata = {}
        tdata = data.T
        for i in range(0, len(tdata)):
            dictdata[i] = tdata[i]
        volume = len(get_points(dictdata, bin_size)) * bin_size ** len(params)
        density = n_iters / volume
        dgms = ripser(data, maxdim = 0)['dgms'][0]
    else:
        dgms = bin_metro(TI, n_iters, params, bin_size = bin_size, verbose = False, thresh = thresh)
    return [dgms, density]

def rip_endpoints(TI, n_iters, sample_size, params: list = [0, 1, 2, 3], bin_size: float = 0.01, hist_bins: int = 100, show: bool = True, inner: bool = False, remove_duplicates: bool = True, thresh: int = 1):
    endpoints = []
    densitys = []
    anrs = []
    if inner:
        for i in tqdm(range(0, sample_size), position = 0, leave = False):
            dgms = get_dgms(TI, n_iters, params, bin_size = bin_size, remove_duplicates = remove_duplicates, thresh = thresh)
            endpoints.append(dgms[0][-2][1])
            anrs.append(np.mean([dgms[0][j][1] for j in range(0, len(dgms[0])-2)]))
            densitys.append(dgms[1])
    else:
        for i in tqdm(range(0, sample_size)):
            dgms = get_dgms(TI, n_iters, params, bin_size = bin_size, remove_duplicates = remove_duplicates, thresh = thresh)
            endpoints.append(dgms[0][-2][1])
            anrs.append(np.mean([dgms[0][j][1] for j in range(0, len(dgms[0])-2)]))
            densitys.append(dgms[1])
    param_names = [r"$d_1$", r"$d_2$", r"$T_{21}$", r"$T_{22}$"]
    if show:
        x = param_names[params[0]]
        y = param_names[params[1]]
        plt.title(f"TI: {TI}, Iterations: {n_iters}, Parameters: {x}, {y}\n Mean: {np.mean(endpoints):.4f}")
        plt.xlabel("Critical radius")
        plt.hist(endpoints, bins = hist_bins, density = True)
        plt.show()
    return [endpoints, densitys, anrs]

def get_means(TIs: list, n_iters, sample_size, params: list = [0, 1, 2, 3], special_TIs: list = [], bin_size: float = 0.01, remove_duplicates: bool = True, thresh: int = 1, save: bool = False):
    means = []
    stds = []
    densitym = []
    densitysd = []
    anrm = []
    anrsd = []
    for TI in tqdm(TIs, position = 1):
        data = rip_endpoints(TI, n_iters, sample_size, params, bin_size = bin_size, show = False, inner = True, remove_duplicates = remove_duplicates, thresh = thresh)
        endpoints = data[0]
        densitym.append(np.mean(data[1]))
        densitysd.append(2*np.std(data[1])/np.sqrt(sample_size))
        anrm.append(np.mean(data[2]))
        anrsd.append(2*np.std(data[2])/np.sqrt(sample_size))
        means.append(np.mean(endpoints))
        stds.append(2*np.std(endpoints)/np.sqrt(sample_size))
    param_names = [r"$d_1$", r"$d_2$", r"$T_{21}$", r"$T_{22}$"]
    parameters = ""
    for param in params:
        parameters += param_names[param] + ", "
    parameters = parameters[:-2]
    if remove_duplicates:
        method = "Remove duplicates"
    else:
        method = "Binning"
    info =  np.array([TIs, means, stds, params, np.array([densitym, densitysd]), np.array([anrm, anrsd]), np.array([n_iters, sample_size, bin_size, remove_duplicates, thresh])])
    if save:
        if remove_duplicates:
            method = "rd"
        else:
            method = f"binned({thresh}, {bin_size})"
        if len(params) == 4:
            paramnames = "all"
        else:
            paramnames = ""
            param_names = ["d1", "d2", "T21", "T22"]
            for param in params:
                paramnames += param_names[param] + ","
            paramnames = paramnames[:-1]
        np.save(f"{method};{TIs[0]}-{TIs[-1]},{TIs[1]-TIs[0]};{n_iters};{sample_size};{paramnames}", info)
    return info

def plot_means(info, special_TIs: list = [], save: bool = False):
    TIs = info[0]
    means = info[1]
    stds = info[2]
    params = info[3]
    n_iters = int(info[6][0])
    sample_size = int(info[6][1])
    bin_size = info[6][2]
    remove_duplicates = info[6][3]
    thresh = int(info[6][4])
    colors = []
    param_names = [r"$d_1$", r"$d_2$", r"$T_{21}$", r"$T_{22}$"]
    parameters = ""
    for param in params:
        parameters += param_names[param] + ", "
    parameters = parameters[:-2]
    if remove_duplicates:
        method = "Remove duplicates"
    else:
        method = "Binning"
    plt.title(f"Iterations: {n_iters}, Parameters: {parameters}, Sample size: {sample_size}\nMethod: {method}")
    plt.xlabel("TI")
    plt.ylabel("Average critical radius")
    for TI in TIs:
        if TI in special_TIs:
            colors.append("red")
        else:
            colors.append("C0")
    plt.errorbar(TIs, means, yerr = stds, ecolor = colors, zorder = 1)
    plt.scatter(TIs, means, c = colors, zorder = 2)
    if save:
        if remove_duplicates:
            method = f"rd"
        else:
            method = f"binned({thresh},{bin_size})"
        if len(params) == 4:
            paramnames = "all"
        else:
            paramnames = ""
            param_names = ["d1", "d2", "T21", "T22"]
            for param in params:
                paramnames += param_names[param] + ","
            paramnames = paramnames[:-1]
        plt.savefig(f"Mean;{method};{TIs[0]}-{TIs[-1]},{TIs[1]-TIs[0]};{n_iters};{sample_size};{paramnames}.png")
    else:
        plt.show()

def plot_densities(info, special_TIs: list = [], save: bool = False):
    densitym = info[4][0]
    densitysd = info[4][1]
    params = info[3]
    n_iters = int(info[6][0])
    sample_size = int(info[6][1])
    bin_size = info[6][2]
    remove_duplicates = info[6][3]
    thresh = int(info[6][4])
    colors = []
    param_names = [r"$d_1$", r"$d_2$", r"$T_{21}$", r"$T_{22}$"]
    parameters = ""
    for param in params:
        parameters += param_names[param] + ", "
    parameters = parameters[:-2]
    if remove_duplicates:
        method = "Remove duplicates"
    else:
        method = "Binning"
    plt.title(f"Iterations: {n_iters}, Parameters: {parameters}, Sample size: {sample_size}\nMethod: {method}")
    plt.xlabel("TI")
    if len(params) == 2:
        cube = "square"
    elif len(params) == 3:
        cube = "cube"
    elif len(params) == 4:
        cube = "hypercube"
    plt.ylabel("Density (iterations/" + cube + " volume)")
    for TI in TIs:
        if TI in special_TIs:
            colors.append("red")
        else:
            colors.append("C0")
    plt.errorbar(TIs, densitym, yerr = densitysd, ecolor = colors, zorder = 1)
    plt.scatter(TIs, densitym, c = colors, zorder = 2)
    if save:
        if remove_duplicates:
            method = f"rd"
        else:
            method = f"binned({thresh},{bin_size})"
        if len(params) == 4:
            paramnames = "all"
        else:
            paramnames = ""
            param_names = ["d1", "d2", "T21", "T22"]
            for param in params:
                paramnames += param_names[param] + ","
            paramnames = paramnames[:-1]
        plt.savefig(f"Density;{method};{TIs[0]}-{TIs[-1]},{TIs[1]-TIs[0]};{n_iters};{sample_size};{paramnames}.png")
    else:
        plt.show()

def plot_anrs(info, special_TIs: list = [], save: bool = False):
    anrm = info[5][0]
    anrsd = info[5][1]
    params = info[3]
    n_iters = int(info[6][0])
    sample_size = int(info[6][1])
    bin_size = info[6][2]
    remove_duplicates = info[6][3]
    thresh = int(info[6][4])
    colors = []
    param_names = [r"$d_1$", r"$d_2$", r"$T_{21}$", r"$T_{22}$"]
    parameters = ""
    for param in params:
        parameters += param_names[param] + ", "
    parameters = parameters[:-2]
    if remove_duplicates:
        method = "Remove duplicates"
    else:
        method = "Binning"
    plt.title(f"Iterations: {n_iters}, Parameters: {parameters}, Sample size: {sample_size}\nMethod: {method}")
    plt.xlabel("TI")
    plt.ylabel("Mean Average Noisy Radius")
    for TI in TIs:
        if TI in special_TIs:
            colors.append("red")
        else:
            colors.append("C0")
    plt.errorbar(TIs, anrm, yerr = anrsd, ecolor = colors, zorder = 1)
    plt.scatter(TIs, anrm, c = colors, zorder = 2)
    if save:
        if remove_duplicates:
            method = f"rd"
        else:
            method = f"binned({thresh},{bin_size})"
        if len(params) == 4:
            paramnames = "all"
        else:
            paramnames = ""
            param_names = ["d1", "d2", "T21", "T22"]
            for param in params:
                paramnames += param_names[param] + ","
            paramnames = paramnames[:-1]
        plt.savefig(f"ANR;{method};{TIs[0]}-{TIs[-1]},{TIs[1]-TIs[0]};{n_iters};{sample_size};{paramnames}.png")
    else:
        plt.show()

#TIs = list(range(336, 496, 8))
#data = get_means(TIs, 2000, 5000, special_TIs = [416, 832], remove_duplicates = True, save = True, bin_size=0.01)

#plot_means(data, special_TIs = [416, 832], save = True)
#plot_densities(data, special_TIs = [416, 832], save = True)
#plot_anrs(data, special_TIs = [416, 832], save = True)

#dgms = get_dgms(200, 10000)[0]
#print(dgms)
#plot_diagrams([dgms], show = True)
