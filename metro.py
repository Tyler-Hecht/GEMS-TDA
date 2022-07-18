import numpy as np
import matplotlib.pyplot as plt
from plotters import *
from Metropolis import run_metro
from tqdm import trange
import time

import warnings
warnings.filterwarnings("ignore")

def normalize(data):
    #normalize data to make it between 0 and 1
    max = np.max(data)
    min = np.min(data)
    return (data - min)/(max - min)

def run_metro_2(TI, n_iters, verbose: bool = True):
    data1 = run_metro(TI, int(n_iters/2), verbose = verbose, flip = False)
    data2 = run_metro(TI, int(n_iters/2), verbose = verbose, flip = True)
    data = np.concatenate((data1, data2))
    return data

def plot_metro(TI: float, n_iters: int, params: list, verbose: bool = True, alpha: float = 0.1, s: float = 1, r: float = 300):
    data = run_metro_2(TI, n_iters, verbose = verbose)
    x = data[:,params[0]]
    y = data[:,params[1]]
    param_names = [r"$d_1$", r"$d_2$", r"$T_{21}$", r"$T_{22}$"]
    x_name = param_names[params[0]]
    y_name = param_names[params[1]]
    fig, axes = plt.subplots()
    plt.scatter(x, y, alpha = 0.1, s = 1)
    if (params[0]+params[1]) % 4 == 1:
        plt.xlim(0,r)
        plt.ylim(0,r)
        axes.set_aspect('equal')
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.title("TI = " + str(TI) + ", Iterations = " + str(n_iters))
    plt.show()

def run_metro_rd(TI, n_iters, params: list = [0, 1, 2, 3], verbose: bool = True, limit: int = 30000):
    good = False
    while not good:
        data = run_metro_2(TI, n_iters, verbose = verbose)
        data = np.unique(data, axis = 0)
        if len(data) <= limit:
            good = True
    new_data = []
    for i in range(len(params)):
        new_data.append(normalize(data[:,params[i]]))
    return np.array(new_data).T

#plot_metro(400, 1000, [3, 2], s = 5)
