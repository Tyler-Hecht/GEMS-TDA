import numpy as np
import matplotlib.pyplot as plt
from ripser import ripser
from Helpers import run_metro_2, normalize, bin_points
from tqdm import trange
from scipy import optimize
import os

class Data:
    def __init__(self, TIs: list, n_iters: int, sample_size: int, SNR: float, nullpts: list = [416, 832]):
        # Data class includes methods for generating, saving, loading, and everything else
        # necessary to calculate the average critical radius
        # These methods can be run sequentially to using generate_all
        
        # Initializes the data object for a given SNR, TI range, number of iterations, sample size, and null points
        self.TIs = TIs
        self.TItitle = f"{min(TIs)}-{max(TIs)},{TIs[1]-TIs[0]}"
        self.n_iters = n_iters
        self.sample_size = sample_size
        self.SNR = SNR
        self.nullpts = nullpts
        # these start out as empty
        self.data = []
        self.binned = []
        self.threshed = []
        self.ripped = []
        self.acr_mean = []
        self.acr_std = []
        self.fit = None
        self.minSNR = None
        self.dTI = {}

    def generate_data(self):
        # Generates data from the Metropolis algorithm
        # Doesn't normalize
        for i in trange(0, len(self.TIs), position = 1):
            self.data.append([])
            for j in trange(0, self.sample_size, position = 0, leave = False):
                self.data[i].append(run_metro_2(TIs[i], self.n_iters, verbose = False, SNR = self.SNR))

    def save_data(self, filename: str = None):
        if filename == None:
            filename = "data;" + self.TItitle + ";" + str(self.n_iters) + ";" + str(self.sample_size) + ";" + str(self.SNR) + ".npy"
        np.save(filename, self.data)

    def load_data(self, filename: str = None):
        if filename == None:
            filename = "data;" + self.TItitle + ";" + str(self.n_iters) + ";" + str(self.sample_size) + ";" + str(self.SNR) + ".npy"
        self.data = np.load(filename, allow_pickle = True)

    def bin_data(self, bin_size: float = 0.01):
        # normalizes and puts the data into bins of size bin_size
        self.bin_size = bin_size
        self.binned = []
        for i in trange(0, len(self.data), position = 1):
            self.binned.append([])
            for j in trange(0, len(self.data[i]), position = 0, leave = False):
                sample = self.data[i][j]
                # normalizes each parameter
                for k in range(0, 4):
                    sample[:,k] = normalize(sample[:,k])
                pts = bin_points(sample, bin_size)
                self.binned[i].append(pts)

    def save_binned(self, filename: str = None):
        if filename == None:
            filename = "binned(" + str(self.bin_size) + ");" + self.TItitle + ";" + str(self.n_iters) + ";" + str(self.sample_size) + ";" + str(self.SNR) + ".npy"
        np.save(filename, self.binned)
    
    def load_binned(self, filename: str = None, bin_size: float = 0.01):
        self.bin_size = bin_size
        if filename == None:
            filename = "binned(" + str(self.bin_size) + ");" + self.TItitle + ";" + str(self.n_iters) + ";" + str(self.sample_size) + ";" + str(self.SNR) + ".npy"
        self.binned = np.load(filename, allow_pickle = True)

    def thresh_bins(self, thresh: float = 1):
        self.thresh = thresh
        self.threshed = []
        for i in trange(len(self.binned), position = 1):
            self.threshed.append([])
            for j in trange(len(self.binned[i]), position = 0, leave = False):
                # thresholds the binned data
                threshed = [x for x in self.binned[i][j] if self.binned[i][j][x] >= thresh]
                self.threshed[i].append(threshed)

    def save_threshed(self, filename: str = None):
        if filename == None:
            filename = "threshed(" + str(self.bin_size) + "," + str(self.thresh) + ");" + self.TItitle + ";" + str(self.n_iters) + ";" + str(self.sample_size) + ";" + str(self.SNR) + ".npy"
        np.save(filename, self.threshed)
    
    def load_threshed(self, filename: str = None, bin_size: float = 0.01, thresh: float = 1):
        self.bin_size = bin_size
        self.thresh = thresh
        if filename == None:
            filename = "threshed(" + str(self.bin_size) + "," + str(self.thresh) + ");" + self.TItitle + ";" + str(self.n_iters) + ";" + str(self.sample_size) + ";" + str(self.SNR) + ".npy"
        self.threshed = np.load(filename, allow_pickle = True)

    def rip_threshed(self):
        self.ripped = []
        for i in trange(len(self.threshed), position = 1):
            self.ripped.append([])
            for j in trange(len(self.threshed[i]), position = 0, leave = False):
                # applies ripser to the threshed data and gets the critical radius
                self.ripped[i].append(ripser(np.array(self.threshed[i][j]), maxdim = 0)["dgms"][0][-2][1])

    def save_ripped(self, filename: str = None):
        if filename == None:
            filename = "ripped(" + str(self.bin_size) + "," + str(self.thresh) + ");" + self.TItitle + ";" + str(self.n_iters) + ";" + str(self.sample_size) + ";" + str(self.SNR) + ".npy"
        np.save(filename, self.ripped)
    
    def load_ripped(self, filename: str = None, bin_size: float = 0.01, thresh: float = 1):
        self.bin_size = bin_size
        self.thresh = thresh
        if filename == None:
            filename = "ripped(" + str(self.bin_size) + "," + str(self.thresh) + ");" + self.TItitle + ";" + str(self.n_iters) + ";" + str(self.sample_size) + ";" + str(self.SNR) + ".npy"
        self.ripped = np.load(filename, allow_pickle = True)

    def acr_ripped(self):
        self.acr_mean = []
        self.acr_std = []
        for i in trange(len(self.ripped)):
            # calculates the mean and standard deviation of the critical radii
            self.acr_mean.append(np.mean(self.ripped[i]))
            self.acr_std.append(2*np.std(self.ripped[i])/np.sqrt(self.sample_size))

    def save_acr(self, filename: str = None):
        if filename == None:
            filename = "acr(" + str(self.bin_size) + "," + str(self.thresh) + ");" + self.TItitle + ";" + str(self.n_iters) + ";" + str(self.sample_size) + ";" + str(self.SNR) + ".npy"
        np.save(filename, [self.acr_mean, self.acr_std])

    def load_acr(self, filename: str = None, bin_size: float = 0.01, thresh: float = 1):
        self.bin_size = bin_size
        self.thresh = thresh
        if filename == None:
            filename = "acr(" + str(self.bin_size) + "," + str(self.thresh) + ");" + self.TItitle + ";" + str(self.n_iters) + ";" + str(self.sample_size) + ";" + str(self.SNR) + ".npy"
        acr = np.load(filename, allow_pickle = True)
        self.acr_mean = acr[0]
        self.acr_std = acr[1]

    def plot_acr(self, polyfit: int = None, save: bool = False, filename: str = None):
        '''
        Plots the average critical radius across all TIs
        See fit_poly for details about fitting

        Parameters:
            polyfit: If specified, will fit a polynomial of degree polyfit
                     to the data and plot the fit
            save: If True, will save the plot to a file
                  Otherwise, will display the plot
            filename: If save is True, will save the plot to this file
        '''
        
        title = f"Iterations: {self.n_iters}, Sample size: {self.sample_size}, Bin size: {self.bin_size}, Threshold: {self.thresh}"
        plt.xlabel("TI")
        plt.ylabel("Average critical radius")
        plt.errorbar(self.TIs, self.acr_mean, yerr = self.acr_std, fmt = 'o', zorder = 1)
        null = None
        for i in self.nullpts:
            if i in self.TIs:
                null = i
                plt.errorbar(i, self.acr_mean[self.TIs.index(i)], yerr = self.acr_std[self.TIs.index(i)], color = 'limegreen', marker = 'o', zorder = 2)
        xlim = plt.xlim()
        ylim = plt.ylim()
        # plots the polynomial fit
        if polyfit != None:
            f = self.fit_poly(polyfit, null)
            fit = self.fit
            min = self.minSNR
            x = np.linspace(330, 500, 1000)
            y = f(x)
            plt.plot(x, y, 'r')
            plt.xlim(xlim)
            plt.ylim(ylim)
            plt.plot(min, f(min), 'ro', zorder = 3)
            title += f"\nSNR: {self.SNR}, Minimum: {min:.2f}, $\Delta$TI = {416 - min:.2f}"
        plt.title(title)
        if save:
            if filename == None:
                filename = ""
                if polyfit != None:
                    filename += "fit(" + str(polyfit) + ");"
                filename += "acr(" + str(self.bin_size) + "," + str(self.thresh) + ");" + self.TItitle + ";" + str(self.n_iters) + ";" + str(self.sample_size) + ";" + str(self.SNR) + ".png"
            plt.savefig(filename)
            plt.close()
        else:
            plt.show()

    def fit_poly(self, degrees: int, null: float = None):
        '''
        Fits a polynomial to the data and sets the minimum point to self.minSNR

        Parameters:
            degrees: How many degrees of the polynomial to fit
                     (2 for quadratic, 4 for quartic, etc.)
            null: The null point in the TI range
                  If specified self.dTI will be set to the difference
                  between the null point and the minimum SNR
        
        Returns:
            The polynomial function
        '''
        fit = np.polyfit(self.TIs, self.acr_mean, degrees)
        def f(x):
            sum = 0
            for i in range(degrees + 1):
                sum += fit[i]*x**(degrees - i)
            return sum
        self.fit = fit
        # finds the SNR where the fit is minimized
        self.minSNR = optimize.fminbound(f, self.TIs[0], self.TIs[-1])
        if null != None:
            self.dTI[null] = null - self.minSNR
        return f

    def generate_all(self, bin_size: float = 0.01, thresh: int = 1, save: bool = True):
        '''
        Runs everything sequentially
        If save is true, data are saved after each step instead of at the end
        in case there's a memory issue

        Parameters:
            bin_size: The bin size used for binning
            thresh: The threshold used after binning
            save: Whether or not to save the data to a file
        '''
        self.bin_size = bin_size
        self.thresh = thresh
        self.generate_data()
        if save:
            self.save_data()
        self.bin_data(bin_size)
        if save:
            self.save_binned()
        self.thresh_bins(thresh)
        if save:
            self.save_threshed()
        self.rip_threshed()
        if save:
            self.save_ripped()
        self.acr_ripped()
        if save:
            self.save_acr()

# example usage
TIs = list(range(366, 466, 2))
n_iters = 1000
sample_size = 200
os.chdir("Data/1000(366-466)")
SNRs = list(range(1000, 40250, 250))
dTIs = []
for SNR in SNRs:
    data = Data(TIs, n_iters, sample_size, SNR)
    try:
        data.load_acr()
    except:
        print(SNR)
        data.generate_all()
        data.load_acr()
    data.fit_poly(2, 416)
    data.plot_acr(2, True)
    dTIs.append(data.dTI[416])
log = False
if log:
    plt.scatter(np.log(SNRs), np.log(dTIs), 10)
    plt.xlabel("log(SNR)")
    plt.ylabel("log($\Delta$TI)")
else:
    plt.scatter(SNRs, dTIs, 10)
    plt.xlabel("SNR")
    plt.ylabel("$\Delta$TI")
plt.title("Iterations = 1000, Sample Size = 200, Bin Size = 0.01, Threshold = 1")
plt.show()
