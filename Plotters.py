import matplotlib.pyplot as plt
from persim import plot_diagrams
from matplotlib.widgets import TextBox
import matplotlib.patches as mpatches
from sklearn import datasets
from ripser import ripser
import numpy as np
import warnings

# some functions to plot the data, persistence barcode, and persistence diagrams
# you should probably use plot, as it can do all or some of these things at once
# plot_each_diagram is only useful when going beyond H_0

def plot_data(data: list, r: float, show: bool = True, subplot: int = 111, axes: list = None, s: int = 5, xlim: list = None, ylim: list = None):
    if len(list(data)[0]) > 2:
        print("Unable to plot data with more than 2 dimensions")
        return
    if axes == None:
        figure, axes = plt.subplots()
    plt.subplot(subplot)
    for pair in data:
        plt.scatter(pair[0], pair[1], color = "black", s = s)
        axes.add_patch(plt.Circle((pair[0], pair[1]), r/2, alpha = 0.2))
    if not xlim == None:
        axes.set_xlim(xlim)
    if not ylim == None:
        axes.set_ylim(ylim)
    axes.set_aspect(1)
    if show: 
        plt.show()

def plot_barcode(dgms: list, show: bool = True, r: float = None, subplot: int = 111, axes: list = None):
    plt.subplot(subplot)
    if axes == None:
        fig, axes = plt.subplots()
    y = 1
    colors = ['red', 'blue', 'green', 'yellow', 'orange', 'purple']
    for i in range(0, len(dgms)):
        for pair in dgms[i]:
            birth = pair[0]
            death = pair[1]
            if i == 0 and float(pair[1]) == float(dgms[i][-1][1]):
                death = dgms[0][-2][1]
                plt.arrow(float(birth), y, float(death), 0, head_length = death/15, head_width = min(1.5, y/66+0.1), color='red')
            plt.plot([birth, death], [y, y], color=colors[i])
            y += 1
    hs = []
    for i in range(0, len(dgms)):
        hs.append(mpatches.Patch(color=colors[i], label="$H_{" +str(i)+"}$"))
    axes.legend(handles=hs, loc='lower right')
    if not r == None:
        axes.plot([r, r], [0, y], color='black')
    plt.gca().get_yaxis().set_visible(False)
    if show:
        plt.show()
    return y

def plot_diagram(dgms: list, show: bool = True, r: float = None, subplot: int = 111, plot_only: list = None):
    if plot_only == None:
        plot_diagrams(dgms, ax = plt.subplot(subplot))
    else:
        plot_diagrams(dgms, ax = plt.subplot(subplot), plot_only=plot_only)
    if not r == None:
        plt.plot([-100, r], [r, r], "--", color='gray')
        plt.plot([r, r], [r, 10000*dgms[-1][-1][-1]], "--", color='gray')
    if show:
        plt.show()

def plot_each_diagram(dgms: list, h: int = None):
    # plots each homology group separately on a persistent diagram
    # h is the number of homology groups to plot
    if h == None:
        h = len(dgms)
    elif h > len(dgms):
        print("Unable to plot more diagrams than data dimensions")
        return
    for i in range(0, h):
        plot_diagram(dgms, plot_only=[i], subplot = int("12" + str(i+1)), show = False)
    plt.show()

def plot(data: list, dgms: list = [], show: bool = True, r: float = None, plots: list[bool] = [True, True, True], dgm_line: bool = True, textbox: bool = True):
    """
    Plots the data, barcode, and persistence diagrams
    
    Parameters:
        data: the point clouds
        dgms: ripser's dgms output
        r: the radius of the circles when first plotting the data
        plots: list of three booleans, whether to plot the data,
            barcode, and persistence diagrams, respectively
        dgm_line: boolean, whether to plot a line on the persistence diagram
        textbox: boolean, whether to plot a textbox on the persistence diagram
                if true, the textbox will be interactive and allow you to
                change the value of the r parameter
    """
    
    # apply ripser as default unless specified
    if dgms == None:
        dgms = ripser(data)['dgms']
    if len(data[0]) > 2:
        if plots[0]:
            plots[0] = False
            print("Unable to plot data with more than 2 dimensions")

    # determine how many plots there will be
    num_plots = 0
    for plot in plots:
        if plot:
            num_plots+=1
    figure, axes = plt.subplots(1, num_plots)
    # determine the index for each plot
    if plots[0]:
        data_index = 1
        if plots[1]:
            data_ax = axes[0]
            bar_ax = axes[1]
            bar_index = 2
            if plots[2]:
                dgm_index = 3
                dgm_ax = axes[2]
        elif plots[2]:
            data_ax = axes[0]
            dgm_ax = axes[1]
            dgm_index = 2
        else:
            data_ax = axes
    elif plots[1]:
        bar_index = 1
        if plots[2]:
            bar_ax = axes[0]
            dgm_ax = axes[1]
            dgm_index = 2
        else:
            bar_ax= axes
    elif plots[2]:
        dgm_index = 1
        dgm_ax = axes
    else:
        print("No plots to plot")
        return

    # plot data and initialize circles
    if plots[0]:
        if r == None:
            plot_data(data, 0, show = False, subplot = 100+10*num_plots+data_index, axes = data_ax)
        else:
            plot_data(data, r, show = False, subplot = 100+10*num_plots+data_index, axes = data_ax)

    # make barcode (and get the number of bars)
    if plots[1]:
        y = plot_barcode(dgms, show=False, r=r, subplot=100+10*num_plots+bar_index, axes = bar_ax)
        # add legend to barcode
        hs = []
        colors = ['red', 'blue', 'green', 'yellow', 'orange', 'purple']
        for i in range(0, len(dgms)):
            hs.append(mpatches.Patch(color=colors[i], label="$H_{" +str(i)+"}$"))
        bar_ax.legend(handles=hs, loc='lower right')

    # plot diagram if applicable
    if plots[2]:
        if dgm_line:
            plot_diagram(dgms, subplot=100+10*num_plots+dgm_index, show = False, r = r)
        else:
            plot_diagram(dgms, subplot=100+10*num_plots+dgm_index, show = False)
    # upon submitting the textbox
    def submit(text):
        if text == "-":
            plt.close()
            exit()
        try:
            r = float(text)
            if r < 0:
                print("Radius must be nonnegative")
        except:
            # if empty get ride of line and circles
            if text == "":
                if plots[0]:
                    for patch in data_ax.patches:
                        patch.radius = 0
                # remove last barcode line added (the vertical one)
                if plots[1]:
                    line = bar_ax.get_lines()[-1]
                    if line._color == 'black':
                        line.remove()
                # same thing with two diagram lines
                if plots[2] and dgm_line:
                    lines = dgm_ax.get_lines()[-2:]
                    for line in lines:
                        if str(line._color) == 'gray':
                            line.remove()
                return
            # if not empty, try to convert to float
            print("Input not a valid number")
            return
        # change circle radius
        if plots[0]:
            for patch in data_ax.patches:
                patch.radius = r/2
        # replace last barcode line added, effectively moving it to the new radius
        if plots[1]:
            line = bar_ax.get_lines()[-1]
            if line._color == 'black':
                line.remove()
            bar_ax.plot([r, r], [0, y], color='black')
        # same thing with the two diagram lines
        if plots[2] and dgm_line:
            lines = dgm_ax.get_lines()[-2:]
            for line in lines:
                if str(line._color) == 'gray':
                    line.remove()
            dgm_ax.plot([-100, r], [r, r], "--", color='gray')
            dgm_ax.plot([r, r], [r, 10000*dgms[-1][-1][-1]], "--", color='gray')
        plt.draw()
    # start off the circles and vertical line if a radius is given
    if r != None:
        submit(r)
    if not textbox:
        plt.show()
        return
    # add textbox
    # all three
    if num_plots == 3:
        axbox = plt.axes([0.13, 0.15, 0.22, 0.05])
    # only diagram
    elif plots[2] and not (plots[0] or plots[1]):
        axbox = plt.axes([0.265, 0.9, 0.5, 0.05])
    # only barcode
    elif plots[1] and not (plots[0] or plots[2]):
        axbox = plt.axes([0.265, 0.9, 0.5, 0.05])
    # only data
    elif plots[0] and not (plots[1] or plots[2]):
        axbox = plt.axes([0.26, 0.9, 0.5, 0.05])
    # data and barcode
    elif plots[0] and plots[1] and not plots[2]:
        axbox = plt.axes([0.122, 0.03, 0.36, 0.05])
    # barcode and diagram
    elif plots[1] and plots[2] and not plots[0]:
        axbox = plt.axes([0.54, 0.03, 0.36, 0.05])
    # data and diagram
    else:
        axbox = plt.axes([0.122, 0.03, 0.36, 0.05])
    text_box = TextBox(axbox, "Radius ")
    text_box.on_submit(submit)
    plt.show()


# example usage
# plot() works best in full screen
data = datasets.make_circles(n_samples=100)[0] + 5 * datasets.make_circles(n_samples=100)[0]
dgms = ripser(data, maxdim = 0)['dgms']
plot(data, dgms, plots = [True, True, True])
