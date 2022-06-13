import matplotlib.pyplot as plt
from persim import plot_diagrams
from matplotlib.widgets import TextBox
import matplotlib.patches as mpatches
from ripser import ripser
import numpy as np
import warnings

def plot_data(data: list, r: float, show: bool = True, subplot: int = 111, axes: list = None):
    if len(list(data)[0]) > 2:
        print("Unable to plot data with more than 2 dimensions")
        return
    if axes == None:
        figure, axes = plt.subplots()
    plt.subplot(subplot)
    for pair in data:
        plt.scatter(pair[0], pair[1], color = "black", s = 5)
        axes.add_patch(plt.Circle((pair[0], pair[1]), r/2, alpha = 0.2))
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
    if h == None:
        h = len(dgms)
    elif h > len(dgms):
        print("Unable to plot more diagrams than data dimensions")
        return
    for i in range(0, h):
        plot_diagram(dgms, plot_only=[i], subplot = int("12" + str(i+1)), show = False)
    plt.show()

def plot_all(data: list, dgms: list = None, show: bool = True, r: float = None, dgm: bool = True, dgm_line: bool = True):
    # apply ripser as default unless specified
    if dgms == None:
        dgms = ripser(data)['dgms']
    if len(dgms) > 2:
        print("Unable to plot data with more than 2 dimensions")
        return
    # determine how many plots there will be
    if dgm:
        num_plots = 3
    else:
        num_plots = 2

    figure, axes = plt.subplots(1, num_plots)
    initial_text = "enter radius"

    # plot data and initialize circles
    if r == None:
        plot_data(data, 0, show = False, subplot = 100+10*num_plots+1, axes = axes[0])
    else:
        plot_data(data, r, show = False, subplot = 100+10*num_plots+1, axes = axes[0])

    # make barcode (and get the number of bars)
    y = plot_barcode(dgms, show=False, r=r, subplot=100+10*num_plots+2, axes = axes[1])

    # add legend
    hs = []
    colors = ['red', 'blue', 'green', 'yellow', 'orange', 'purple']
    for i in range(0, len(dgms)):
        hs.append(mpatches.Patch(color=colors[i], label="$H_{" +str(i)+"}$"))
    axes[1].legend(handles=hs, loc='lower right')
    # plot diagram if applicable
    if dgm:
        if dgm_line:
            plot_diagram(dgms, subplot=100+10*num_plots+3, show = False, r = r)
        else:
            plot_diagram(dgms, subplot=100+10*num_plots+3, show = False)
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
                for patch in axes[0].patches:
                    patch.radius = 0
                # remove last barcode line added (the vertical one)
                line = axes[1].get_lines()[-1]
                if line._color == 'black':
                    line.remove()
                # same thing with two diagram lines
                lines = axes[2].get_lines()[-2:]
                for line in lines:
                    if str(line._color) == 'gray':
                        line.remove()
                return
            # if not empty, try to convert to float
            print("Input not a valid number")
            return
        # change circle radius
        for patch in axes[0].patches:
            patch.radius = r/2
        # replace last barcode line added, effectively moving it to the new radius
        line = axes[1].get_lines()[-1]
        if line._color == 'black':
            line.remove()
        axes[1].plot([r, r], [0, y], color='black')
        plt.draw()
        # same thing with the two diagram lines
        if dgm:
            lines = axes[2].get_lines()[-2:]
            for line in lines:
                if str(line._color) == 'gray':
                    line.remove()
            if dgm_line:
                axes[2].plot([-100, r], [r, r], "--", color='gray')
                axes[2].plot([r, r], [r, 10000*dgms[-1][-1][-1]], "--", color='gray')
    # start off the circles and vertical line if a radius is given
    if r != None:
        submit(r)
    # add textbox (different location if diagram is plotted)
    if dgm:
        axbox = plt.axes([0.13, 0.15, 0.22, 0.05])
    else:
        axbox = plt.axes([0.12, 0.15, 0.36, 0.05])
    text_box = TextBox(axbox, "Radius ")
    text_box.on_submit(submit)

    plt.show()