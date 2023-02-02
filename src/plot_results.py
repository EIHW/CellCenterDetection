import os, sys, csv
import matplotlib.pyplot as plt
import numpy as np
from random import shuffle
import matplotlib.ticker as mticker
import pandas as pd
# enable command line argument parsing
from argparse import ArgumentParser

def custom_multiple_subplots(path, nfigs_height, nfigs_width, Xs, Ys, labels, title=[], font_size=12, x_axis="", y_axis="", y_lims=[], symbols = ["o-", "x-", "v-"], fig_size=(10,4), colors=[], line_styles=[]):
    fig, fig_subplots = plt.subplots(nfigs_height, nfigs_width, figsize=fig_size)
    
    for plot_nr in range(len(Xs)):
        Xs_line = Xs[plot_nr]
        Ys_line = Ys[plot_nr]
        for line_nr in range(len(Xs_line)):
            fig_subplots[plot_nr].plot(Xs_line[line_nr], Ys_line[line_nr], symbols[line_nr], label=labels[line_nr])
            fig_subplots[plot_nr].set(xlabel=x_axis, ylabel=y_axis)
            #fig_subplots.set_xscale("log")
            fig_subplots[plot_nr].set_xscale("symlog")
            if len(y_lims) > 0:
                fig_subplots[plot_nr].set_ylim((0.725, 0.925))
            #fig_subplots.xaxis.set_minor_formatter(mticker.ScalarFormatter())
            fig_subplots[plot_nr].xaxis.set_major_formatter(mticker.ScalarFormatter())
            fig_subplots[plot_nr].xaxis.get_major_formatter().set_scientific(False)
            fig_subplots[plot_nr].xaxis.get_major_formatter().set_useOffset(False)

            fig_subplots[plot_nr].set_xticks(Xs_line[line_nr])
            fig_subplots[plot_nr].set_title(title[plot_nr])
            fig_subplots[plot_nr].legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path)





def make_nice_line_plot(path ,Xs, Ys, labels, title="", font_size=12, x_axis="", y_axis="", fig_size=[], colors=[], line_styles=[]):
    #plt.figure(figsize=fig_size)
    plt.rcParams.update({'font.size': font_size})
    plt.title(title)
    plt.xlabel(x_axis, fontsize=font_size)
    plt.ylabel(y_axis, fontsize=font_size)
    for i in range(len(Xs)):
        X = Xs[i]
        Y = Ys[i]
        label = labels[i]
        if len(colors) == 0:
            plt.plot(X, Y, label=label)
        else:
            plt.plot(X, Y, color=colors[i], linestyle=line_styles[i], label=label)


    plt.legend()
    plt.savefig(path)
    plt.clf()


if __name__ == "__main__":
    code_path = os.path.dirname(os.path.realpath(__file__))

    # command line arg parsing
    
    
    parser = ArgumentParser()
    #my_root_dir = os.path.join(os.path.dirname(code_path),"Aufnahmen_bearbeitet/20190420_Easter_special/Images_Part_2_DcAMP")
    parser.add_argument('--result_dir', default="")
    args = parser.parse_args()
    
    data_dir = args.result_dir
    # val_fcnn_file = data_dir + "fcnn/val_fcnn.csv"
    # test_fcnn_file = data_dir + "fcnn/test_fcnn.csv"
    # val_frcnn_file = data_dir + "frcnn/val_cell_cores_frcnn.csv"
    # test_frcnn_file = data_dir + "frcnn/test_cell_cores_frcnn.csv"
    Xs = []
    Ys = []
    titles = []
    for partition in ["val", "test"]:
        fcnn_data = pd.read_csv(data_dir + "fcnn/" + partition + "_fcnn.csv")
        frcnn_data = pd.read_csv(data_dir + "frcnn/" + partition + "_cell_cores_frcnn.csv")
        distances_fcnn = list(fcnn_data["Distance Threshold"].values)
        distances_frcnn = list(frcnn_data["Distance Threshold"].values)
        fcnn_f1s = list(fcnn_data["F1"].values)
        frcnn_f1s = list(frcnn_data["F1"].values)

        labels = ["General VAD", "Child VAD", ""]
        if partition == 'val':
            titles.append('Development')
        elif partition == 'test':
            titles.append('Test')
        else:
            titles.append(partition)
        x_axis = "Distance Threshold"
        y_axis = "F1"

        #line_style = ["r", "b", "--"]
        colors = ["r", "b"]
        line_styles = "-", "-"
        Xs.append([distances_fcnn, distances_frcnn])
        Ys.append([fcnn_f1s, frcnn_f1s])
        fig_size = (5,7)
        #font_size = 10
        # custom_multiple_subplots(path ,Xs, Ys, labels, title=[], font_size=12, x_axis="", y_axis="", symbols = ["o-", "x-", "v-"], fig_size=(10,4), colors=[], line_styles=[]):        
        #make_nice_line_plot(data_dir + partition + ".pdf", [distances, distances], [fcnn_f1s, frcnn_f1s], ["FCNN", "FRCNN"], title=title, font_size=16, x_axis=x_axis, y_axis=y_axis, colors=colors, line_styles=line_styles)
    custom_multiple_subplots(data_dir + "results.pdf", 2, 1, Xs, Ys, ["FCNN", "FRCNN"], fig_size=fig_size,title=titles, font_size=18, x_axis=x_axis, y_axis=y_axis)
    print("done")
    



        

