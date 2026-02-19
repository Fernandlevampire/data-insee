import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import optimize

import utils

def normal_density(df: pd.DataFrame, col:str):
    """
    displays the normal nature of a distribution of data
    among key parameters, it highlights :
    - the size of the dataset
    - the value for which the gaussian approximation reaches a maximum (value with highest frequency)
    - the coefficient of determination R²
    """
    data = df[col]

    fig = plt.figure(figsize=(8, 5))
    ax = fig.subplots(1)

    # set axes format        
    ax.set_facecolor("#2C2C2C")
    ax.grid(True, linestyle="--", alpha=0.7, color="#FFFFFF")

    # basic hist displays data density
    n_bins = min(int(len(data)/10), 50)
    y, *_ = ax.hist(data, bins=n_bins, weights=np.ones_like(data)/len(data), edgecolor="white", alpha=0.7, facecolor="#959595")

    # asymetric gaussian distribution
    def gauss(x, a, x0, sigma):
        return a*np.exp(-(x-x0)**2 / (2*sigma**2))
    
    x = np.linspace(data.min(),data.max(), n_bins)
    (a, x0, sigma), _ = optimize.curve_fit(gauss, x, y, p0=[250, 4800, 1000])
    result = utils.model_estimate(y, gauss, x,(a, x0, sigma))

    ax.plot(x, np.vectorize(lambda n: gauss(n, a, x0, sigma))(x), color="white", linestyle=":", linewidth=2)
    ax.scatter(x0, a, facecolor="white")

    # text box and annotations
    box_text = "\n".join([
        f"# ventes : {utils.reformat_number(data.count())}"
        , f"Prix type : {utils.reformat_number(x0)} €/m²"
        , f"R² modèle : {utils.reformat_number(result['r_squared'])}"
    ])

    props = dict(boxstyle="round", facecolor="white", pad=0.5)

    # place a text box up right
    ax.text(0.70, 0.96, box_text, fontsize=10,
            verticalalignment="top",
            transform=ax.transAxes, bbox=props)
    
    # annotate max frequency point
    annotation_text = f"{utils.reformat_number(x0)} €/m²"
    kw = dict(arrowprops=dict(arrowstyle="-", ec="w", connectionstyle="angle,angleA=-0,angleB=65"), bbox=dict(fc="none", ec="k", lw=.0), zorder=0, va="center")
    ax.annotate(annotation_text, xy=(x0, a), xytext=(x0*1.1, a*1.06), color="white", **kw)

    plt.tight_layout()
    plt.show()

def bubble_scatter(df: pd.DataFrame, col_y:str, col_x:str):
    # aggregate the y values by x parameter. We need mean and count agglomerates to visualize the evolution
    df_agg = df[[col_y,col_x]].groupby(by=col_x).agg({col_y:["mean","count"]})

    # rename columns for better readability
    col_y_mean, col_y_count = f"{col_y}_mean", f"{col_y}_count"
    df_agg.columns = [col_y_mean, col_y_count]

    # set basic geometric parameters
    nb_x, nb_y = df_agg.shape[0], df_agg[col_y_count].sum()
    ax_width = max(int(nb_x*1.3), 4)
    fig, ax = plt.subplots(figsize=(ax_width, 5))
    
    # set display parameters
    x_range = np.arange(nb_x)    
    min_y, max_y = int(df_agg[col_y_mean].min()/1000), int(df_agg[col_y_mean].max()/1000)
    y_range = 1000*np.r_[min_y-0.5:max_y+1.5:0.25]
    count_weight = nb_x*(max_y-min_y)*1000/nb_y

    # display a bubble scatter plot with the center of the bubbles
    ax.scatter(x=x_range, y=df_agg[col_y_mean], s=df_agg[col_y_count]*count_weight, facecolors="pink", edgecolors="r")
    ax.scatter(x=x_range, y=df_agg[col_y_mean], facecolors="r", marker="+")
    
    # show vertical grid lines but hide tick labels
    ax.tick_params(axis='x', which='both', labelbottom=False)
    ax.set_xlim(-0.5, max(x_range)+0.5)
    ax.set_yticks(y_range)
    ax.grid(True, which='both', linestyle="--", alpha=0.7, color="gray")

    # create an annotation above the scatter 
    x_pos = 0
    for x, y_mean, y_count in df_agg.itertuples(index=True):
        # compute the vertical text position in offset points
        offset_pts = np.sqrt(y_count*count_weight)/2
        ax.annotate(
            text=x,
            xy=(x_pos, y_mean),
            xytext=(0, offset_pts+5),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color="red",
            fontweight="bold"
        )
        x_pos+=1

    # create a nicely formatted table below the plot
    df_format = df_agg.copy()
    df_format[col_y_mean] = df_format[col_y_mean].apply(utils.reformat_number)
    df_format[col_y_count] = df_format[col_y_count].apply(utils.reformat_number)
    df_format = df_format.T
    
    table = ax.table(
        df_format,
        cellLoc="center",
        bbox=[0, -0.3, 1, 0.2],
        fontsize=10
        )
    # make the uppermost row bold for emphasis
    for i in np.arange(nb_x):
        table[0, i].set_text_props(fontweight='bold')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    pass