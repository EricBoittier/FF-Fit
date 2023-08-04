"""
Matplotlib Style Sheet based on Science Plots
"""

import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import pandas as pd
import seaborn as sns


def set_style():
    plt.style.use('science')
    rcParams['figure.figsize'] = 8, 6
    rcParams['axes.labelsize'] = 18
    rcParams['axes.titlesize'] = 22
    rcParams['xtick.labelsize'] = 18
    rcParams['ytick.labelsize'] = 18
    rcParams['legend.fontsize'] = 18
    rcParams['font.family'] = 'san-serif'
    rcParams['font.sans-serif'] = ['DejaVu Sans']
    # bold title and labels
    rcParams['axes.labelweight'] = 'bold'
    rcParams['axes.titleweight'] = 'bold'
    rcParams['axes.linewidth'] = 2
    rcParams['xtick.major.width'] = 2
    rcParams['ytick.major.width'] = 2
    rcParams['xtick.minor.width'] = 2
    rcParams['ytick.minor.width'] = 2

    # set colors
    rcParams['axes.prop_cycle'] = plt.cycler(
        color=plt.cm.Dark2.colors)
