import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import numpy as np

def corrfunc(x, y, **kws):
    (r, p) = pearsonr(x, y)
    ax = plt.gca()
    ax.annotate("r = {:.6f} ".format(r),
                xy=(.1, .9), xycoords=ax.transAxes)

    MAE = np.mean(np.abs(x - y))
    ax.annotate("MAE = {:.6f}".format(MAE),
                xy=(.1, .8), xycoords=ax.transAxes)
    RMSE = np.sqrt(np.mean((x - y) ** 2))
    ax.annotate("RMSE = {:.6f}".format(RMSE),
                xy=(.1, .7), xycoords=ax.transAxes)
    ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c=".3")


class Plotter:
    def __init__(self, data, uncertainty):
        self.data = data
        self.uncertainty = uncertainty

    # def generate_plots(self):
    #     # Generate a regression line plot
    #     plt.figure(figsize=(10, 6))
    #     plt.plot(self.data['Eref'], self.data.mean(axis=1), 'o')
    #     plt.plot(self.data['Eref'], self.data['Eref'], 'r-')
    #     plt.xlabel('Eref')
    #     plt.ylabel('Model prediction')
    #     plt.title('Regression line plot')
    #     plt.show()
    #
    #     # Print the r squared, Pearson's r and other statistics
    #     r, p = pearsonr(self.data['Eref'], self.data.mean(axis=1))
    #     print(f"Pearson's r: {r}, p-value: {p}")
