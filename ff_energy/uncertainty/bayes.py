import os
import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats
import numpyro


plt.style.use("bmh")
if "NUMPYRO_SPHINXBUILD" in os.environ:
    set_matplotlib_formats("svg")

assert numpyro.__version__.startswith("0.11.0")

