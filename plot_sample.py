# %%
from torch_wrapper.analysis import Plotter
from torch_wrapper.tuning import Metric

# %%
plotter = Plotter()

# %%
plotter.plot2d([Metric.Accuracy, Metric.LOSS], filename="results.txt")
