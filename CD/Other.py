#####################
# Theory
#####################

# MSE: 
# $ 1/n \sum^n_{i=1} (y_i−x_iβ)^2 $
"""
y_hat   = X @ beta
res     = y - y_hat

MSE     = np.mean(res**2)
or 
MSE     = (1/n) * (np.sum(res**2))  

"""

# RSS
# $ ∥y − Xβ∥^2_2 $
"""
y_hat   = X @ beta
res     = y - y_hat

RSS     = np.sum(res **2)
"""

# TSS
# $ TSS = ∥y − y_bar∥_2^2 $
"""
np.sum((y - np.mean(y))** 2)
"""

# MAE
# $ 1/n \sum^n_{i=1} |y_i−x_iβ| $



#####################
# other:
#####################

from __future__ import division
from statsmodels.sandbox.stats.multicomp import multipletests  


import warnings 
with warnings.catch_warnings(): # done to disable all the convergence warnings from elastic net
    warnings.simplefilter("ignore")
"""
use it as a loop block
"""


from pathlib import Path
path = Path().cwd()
data_file = path.parent / "Data" / "Actors.csv"


