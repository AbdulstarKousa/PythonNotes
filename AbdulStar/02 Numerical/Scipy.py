# #####################
# # scipy
# #####################
# import scipy.io
# import scipy.linalg as lng
# from scipy import linalg
# from scipy.spatial import distance
# from scipy.stats import linregress


# mat = scipy.io.loadmat('Silhouettes.mat')
# X = mat['X']
# y = mat['Y']
# """
# load .mat data
# """

# distance.euclidean(X[i,:], X[j, :])
# """
# euclidean distance between two vectors
# """

# betas, res_SS, _, _ = lng.lstsq(X, y)
# """
# numerically ’smarter’ solve ols
# to invert the matrix or to solve the linear system of equation.
# """

# slope, intercept, r_value, PValues[j], std_err = linregress(Xsub, y)
# """
# linear least square reg with pvalues
# """
