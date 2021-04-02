#####################
# plot:
#####################
# import matplotlib.pyplot as plt
# import matplotlib.colors as colors
# import seaborn as sns # "prettify" default matplotlib
# sns.set() # Set searborn as default

# plt.figure()
"""
create a figure
"""

# fig = plt.figure(figsize=(15,15))
# fig.subplots_adjust(hspace=0.4, wspace=0.4)
"""
multiple plots adjust the spaces between plots
"""

# plt.axis([xmin, xmax, ymin, ymax])
"""
set dim
"""

# ax.plot(betas.T, ".")
"""
scatter plot of matrix 
the variables should be columns in the matrix
x-axis is from 0, .., n
"""

# ax2.boxplot(betas.T)
"""
boxplot of matrix
the variables should be columns in the matrix
"""

# multiple plots without loop Example 01
""""
fig = plt.figure(figsize=(15,15))

ax = fig.add_subplot(2, 2, 1)
ax.plot(betas.T, ".")

ax2 = fig.add_subplot(2,2,2)
ax2.boxplot(betas.T)

plt.show
"""

# multiple plots without loop Example 02
"""
fig, ax = plt.subplots(1,2, figsize=(15,5))    
ax[0].plot(K, err_tr, 'b', label='train')
ax[0].legend()
ax[0].set_xlabel('k')
ax[0].set_ylabel('error estimate')
ax[0].set_title("error estimate")

ax[1].plot(K, np.log(err_tr), 'b', label='train')
ax[1].legend()
ax[1].set_xlabel('k')
ax[1].set_ylabel('error estimate')
ax[1].set_title("Log error estimate")
plt.show()
"""

# plt.semilogx(lambdas, betas_mean)
"""
log x axis
change to plt.semilogy(lambdas, betas_mean) for y axis 
change to loglog 
"""

# plt.errorbar(lambdas, testError, testStd, marker='.', color='orange', markersize=10)
"""
error bars 
"""







