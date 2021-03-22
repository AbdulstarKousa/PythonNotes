import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from math import log


#========================= D01

# plt.hist(L, bins=100, density=True)
# plt.plot([mu, mu], [0,.5], c="black") # Line
# plt.annotate(s="$\mu$",xy=(mu+.1,.47), size=20)
# plt.scatter(x,y,alpha=0.1)

# pd.DataFrame([[0.068966, 0.137931, 0.068966], [0.344828, 0.241379, 0.137931]], columns=['T=Hot', 'T=Mild', 'T=Cold'], index=['W=Sunny', "W=Cloudy"])
# np.random.randint()
# z_items, z_counts=np.unique(z, return_counts=True, axis=0)
# for item, ct in zip(z_items, z_counts):     
# np.stack((z,t), axis=-1)

# from collections import defaultdict
# z_dict=defaultdict(list)
# z_dict[Z].append([X,Y])

# from sklearn.linear_model import LinearRegression
# lr1=LinearRegression(fit_intercept=False)
# lr1.fit(z.T, x.T)
# print(lr1.coef_)

# np.random.normal(size=5000)
# np.random.multivariate_normal([0,0], [[1,.6],[.6, 1]], size=5000)


"""
Function p. It receives a list of values, K, and simply prints the probability of each of its values 
nam and given - Two strings just for printing. Just run this once and you'll see... ;-) 
"""
def p(nam, K, given=""):    
    k_items, k_counts=np.unique(K, return_counts=True, axis=0)
    for item, ct in zip(k_items, k_counts):
        print("p(%s=%s%s)=%f"%(nam, item, given, ct/sum(k_counts)))



#========================= D03

# beta=np.random.multivariate_normal([10, 0], np.eye(2)*sigmab)
# zn=np.random.binomial(1, pi)
# kn=np.random.normal(beta[0]+beta[1]*X[n], sigmae)


def Gaussian_loglikelihood(X, mu, sigma):
    n=len(X)
    return -n/2*log(2*np.pi)-n/2*log(sigma**2)-1/(2*sigma**2)*sum([(x-mu)**2 for x in X])

def MLE(X, sigma):
    l=[]
    maxll=-9999999
    bestmu=-9999999
    mus=np.arange(-100, 100, 0.01)
    for mu in mus:
        ll=Gaussian_loglikelihood(X, mu, sigma)
        l.append(ll)
        if ll>maxll:
            bestmu=mu
            maxll=ll
    return bestmu, mus, l



def log_mu_prior(mu, alpha, gamma):
    return -1/2*log(2*np.pi)-1/2*log(gamma**2)-1/(2*gamma**2)*(alpha-mu)**2

def log_posterior_mu(X, sigma, alpha, gamma):
    lposterior=[]
    MAPll=-9999999
    MAP=-9999999
    mus=np.arange(-100, 100, 0.01)
    for mu in mus:
        ll=Gaussian_loglikelihood(X, mu, sigma)+log_mu_prior(mu, alpha, gamma)
        lposterior.append(ll)
        if ll>MAPll:
            MAP=mu
            MAPll=ll
    return MAP, mus, lposterior
        

from scipy.stats import norm, dirichlet, multivariate_normal
"""
see last functions from w03
"""

