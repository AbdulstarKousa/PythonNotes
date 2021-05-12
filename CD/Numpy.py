#####################
# numpy
#####################


np.random.choice(a, size=None, replace=True)
"""
Generates a random sample from a given 1-D array
"""

np.nonzero(coef)[1]
"""
non zero elements of an array
"""


.ravel()
"""
array of array -> array of values  
"""

np.loadtxt(diabetPath, delimiter = ' ', skiprows = 1, , usecols=[1,2,3,4,5,6,7,8,9])
"""
To improt data as a matrix 
skiprows:   here it does not start from 0
delimiter:  the separator  
"""

np.linalg.inv(x)
"""
calculate the inv of a Matrix
"""

np.matmul(X.T, X)
"""
Matrix multiplication 
"""

np.linalg.norm(beta_ana-beta_num)
"""
calculate the norm (length of the vector from the origin)
"""

np.column_stack((off, X))
"""
stack a column to a matrix 
"""

np.ones(n)
"""
create a vector of size n with 1's
"""

np.mean(res)
"""
mean of array
"""

np.abs()
"""
abs
"""

np.sqrt()
"""
sqrt
"""

np.random.normal(loc=0.0, scale=1.0, size=None)
"""
create a normally distributed sample 
"""

np.random.randn(n, p)
"""
create a matrix of size n,p. With float numbers from the standard normal distribution 
"""


numpy.random.randint(low, high=None, size=None, dtype='l')
"""
Return random integers from low (inclusive) to high (exclusive)
"""

np.eye(p,p)
"""
create identity matrix 
"""

np.logspace(start, stop, k ,base=10.0)
"""
create values 10**start to 10**stop of size k 
"""

np.argsort(distances)
"""
returns an array of indices of the same shape as a that index data along the given axis in sorted order.
"""

np.random.permutation(n)
"""
permutate data randomly without replacement. 
F.ex: data[np.random.permutation(len(data))] 
"""

np.argsort(lst)
"""
list of the index of the cor. sorted array
"""

np.where(meanMSE[jOpt] + seMSE[jOpt] > meanMSE)
"""
list of index
"""

numpy.trace(a, offset=0, axis1=0, axis2=1, dtype=None, out=None)
"""
Return the sum along diagonals of the array.
"""

math.ceil(n/CV)
"""
Ceil
"""

nplst.ravel() 
"""
ravel collapses the array, ie dim(x,1) to (x,)
"""

X.T
"""
Transprose
"""

X.T @ y
"""
matrix multiplications
"""
