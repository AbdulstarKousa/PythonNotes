"""
@author: Abdulstar Kousa
Numpy notes
"""
#################################
# libraries:
#################################
import numpy as np
#something
    
#################################
#build:
#################################

# format 
v=np.array([v0,..,vn])
v=np.array(ls)
v=np.zeros(Nr)  
v=np.ones(Nr) 
v=np.empty(Nr)

# linspace:
v=np.linspace(startNr,endNr,nr_Of_Elements_Within_The_Interval)  

# range of steps
v=np.arange(start=2,stop=17,step=4)                             

# random array 
v=np.random.randint(maxNr,size=nr)



#################################
# slicing 
#################################

# get
v[index]
v[-1]

# range slicing
v[sartIndex:endIndex]

# step slicing
v[0::2] """ start:end:step """

# conditional slicing
v[v arithmeticLogicalOperator nr]




#################################
# math:
#################################
# arithmetic operations element wise  
v1 arithmeticOperator v2

# arithmeticLogical element wise
v arithmeticLogicalOperator nr  

# dot product
v1 @ v2

# min/max value 
np.max(v)
np.min(v)

# index of max/min value
np.argmax(v)
np.argmin(v)                                        

# return sum/product of elements 
np.sum(v) """ also used to count true"""
np.prod(v)

# sin/cos
np.sin(v)
np.cos(v)

# Stat:
np.mean(v)
np.std(v)
np.var(v)

# Transpose
v[::-1]    
v.t

#################################
# Diminsions
#################################

# length 
len(v)

# get diminsions 
v.shape

# change to one colomn Matrix 
v.shape=(len(v),1)

# reshape 
v.reshape(2,2)                                                 

# Transpose
v[::-1]    
V.T


#################################
# More
#################################
np.where(v arithmaticLogicalOprator nr, replaceValueIfTrue, replaceValueIfFalse)


#################################
# Matrixs: (use the arrays commands) see exer0
#################################

# build 
M=np.array([[rowArray0],..,[rowArrayN]])

# slicing 
M[startRow:endRow,startCol:endCol] 
M[::2,::2] """ start:end:step """

# get the shape  
M.shape 

# Transpose
M.T 

# inverse
np.linalg.inv(v)