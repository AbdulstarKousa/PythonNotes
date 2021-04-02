"""
@author: Abdulstar Kousa
Python notes
"""

#################################
# Comments:
#################################

# one line
"""
multi lines command
"""

#################################
# Libraries:
#################################
import sys 
import random 
import os


#################################
# Print:
#################################

# print command 
print("string")

# print n time
print("string 100 time",100)

# pass in arguments 
print("%s %d" % ("string",2))
print("{} {}".format("Abdul",2))

# force not having a new line
print("string",end="")              

# abbriviations 
"""
\n      =   new line
%c      = char
%d      = int 
{:.2f} = float 
%s      = strings
"""

# smile
print(u'\U0001f60a' u'\U0001f618' u'\U0001f604')
print('  _________\n /         \\\n |  /\\ /\\  |\n |    -    |\n |  \\___/  |\n \\_________/');



#################################
# input:
#################################
sys.stdin.readline()


#################################
# operators:
#################################
"""
+ - * / 
%  : remainder of division 
** : power
// : times of division without remainder 
== != > < >= <= 
and or not
"""

#################################
# variables:
#################################
 
"""
you don't need to specify the type of the var when declaring it.
"""



#################################
# strings:
#################################

# slicing 
str[index1:index2]      

# concatenate
str1+str2               

# capitalize
str.capitalize()

# find
str.find("string")

# isString
str.isalpha()

# isNumber
str.isalnum()

# replace
str.replace("string1","string2")

# space seprated values returns a lst of the seprated parts
str.split(" ")

# convert list to string
str1 = ''.join(str(e) for e in x)

# convert string into list
y = list(str)

# count how many times a letter or sets of letters apeares 
str1.count('h')


#################################
# lists:
#################################
# format
ls=[val1,...,valn]

# get
ls[index] 

# slice
ls[index1:index2]

# append a val
ls.append(val)

# concatenate                            
ls= ls1+ls2                             

# insert in a specific place
ls = [lst]
ls.insert(index,val)
ls

# Remove the first appearance of a value
ls = [lst]
ls.remove(1)
ls

# remove an index
del ls[index]

# sort
ls.sort()

# reverse
ls.reverse()

# More stuff
len(lst)
min(lst)
max(lst)

# convert list to string
str1 = ''.join(str(e) for e in x)

# convert string into list
y = list(str)


#################################
# matrix:
#################################

# format
ls=[ls1,ls2]

# get
ls[rowIndex][colomnIndex]



#################################
# tuples:
#################################
"""
cannot be cheanged once created
other than the obove it is the same as lists
"""
list(tup)
tuple(ls)



#################################
# dictionaries:
#################################

# format
dic={key1:val1,....,keyn:valn}
dic= dict(zip(keys, values))

# get a key value
dic[key] 

# set a key value
dic[key]=newVal

# get keys 
dic.keys()

# get values 
dic.values()


#################################
# if statement:
#################################
if b : c  
elif b : c
else : c


#################################
# loops:
#################################

# for:
for var in range(val1,val2) : c
for var in lst: c
 
# while:
while(b): c  
"""
remember that you need a start condition and end case
"""

# while true
while True: if b : break


#################################
# functions:
#################################

# build
def functionName(attributes): 
    c 
    return val

# call
functionName(attributes)



#################################
# in-out (txt):
#################################
# writing
open("fileName.extinsion","wb")
fileVarName.write(bytes("String",'UTF-8'))
fileVarName.close()                        

# read and write
open("fileName.extinsion","r+")             
fileVarName.read()
fileVarName.close()

# remove file
os.remove("fileName.extinsion")

# More
print(fileVarName.mode)
print(fileVarName.name)


#################################
# O.O.P:
#################################

# for more see the link
"""
https://www.youtube.com/watch?v=pF7xdh4DW-o&list=PL3072C720775B213E&index=6
"""

#### BluePrint:
class className:
    # Atributes:
    __atribute=val                                                              #__:private
    
    # Consructor:
    def __init__(self,parameters):
        self.attribute=parameter

    # Methods:
    """ Get """     
    def get_attribute(self):
        return self.__attribute
    
    """ Set """    
    def set_attribute(self,parameters): 
        self.__attribute= parameterValue        
    
    """ fun """
    def functionName(self,parameters):
        commands          
    
    """ override """
    def __add__(self,other):
        return self.attribute + other.attribute

    """ toString """
    def toString(self):
        return "{} string {}".format(self.__attribute1,self.__attribute2)
    
#### inheritance:
class subClassName(superClassName):
    # Attributes:
    """as above"""
    
    # constructor:
    def __init__(self,Parameters_For_The_Sub_Class):
        """as above"""
        super(subClassName,self).__init__(attributesForTheSuperClass)

#### creating an object:
object = className(constructor's parameters)  


# Example:
class Vector:

    # Atributes: x, y, z
    __x=0                                                              
    __y=0
    __z=0
    
    # Constructor:
    def __init__(self,x,y,z):
        self.x=x
        self.y=y
        self.z=z

    # Methods:
    """ override: +  """        
    def __add__(self,other):
        return Vector(self.x + other.x,self.y + other.y,self.z + other.z) 
    
    """ override: -  """                
    def __sub__(self,other):
        return Vector(self.x - other.x,self.y - other.y,self.z - other.z)

    """ override: *  """                    
    def __mul__(self,other):
        return Vector(self.x * other.x,self.y * other.y,self.z * other.z)
    
    """ fun """
    def norm(self):
        import math
        return math.sqrt(self.x*self.x + self.y*self.y + self.z*self.z)
    
    """ fun """
    def print(self):
        print("({},{},{})".format(self.x,self.y,self.z))
    
V1=Vector(1,1,1) 
V2=Vector(2,2,2) 
V3=V1+V2 
print(V3.norm())
V3.print()



#################################
# More:
################################# 

# help
?commandOrval                                     

# get the type of value
type(val)

#to break loops
break                                     

# continue in to next loop cycle without seeing the rest of the code
continue                                 

# pass a structure within a loop and continue reading the code
pass                                     

# lack of value (tis is a float)
None                                     

# access var from out the function
global

# even number
Nr%2==0                                  

# odd  number
Nr%2!=0                                  

# exit
import sys
sys.exit(0)

# random
import random
random.random()

# zip 
zip(lst1,lst2)
zip(*lst)


#################################
# examples: 
#################################
"""
import sys, gc

def create_cycle():
    list = [8, 9, 10]
    list.append(list)

def main():
    print("Creating garbage...")
    for i in range(8):
        create_cycle()

    print("Collecting...")
    n = gc.collect()
    print("Number of unreachable objects collected by GC:", n)
    print("Uncollectable garbage:", gc.garbage)

if __name__ == "__main__":
    main()
    sys.exit()
"""