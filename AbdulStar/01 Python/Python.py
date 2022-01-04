# """
# @author: Abdulstar Kousa
# Python notes
# """

# #################################
# # Comments:
# #################################

# # one line
# """
# multi lines command
# """

# #################################
# # Libraries:
# #################################
# import sys 
# import os


# #################################
# # Print:
# #################################

# # print command 
# print("string")

# # print n time
# print("string 100 time",100)

# # pass in arguments 
# print("%s %d" % ("string",2))
# print("{} {}".format("Abdul",2))

# # force not having a new line
# print("string",end="")              

# # abbriviations 
# """
# \n      =   new line
# %c      = char
# %d      = int 
# {:.2f} = float 
# %s      = strings
# """

# # smile
# print(u'\U0001f60a' u'\U0001f618' u'\U0001f604')
# print('  _________\n /         \\\n |  /\\ /\\  |\n |    -    |\n |  \\___/  |\n \\_________/');



# #################################
# # input:
# #################################
# sys.stdin.readline()


# #################################
# # operators:
# #################################
# """
# + - * / 
# %  : remainder of division 
# ** : power
# // : times of division without remainder 
# == != > < >= <= 
# and or not
# """

# #################################
# # variables:
# #################################
 # """
# you don't need to specify the type of the var when declaring it.
# """



# #################################
# # strings:
# #################################

# # slicing 
# str[index1:index2]      

# # concatenate
# str1+str2               

# # capitalize
# str.capitalize()

# # find
# str.find("string")

# # isString
# str.isalpha()

# # isNumber
# str.isalnum()

# # replace
# str.replace("string1","string2")

# # space seprated values returns a lst of the seprated parts
# str.split(" ")

# # convert list to string
# str1 = ''.join(str(e) for e in x)

# # convert string into list
# y = list(str)

# # count how many times a letter or sets of letters apeares 
# str1.count('h')


# #################################
# # lists:
# #################################
# # format
# ls=[val1,...,valn]

# # get
# ls[index] 

# # slice
# ls[index1:index2]

# # append a val
# ls.append(val)

# # concatenate                            
# ls= ls1+ls2                             

# # insert in a specific place
# ls = [lst]
# ls.insert(index,val)
# ls

# # Remove the first appearance of a value
# ls = [lst]
# ls.remove(1)
# ls

# # remove an index
# del ls[index]

# # sort
# ls.sort()

# # reverse
# ls.reverse()

# # More stuff
# len(lst)
# min(lst)
# max(lst)

# # convert list to string
# str1 = ''.join(str(e) for e in x)

# # convert string into list
# y = list(str)


# #################################
# # matrix:
# #################################

# # format
# ls=[ls1,ls2]

# # get
# ls[rowIndex][colomnIndex]



# #################################
# # tuples:
# #################################
# """
# cannot be cheanged once created
# other than the obove it is the same as lists
# """
# list(tup)
# tuple(ls)



# #################################
# # dictionaries:
# #################################

# # syntax
# dic = {key1:val1, ..}
# dic = dict(zip(keys, values))

# # Initilize and empty dic
# dic = {}

# # Add new_key:new_value 
# dic[new_key]= new_value

# # get a key value
# dic[key] 

# # set a key value
# dic[key]=newVal

# # get keys 
# dic.keys()

# # get values 
# dic.values()



# #################################
# # Map:
# #################################
# # Syntax 
# map(func, lst)

# # Example: 
# map(str, Table['Hours'].values)

# # Get values: 
# list(map(str, Table['Hours'].values))




# #################################
# # if statement:
# #################################
# if b : c  
# elif b : c
# else : c


# #################################
# # loops:
# #################################

# # for:
# for var in range(val1,val2) : c
# for var in lst: c
 
# # while:
# while(b): c  
# """
# remember that you need a start condition and end case
# """

# # while true
# while True: if b : break


# #################################
# # functions:
# #################################

# # build
# def functionName(attributes): 
#     c 
#     return val

# # call
# functionName(attributes)



# #################################
# # in-out (txt):
# #################################
# # writing
# open("fileName.extinsion","wb")
# fileVarName.write(bytes("String",'UTF-8'))
# fileVarName.close()                        

# # read and write
# open("fileName.extinsion","r+")             
# fileVarName.read()
# fileVarName.close()

# # remove file
# os.remove("fileName.extinsion")

# # More
# print(fileVarName.mode)
# print(fileVarName.name)


# #################################
# # Generators and Iterators
# #################################
# def lazy_range(n):
#     """a lazy version of range"""
#     i = 0
#     while i < n:
#         yield i
#         i += 1

# def natural_numbers():
#     """returns 1, 2, 3, ..."""
#     n = 1
#     while True:
#         yield n
#         n += 1
# 10000000000000000000 in natural_numbers()

# x = iter(natural_numbers())
# next(x)




# #################################
# # loop else
# #################################
# """ reach else if the loop was done """
# for i in range(5):
#     pass
# else: 
#     print("the loop is done!") 

# """ can't reach else if the loop broken """
# for i in range(10):
#         break     
# else: 
#     print("this can't be reached!") 








# #################################
# # More:
# ################################# 

# # help
# ?commandOrval                                     

# # get the type of value
# type(val)

# #to break loops
# break                                     

# # continue in to next loop cycle without seeing the rest of the code
# continue                                 

# # pass a structure within a loop and continue reading the code
# pass                                     

# # lack of value (tis is a float)
# None                                     

# # access var from out the function
# global

# # even number
# Nr%2==0                                  

# # odd  number
# Nr%2!=0                                  

# # exit
# import sys
# sys.exit(0)

# # random
# import random
# random.random()

# # zip 
# zip(lst1,lst2)
# zip(*lst)


# #################################
# # examples: 
# #################################
# """
# import sys, gc

# def create_cycle():
#     list = [8, 9, 10]
#     list.append(list)

# def main():
#     print("Creating garbage...")
#     for i in range(8):
#         create_cycle()

#     print("Collecting...")
#     n = gc.collect()
#     print("Number of unreachable objects collected by GC:", n)
#     print("Uncollectable garbage:", gc.garbage)

# if __name__ == "__main__":
#     main()
#     sys.exit()
# """



# def main():
#     test_lcg()
#     test_system_available_generator()

# if __name__ == "__main__":
#     main()



# print(f'{"mean =":<6} {np.round(crude_values["mean"],3)}\n'\
#       f'{"std  =":<6} {np.round(crude_values["std"],3)}\n'\
#       f'{"CI   =":<6} ({np.round(crude_values["CI"][0],3)}, {np.round(crude_values["CI"][1],3)})')


