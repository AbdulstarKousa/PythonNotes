# #################################
# # O.O.P:
# #################################

# # for more see the link
# """
# https://www.youtube.com/watch?v=pF7xdh4DW-o&list=PL3072C720775B213E&index=6
# """

# #### BluePrint:
# class className:
#     # Atributes:
#     __atribute=val                                                              #__:private
    
#     # Consructor:
#     def __init__(self,parameters):
#         self.attribute=parameter

#     # Methods:
#     """ Get """     
#     def get_attribute(self):
#         return self.__attribute
    
#     """ Set """    
#     def set_attribute(self,parameters): 
#         self.__attribute= parameterValue        
    
#     """ fun """
#     def functionName(self,parameters):
#         commands          
    
#     """ override """
#     def __add__(self,other):
#         return self.attribute + other.attribute

#     """ toString """
#     def toString(self):
#         return "{} string {}".format(self.__attribute1,self.__attribute2)
    
# #### inheritance:
# class subClassName(superClassName):
#     # Attributes:
#     """as above"""
    
#     # constructor:
#     def __init__(self,Parameters_For_The_Sub_Class):
#         """as above"""
#         super(subClassName,self).__init__(attributesForTheSuperClass)

# #### creating an object:
# object = className(constructor's parameters)  


# # Example:
# class Vector:

#     # Atributes: x, y, z
#     __x=0                                                              
#     __y=0
#     __z=0
    
#     # Constructor:
#     def __init__(self,x,y,z):
#         self.x=x
#         self.y=y
#         self.z=z

#     # Methods:
#     """ override: +  """        
#     def __add__(self,other):
#         return Vector(self.x + other.x,self.y + other.y,self.z + other.z) 
    
#     """ override: -  """                
#     def __sub__(self,other):
#         return Vector(self.x - other.x,self.y - other.y,self.z - other.z)

#     """ override: *  """                    
#     def __mul__(self,other):
#         return Vector(self.x * other.x,self.y * other.y,self.z * other.z)
    
#     """ fun """
#     def norm(self):
#         import math
#         return math.sqrt(self.x*self.x + self.y*self.y + self.z*self.z)
    
#     """ fun """
#     def print(self):
#         print("({},{},{})".format(self.x,self.y,self.z))
    
# V1=Vector(1,1,1) 
# V2=Vector(2,2,2) 
# V3=V1+V2 
# print(V3.norm())
# V3.print()



# #################################
# # Main
# #################################
# if __name__ == "__main__": 
#     # Question 01.AB
#     Run_Question_1_A_B()
#     # Question 01.C
#     Run_Question_1_C()
#     # Question 02
#     Run_Question_2()