#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import sympy as sym
from sympy.matrices import Matrix
from sympy import simplify, cos, sin
from sympy import *


# In[2]:


################CALCULATION OF THE JACOBIAN MATRIX###################


# In[3]:


def Jacobian(v_str, f_list, f1_list, f2_list):
    vars = sym.symbols(v_str)
    f =(f_list)
    f1 = (f1_list)
    f2 = (f2_list)
    #print (f)
    J = sym.zeros(len(vars))
    for j, s in enumerate(vars):
        J[0,j] = sym.diff(f, s)
        
    for j, s in enumerate(vars):
        J[1,j] = sym.diff(f1, s)
        
    for j, s in enumerate(vars):
        J[2,j] = sym.diff(f2, s)
  
    return J


# In[4]:


#function for matrix multiplication
def matrix_multiplication(v_str, f_list1, f_list2, f_list3, f_list4, f_list5, f_list6, f_list7):
 
    vars = sym.symbols(v_str)
    f1 = Matrix(f_list1)
    f2 = Matrix(f_list2)
    f3 = Matrix(f_list3)
    f4 = Matrix(f_list4)
    f5 = Matrix(f_list5)
    f6 = Matrix(f_list6)
    f7 = Matrix(f_list7)
    
    expr = f1*f2
    expr = expr*f3
    expr = expr*f4
    expr = expr*f5
    expr = expr*f6
    expr = expr*f7
    
    return expr    


# In[5]:


#we set the variables that we have and the matrixes from kinematics.py
vars = sym.symbols('q1 q2 q3 q4 q5 q6 q7 l1 l2 l3 l4 l5 theta1 theta2')
tf0_1 = Matrix([['cos(q1)' , '-sin(q1)' , '0' , '0'],['sin(q1)' , 'cos(q1)' , '0' , '0'],['0' , '0' , '1' , 'l1'],['0' , '0' , '0' , '1']])
tf1_2 = Matrix([['cos(q2)' , '-sin(q2)' , '0' , '0'],['0' , '0' , '1' , '0'],['-sin(q2)' , '-cos(q2)' , '0' , '0'],['0' , '0' , '0' , '1']])
tf2_3 = Matrix([['cos(q3)' , '-sin(q3)' , '0' , '0'],['0' , '0' , '-1' , '-l2'],['sin(q3)' , 'cos(q3)' , '0' , '0'],['0' , '0' , '0' , '1']])
tf3_4 = Matrix([['cos(q4)' , '-sin(q4)' , '0' , 'l3'],['0' , '0' , '-1' , '0'],['sin(q4)' , 'cos(q4)' , '0' , '0'],['0' , '0' , '0' , '1']])
tf4_5 = Matrix([['cos(q5)' , '-sin(q5)' , '0' , 'l4*sin(theta1)'],['0' , '0' , '-1' , '-l4*cos(theta1)'],['sin(q5)' , 'cos(q5)' , '0' , '0'],['0' , '0' , '0' , '1']])
tf5_6 = Matrix([['cos(q6)' , '-sin(q6)' , '0' , '0'],['0' , '0' , '-1' , '0'],['sin(q6)' , 'cos(q6)' , '0' , '0'],['0' , '0' , '0' , '1']])
tf6_7 = Matrix([['cos(q7)' , '-sin(q7)' , '0' , 'l5*sin(theta2)'],['0' , '0' , '1' , 'l5*cos(theta2)'],['-sin(q7)' , '-cos(q7)' , '0' , '0'],['0' , '0' , '0' , '1']])


# In[6]:


A07 = matrix_multiplication('q1 q2 q3 q4 q5 q6 q7 l1 l2 l3 l4 l5 theta1 theta2', tf0_1, tf1_2, tf2_3, tf3_4, tf4_5, tf5_6, tf6_7)


# In[7]:


A07


# In[9]:


mhtra = simplify(A07)


# In[10]:


px = mhtra[0,3]
py = mhtra[1,3]
pz = mhtra[2,3]


# In[11]:


J = Jacobian('q1 q2 q3 q4 q5 q6 q7', px,py,pz)


# In[12]:


#printing the jackobian matrix
J


# In[8]:


##### NOW WE COMPUTE THE CRITICAL POINTS AND THE ξS for the onjective function #####


# In[9]:


#For the computation of every critical point we only need the
#A04 matrix and some transpositions and rotations, that we have already
#computed in kinematics.py


# In[10]:


#a variation of the above function for matrix multiplication to compute A04
def matrix_multiplication_1(v_str, f_list1, f_list2, f_list3, f_list4):
 
    vars = sym.symbols(v_str)
    f1 = Matrix(f_list1)
    f2 = Matrix(f_list2)
    f3 = Matrix(f_list3)
    f4 = Matrix(f_list4)
    
    expr = f1*f2
    expr = expr*f3
    expr = expr*f4
    
    return expr   


# In[11]:


A04 = matrix_multiplication_1('q1 q2 q3 q4 l1 l2 l3', tf0_1, tf1_2, tf2_3, tf3_4)


# In[12]:


A04


# In[13]:


#one more variation of the above function for matrix multiplication to compute each critical point
def matrix_multiplication_2(v_str, f_list1, f_list2):
 
    vars = sym.symbols(v_str)
    f1 = Matrix(f_list1)
    f2 = Matrix(f_list2)
    
    expr = f1*f2
    
    return expr 


# In[14]:


#all the tra/rot matrices for the computation of critical points, from kinematics.py

tra1 = Matrix([['1' , '0' , '0' , '0'],['0' , '1' , '0' , '0'],['0' , '0' , '1' , '0.1062'],['0' , '0' , '0' , '1']])
rot1 = Matrix([['0.70710' , '0.70710' , '0' , '0'],['-0.70710' , '0.70710' , '0' , '0'],['0' , '0' , '1' , '0'],['0' , '0' , '0' , '1']])
tra1_2 = Matrix([['1' , '0' , '0' , '0'],['0' , '1' , '0' , '-0.063'],['0' , '0' , '1' , '0'],['0' , '0' , '0' , '1']])
tra2 = Matrix([['1' , '0' , '0' , '0.05'],['0' , '1' , '0' , '0'],['0' , '0' , '1' , '0'],['0' , '0' , '0' , '1']])
tra3 = Matrix([['1' , '0' , '0' , '0.05'],['0' , '1' , '0' , '0'],['0' , '0' , '1' , '0'],['0' , '0' , '0' , '1']])
tra4 = Matrix([['1' , '0' , '0' , '0'],['0' , '1' , '0' , '0'],['0' , '0' , '1' , '-0.0676'],['0' , '0' , '0' , '1']])
tra5 = Matrix([['1' , '0' , '0' , '0'],['0' , '1' , '0' , '0'],['0' , '0' , '1' , '-0.1473'],['0' , '0' , '0' , '1']])
rot6 = Matrix([['0.70710' , '0.70710' , '0' , '0'],['-0.70710' , '0.70710' , '0' , '0'],['0' , '0' , '1' , '0'],['0' , '0' , '0' , '1']])
tra6 = Matrix([['1' , '0' , '0' , '0.05'],['0' , '1' , '0' , '0'],['0' , '0' , '1' , '0'],['0' , '0' , '0' , '1']])
tra7 = Matrix([['1' , '0' , '0' , '0.05'],['0' , '1' , '0' , '0'],['0' , '0' , '1' , '0'],['0' , '0' , '0' , '1']])


# In[18]:


#the critical points
L1_0 = matrix_multiplication_2('q1 q2 q3 q4 l1 l2 l3', A04, tra1)
L1_1 = matrix_multiplication_2('q1 q2 q3 q4 l1 l2 l3', L1_0, rot1)
L1 = matrix_multiplication_2('q1 q2 q3 q4 l1 l2 l3', L1_1, tra1_2)
L2 = matrix_multiplication_2('q1 q2 q3 q4 l1 l2 l3', L1, tra2)
L3 = matrix_multiplication_2('q1 q2 q3 q4 l1 l2 l3', L2, tra3)
L4 = matrix_multiplication_2('q1 q2 q3 q4 l1 l2 l3', A04, tra4)
L5 = matrix_multiplication_2('q1 q2 q3 q4 l1 l2 l3', L3, tra5)
L6_0 = matrix_multiplication_2('q1 q2 q3 q4 l1 l2 l3', L5, rot6)
L6 = matrix_multiplication_2('q1 q2 q3 q4 l1 l2 l3', L6_0, tra6)
L7 = matrix_multiplication_2('q1 q2 q3 q4 l1 l2 l3', L6, tra7)


# In[19]:


#function to find the partial derivative of a function ('function') to
#the first variable in variables list ('v_str')

def Deriv(v_str, function):
    vars = sym.symbols(v_str)
    f = (function)
    #print (f)
    J = sym.zeros(len(f))
    for i, fi in enumerate(f):
            J[i] = sym.diff(fi, vars[0])
            
    return J


# In[21]:


#The function is 'sqrt((x - a)**2 + (y - b)**2)' where x,y are the
#coordinates of the critical point, and a,b are the coordinates of the obstacle

#so this is the way we computed all ξs for the objective function
#for example, for computing the ξs for critical point L10,
#the procedure is the following


# In[23]:


print(L3[0,3]) #print Px of L3
print("\n")
print(L3[1,3]) #print Py of L3


# In[24]:


#partial derivative to q1
ksi1 = Deriv('q1 q2 q3 q4 l1 l2 l3 a1 b1', ['sqrt((l2*sin(q2)*cos(q1) + l3*(-sin(q1)*sin(q3) + cos(q1)*cos(q2)*cos(q3)) + 0.1152573*(-sin(q1)*sin(q3) + cos(q1)*cos(q2)*cos(q3))*sin(q4) + 0.0261627*(-sin(q1)*sin(q3) + cos(q1)*cos(q2)*cos(q3))*cos(q4) + 0.1062*sin(q1)*cos(q3) + 0.0261627*sin(q2)*sin(q4)*cos(q1) - 0.1152573*sin(q2)*cos(q1)*cos(q4) + 0.1062*sin(q3)*cos(q1)*cos(q2) - a1)**2 + (l2*sin(q1)*sin(q2) + l3*(sin(q1)*cos(q2)*cos(q3) + sin(q3)*cos(q1)) + 0.1152573*(sin(q1)*cos(q2)*cos(q3) + sin(q3)*cos(q1))*sin(q4) + 0.0261627*(sin(q1)*cos(q2)*cos(q3) + sin(q3)*cos(q1))*cos(q4) + 0.0261627*sin(q1)*sin(q2)*sin(q4) - 0.1152573*sin(q1)*sin(q2)*cos(q4) + 0.1062*sin(q1)*sin(q3)*cos(q2) - 0.1062*cos(q1)*cos(q3) - b1)**2)'])
print(ksi1[0])


# In[25]:


#partial derivative to q2
ksi2 = Deriv('q2 q1 q3 q4 l1 l2 l3 a1 b1', ['sqrt((l2*sin(q2)*cos(q1) + l3*(-sin(q1)*sin(q3) + cos(q1)*cos(q2)*cos(q3)) + 0.1152573*(-sin(q1)*sin(q3) + cos(q1)*cos(q2)*cos(q3))*sin(q4) + 0.0261627*(-sin(q1)*sin(q3) + cos(q1)*cos(q2)*cos(q3))*cos(q4) + 0.1062*sin(q1)*cos(q3) + 0.0261627*sin(q2)*sin(q4)*cos(q1) - 0.1152573*sin(q2)*cos(q1)*cos(q4) + 0.1062*sin(q3)*cos(q1)*cos(q2) - a1)**2 + (l2*sin(q1)*sin(q2) + l3*(sin(q1)*cos(q2)*cos(q3) + sin(q3)*cos(q1)) + 0.1152573*(sin(q1)*cos(q2)*cos(q3) + sin(q3)*cos(q1))*sin(q4) + 0.0261627*(sin(q1)*cos(q2)*cos(q3) + sin(q3)*cos(q1))*cos(q4) + 0.0261627*sin(q1)*sin(q2)*sin(q4) - 0.1152573*sin(q1)*sin(q2)*cos(q4) + 0.1062*sin(q1)*sin(q3)*cos(q2) - 0.1062*cos(q1)*cos(q3) - b1)**2)'])
print(ksi2[0])


# In[26]:


#partial derivative to q3
ksi3 = Deriv('q3 q2 q1 q4 l1 l2 l3 a1 b1', ['sqrt((l2*sin(q2)*cos(q1) + l3*(-sin(q1)*sin(q3) + cos(q1)*cos(q2)*cos(q3)) + 0.1152573*(-sin(q1)*sin(q3) + cos(q1)*cos(q2)*cos(q3))*sin(q4) + 0.0261627*(-sin(q1)*sin(q3) + cos(q1)*cos(q2)*cos(q3))*cos(q4) + 0.1062*sin(q1)*cos(q3) + 0.0261627*sin(q2)*sin(q4)*cos(q1) - 0.1152573*sin(q2)*cos(q1)*cos(q4) + 0.1062*sin(q3)*cos(q1)*cos(q2) - a1)**2 + (l2*sin(q1)*sin(q2) + l3*(sin(q1)*cos(q2)*cos(q3) + sin(q3)*cos(q1)) + 0.1152573*(sin(q1)*cos(q2)*cos(q3) + sin(q3)*cos(q1))*sin(q4) + 0.0261627*(sin(q1)*cos(q2)*cos(q3) + sin(q3)*cos(q1))*cos(q4) + 0.0261627*sin(q1)*sin(q2)*sin(q4) - 0.1152573*sin(q1)*sin(q2)*cos(q4) + 0.1062*sin(q1)*sin(q3)*cos(q2) - 0.1062*cos(q1)*cos(q3) - b1)**2)'])
print(ksi3[0])


# In[27]:


#partial derivative to q4
ksi4 = Deriv('q4 q2 q3 q1 l1 l2 l3 a1 b1', ['sqrt((l2*sin(q2)*cos(q1) + l3*(-sin(q1)*sin(q3) + cos(q1)*cos(q2)*cos(q3)) + 0.1152573*(-sin(q1)*sin(q3) + cos(q1)*cos(q2)*cos(q3))*sin(q4) + 0.0261627*(-sin(q1)*sin(q3) + cos(q1)*cos(q2)*cos(q3))*cos(q4) + 0.1062*sin(q1)*cos(q3) + 0.0261627*sin(q2)*sin(q4)*cos(q1) - 0.1152573*sin(q2)*cos(q1)*cos(q4) + 0.1062*sin(q3)*cos(q1)*cos(q2) - a1)**2 + (l2*sin(q1)*sin(q2) + l3*(sin(q1)*cos(q2)*cos(q3) + sin(q3)*cos(q1)) + 0.1152573*(sin(q1)*cos(q2)*cos(q3) + sin(q3)*cos(q1))*sin(q4) + 0.0261627*(sin(q1)*cos(q2)*cos(q3) + sin(q3)*cos(q1))*cos(q4) + 0.0261627*sin(q1)*sin(q2)*sin(q4) - 0.1152573*sin(q1)*sin(q2)*cos(q4) + 0.1062*sin(q1)*sin(q3)*cos(q2) - 0.1062*cos(q1)*cos(q3) - b1)**2)'])
print(ksi4[0])


# In[28]:


#obviously, there is no q5,q6 or q7 in any critical point's matrix, so
#the partial derivatives for these variables are 0

