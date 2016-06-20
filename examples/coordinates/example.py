#!/usr/bin/env python3
import pyclamster
import numpy as np

x = pyclamster.x([1,2,3,4])
y = pyclamster.y([[5,6],[7,8]])
z = pyclamster.z([[[2,4]],[[6,8]]])

set = pyclamster.DependantQuantitySet(x,y,z)

print("INITIAL set")
print(set)

var2 = pyclamster.DependantQuantity([1,2,3])

try:
    set.addquantity(var2)
except:
    print("could not add quantity!")
    

print("AFTER trying to add new invalid shape quantity")
print(set)

var3 = pyclamster.DependantQuantity(np.ma.masked_array([11,22,33,44]),
    name="var3")

try:
    set.addquantity(var3)
except:
    print("could not add quantity!")
    

print("AFTER trying to add new valid quantity")
print(set)

set.shape = None
print("AFTER setting shape to None")
print(set)

set.shape = (2,2)
print("AFTER setting shape to new value")
print(set)
