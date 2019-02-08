from numpy import linalg as LA
import numpy as np


a = np.array([0,1,0,0,1,1,1,2,1,1,2,0]).reshape([4,3])
b = np.array([1,2,2,2,2,2,1,2,-1,2,2,3]).reshape([4,3])
c = np.array([-1,-1,-1,0,-1,-2,0,-1,1,-1,-2,1]).reshape([4,3])

test = np.array([0,0,0])

list = []


print("\nCLASS a:")
for t in a:
    print("Point", t, "L2:", str(LA.norm(t - test, 2)))
    x = ('a', LA.norm(t - test, 2))  # make a pair
    list.append(x)

print("\nCLASS b:")
for t in b:
    print("Point", t, "L2:", str(LA.norm(t - test, 2)))
    x = ('b', LA.norm(t - test, 2))  # make a pair
    list.append(x)

print("\nCLASS c:")
for t in c:
    print("Point", t, "L2:", str(LA.norm(t - test, 2)))
    x = ('c', LA.norm(t - test, 2))  # make a pair
    list.append(x)


def getKey(item):
    return item[1]

print(sorted(list, key=getKey))