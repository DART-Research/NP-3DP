import math
import numpy as np
list=[[73,26,67,9,18,0],
      [86,81,91,77,4,0],
      [84,29,49,68,0,22],
      [15,92,60,74,0,16],
      [36,0,91,87,65,32],
      [80,7,25,87,66,98],
      [72,13,92,77,12,87],
      [73,39,0,52,37,21],
      [53,12,0,22,83,75],
      [15,76,96,0,43,28],
      [63,77,28,0,25,27],
      [54,47,24,79,23,60],
      [0,87,66,56,96,36]]
W=[]
print(list)
for i in list:
    il=[]
    sum=0
    only=-1
    for ij,j in enumerate(i):
        if j==0:
            only=ij
            il.append(1)
        else:
            w=1/math.sinh(0.01*j)

            il.append(w)
        sum+=w
    il=[ili/sum for ili in il]
    if only>=0:
        for j,ili in enumerate(il):
            if j==only:
                il[j]=1
            else:
                il[j]=0

    print(il)
    W.append(il)

Wa=np.array(W)

offsetvalue=[[13,0],
             [0,23],
             [13,0],
             [13,0],
             [0,23],
             [0,23]]
print(offsetvalue)
Oa=np.array(offsetvalue)
Ova=np.dot(Wa,Oa)
print(Ova)

distancefield=[[275,300],
               [271,297],
               [281,287],
               [283,273],
               [256,233,],
               [236,223],
               [178,398],
               [0,280],
               [0,257],
               [257,0],
               [285,0],
               [201,130],
               [139,152,]

]
print([[256,233,],[139,152,]])
DA=np.array(distancefield)
print(distancefield)
OD=DA+Ova
print(OD)
UOD=(np.min(OD,axis=1))[:,np.newaxis]




distancefieldOlow=[[0,300],
                  [0,267],
                  [267,8],
                  [310,8],
                  [160,160],
                  [180,181],
                  [237,256],
                  [270,287],
                  [273,276],
                  [301,283],
                  [305,286],
                  [299,256],
                  [170,170]]

LDF=np.array(distancefieldOlow)
print(distancefieldOlow)
ULDF=(np.min(LDF,axis=1))[:,np.newaxis]
F=UOD/(UOD+ULDF)
print(F)

