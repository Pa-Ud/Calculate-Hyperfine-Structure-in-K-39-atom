# Calculate-Hyperfine-Structure-in-K-39-atom
#Creator: Mr Parinya Udommai
#Purpose: Calulate Hyperfine structure in 4(2S1/2), 4(2P1/2) and 4(2P3/2) fine levels in K39 atom.
#The Python code is below

import math
import numpy as np
import plotly as plt
import plotly.graph_objs as go
from plotly import tools
plt.offline.init_notebook_mode(connected=True)
from array import array
import csv
from scipy.optimize import curve_fit
import statistics as stat
c = 299792458
############################################################################
#Calculate matrix of Hyperfine structure
def delFnc(m,n):
    if m == n:
        out = 1.0
    else:
        out = 0.0
    return out
def Sqrt(K,mK,z): #z is +/-
    return np.sqrt(K*(K+1)-mK*(mK+z))
def SqrtX(K,mK,z,L,mL,y):
    return np.sqrt((K*(K+1)-mK*(mK+z)) * (L*(L+1)-mL*(mL+y)))
def SqrtXX(K,mK,z,L,mL,y,P,mP,x,Q,mQ,v):
    return np.sqrt((K*(K+1)-mK*(mK+z))*(L*(L+1)-mL*(mL+y))*(P*(P+1)-mP*(mP+x))*(Q*(Q+1)-mQ*(mQ+v)) )
def HFSelement1(J,I,mJ1,mI1,mJ2,mI2):
    term1 = SqrtX(J,mJ2,-1,I,mI2,+1)*delFnc(mJ1,mJ2-1)*delFnc(mI1,mI2+1)
    term2 = SqrtX(J,mJ2,1,I,mI2,-1)*delFnc(mJ1,mJ2+1)*delFnc(mI1,mI2-1)
    term3 = 2*mJ2*mI2*delFnc(mJ1,mJ2)*delFnc(mI1,mI2)
    return term1+term2+term3
def HFSelement2(J,I,mJ1,mI1,mJ2,mI2):
    term11 = 6*(mJ2*mI2)**2*delFnc(mJ1,mJ2)*delFnc(mI1,mI2)
    term12 = 6*(mJ2-1)*(mI2+1)*SqrtX(J,mJ2,-1,I,mI2,+1)*delFnc(mJ1,mJ2-1)*delFnc(mI1,mI2+1)
    term12 = term12+6*(mJ2+1)*(mI2-1)*SqrtX(J,mJ2,1,I,mI2,-1)*delFnc(mJ1,mJ2+1)*delFnc(mI1,mI2-1)
    term13 = 1.5*SqrtXX(J,mJ2,-1,J,mJ2-1,-1,I,mI2,1,I,mI2+1,1)*delFnc(mJ1,mJ2-2)*delFnc(mI1,mI2+2)
    term13 = term13+3*SqrtX(J,mJ2,1,I,mI2,-1)**2*delFnc(mJ1,mJ2)*delFnc(mI1,mI2)
    term13 = term13+1.5*SqrtXX(J,mJ2,1,J,mJ2+1,1,I,mI2,-1,I,mI2-1,-1)*delFnc(mJ1,mJ2+2)*delFnc(mI1,mI2-2)
    term15 = 1.5*SqrtX(J,mJ2,-1,I,mI2,1)*delFnc(mJ1,mJ2-1)*delFnc(mI1,mI2+1)
    term15 = term15+1.5*SqrtX(J,mJ2,1,I,mI2,-1)*delFnc(mJ1,mJ2+1)*delFnc(mI1,mI2-1)
    term15 = term15+1.5*2*mJ2*mI2*delFnc(mJ1,mJ2)*delFnc(mI1,mI2)
    term17 = 2*J*(J+1)*I*(I+1)*delFnc(mJ1,mJ2)*delFnc(mI1,mI2)
    return term11+term12+term13+term15+term17

def CalHFS(AJ,BJ,J,I):
    AJterm = 0.5*AJ
    if BJ != 0:
        BJterm = BJ/(2*I*(2*I-1)*2*J*(2*J-1))
    else:
        BJterm = 0
    mJ1,mJ2,mI1,mI2 = [],[],[],[]
    mJ1.append(J)
    mI1.append(I)
    while min(mJ1) > -J:
        mJ1.append(min(mJ1)-1)
    while min(mI1) > -I:
        mI1.append(min(mI1)-1) 
    mJ2,mI2 = mJ1, mI1
    HHFS = []
    for u in mJ1:
        for v in mI1:
            HHFS.append([])
            for x in mJ2:
                for y in mI2:
                    if BJ == 0:
                        element = AJterm*HFSelement1(J,I,u,v,x,y)
                    else:
                        element = AJterm*HFSelement1(J,I,u,v,x,y)+BJterm*HFSelement2(J,I,u,v,x,y)
                    HHFS[len(HHFS)-1].append(element)
    return HHFS
##########################################################################################################
# See https://www.tobiastiecke.nl/archive/PotassiumProperties.pdf for AJ and BJ values
#AJ,BJ,J,I = 27.775,0,1/2,3/2
AJ,BJ,J,I = 6.093,2.786,3/2,3/2
HHFS = CalHFS(AJ,BJ,J,I)
for i in range(0,len(HHFS)):
    print(HHFS[i])
Eigval = np.linalg.eig(HHFS)[0] -4.353125
print(Eigval)
