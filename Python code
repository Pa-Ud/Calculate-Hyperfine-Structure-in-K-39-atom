# Calculate-Hyperfine-Structure-in-K-39-atom
#Creator: Mr Parinya Udommai
#Purpose: Calulate Hyperfine structure in 4(2S1/2), 4(2P1/2) and 4(2P3/2) fine levels in K39 atom.
#The Python code is below

#################################################################################################
#39K's peaks
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

#################################################################################################
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
def HBfield(B,L,S,J,I,mJ1,mI1,mJ2,mI2): #Must give result in MHz unit
    h,uB,uN = 6.623e-34, 9.274078e-24, 5.050784e-27
    Bx = 1e-6*(B*1e-4)/h #coverint to MHz
    gJ = 1 + (J*(J+1)+S*(S+1)-L*(L+1)) / (2*J*(J+1))
    gI = -0.0014193489
    return (gJ*uB*mJ2 + gI*uN*mI2)*Bx*delFnc(mJ1,mJ2)*delFnc(mI1,mI2)
def CalHFS(AJ,BJ,J,I,B,L,S):
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
                        element = AJterm*HFSelement1(J,I,u,v,x,y)+HBfield(B,L,S,J,I,u,v,x,y)
                    else:
                        element = AJterm*HFSelement1(J,I,u,v,x,y)+BJterm*HFSelement2(J,I,u,v,x,y)+HBfield(B,L,S,J,I,u,v,x,y)
                    HHFS[len(HHFS)-1].append(element)
    return HHFS
    
 ######################################################################################################################
 offsetfreqArray = [0, 389.286058716e6, -4.353125+391.01617003e6]
AJar,BJar,Jar,Lar = [230.86,27.775,6.093], [0,0,2.786], [1/2,1/2,3/2], [0,1,1]
index = 0
offsetfreq = offsetfreqArray[index]
AJ,BJ,J,I = AJar[index], BJar[index], Jar[index], 3/2
L,S = Lar[index], 0.5
#for i in range(0,len(HHFS)):
    #print(HHFS[i])
#Eigval = np.linalg.eig(HHFS)[0] +offsetfreq #-4.353125
Bmin,Bmax,Bstep = 0,100.02,0.02
Ndata = int((Bmax-Bmin)/Bstep)
B = np.linspace(Bmin,Bmax,Ndata+1)
B = np.sort(B)[::-1]
Data, = [],
for i in range(0,len(B)):
    HHFS = CalHFS(AJ,BJ,J,I,B[i],L,S)
    Eigval = np.linalg.eig(HHFS)[0] + offsetfreq 
    Eigval = np.sort(Eigval)
    #Sort Data by curves    
    for j in range(0,len(Eigval)):
        if len(Data) < len(Eigval):
            Data.append([])
        else:
            Data[j].append(Eigval[j])
#find turn corner
limit = 0.05 #MHz
Slc = 5 #Range length used for Slope calculation
for l in range(0,len(B)):
    if B[l]>0.15 and B[l]<15.0: #Only consider this B range, where overlaps occur
        for k in range(2,len(Data)-4-1): #Consider curves 2 to len(Data)-4
            #check if any of these k curves cross
            if abs(Data[k][l]-Data[k+1][l])<limit:
                Slope0 = Data[k][l]-Data[k][l-Slc] #Original slope of C1
                Slope1 = Data[k][l+Slc]-Data[k][l]
                Slope2 = Data[k+1][l+Slc]-Data[k+1][l]
                if abs(Slope0-Slope2) < abs(Slope0-Slope1): #Swap data here
                    cut1 = Data[k][0:l]
                    cut2 = Data[k+1][0:l]
                    Data[k] = Data[k][l:len(B)]
                    Data[k] = np.insert(Data[k],0,cut2)
                    Data[k+1] = Data[k+1][l:len(B)]
                    Data[k+1] = np.insert(Data[k+1],0,cut1)
#Re-sort data ascendingly wrt B
for m in range(0,len(Data)):
    Bwaste, Data[m] = zip(*sorted(zip(B, Data[m])))
B.sort()
                                                                                       
#Plot data
figZ = go.Figure()
setFigureTextStyle = dict(family='Times New Roman', size=10, color='black')  
for i in range(0,len(Data)):
    figZ.add_trace(go.Scatter(
        #name
        x = B, y = Data[i],
        mode = 'lines', line = dict(width=2, #color='green'
                                   ),
        ))
figZ.update_layout(go.Layout(
    #title = '',
    width=600, height=600,
    plot_bgcolor = 'white',
    font = setFigureTextStyle,
    showlegend=False,
    xaxis=dict(
        title='B [G]',
        showgrid = True, gridcolor = 'rgb(240,240,240)', dtick = 10,
        linecolor = 'black', zeroline = True, zerolinecolor = 'black'
        ),
    yaxis=dict(
        title='Frequency [MHz]',
        showgrid = True, gridcolor = 'rgb(240,240,240)',
        #dtick = 50, #MHz
        linecolor = 'black',
            ),  ),
    )
figZ.show()

####################################################################################################
