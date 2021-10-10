"""
This program constructs the Kroger-Vink plot for Al2O3 at a given temperature.

The charge neutrality equation has the following terms: sum of (charge of defect*concentration of defect),
concentration of valence band holes and concentration of conduction band electrons.

Each term is a non-linear function of the Fermi level Ef, which ranges from 0 to Eg (bandgap).
The electronic defect concentrations are calculated using the Fermi-Dirac distribution and 
integrating over the density of states suing the trapezoidal method.

Thus, the bisection method is used to solve for Ef that achieves charge neutrality at each pO2. 

From Ef, we can calculate the equilibrium defect concentration at a given pO2.
Performing this over a range of pO2 gives us the Kroger-Vink diagram.
"""
import numpy as np
import math as m
import matplotlib.pyplot as plt
import pylab
"""
*****************TO FIND CONCENTRATION OF ELECTRONIC DEFECTS*******************
"""
"""
To find VBM and CBM row numbers
"""
# finds the VBM and returns its row number
def find_VBM(rows_DOS, E_VBM): 
    for i in range(0, rows_DOS): 
        if DOS[i,0] == E_VBM: #first column of DOS has the energy
            VBM_line = i
    return VBM_line
# finds the CBM and returns its row number
def find_CBM(rows_DOS, E_CBM): 
    for i in range(0, rows_DOS): 
        if DOS[i,0] == E_CBM: #first column of DOS has the energy
            CBM_line = i
    return CBM_line

"""
Trapezoidal method to find concentration of electronic defects by using the Fermi-Dirac distribution
and integrating over density of states
Ef is calculated assuming E_VBM is the reference level, therefore E_VBM term included in the F-D distribution
"""
# returns the concentration of valence band holes when Fermi level=Ef
def trapz_holes(VBM_line, DOS, Ef, E_VBM): 
    conc_holes = 0
    func = np.empty((VBM_line+1, 1), dtype=float) 
    for i in range(0, VBM_line+1): 
        func[i] = ((DOS[i,1] + DOS[i,2]) * m.exp(-(Ef + E_VBM - DOS[i,0])/(k * T)))/(1 + m.exp(-(Ef + E_VBM - DOS[i,0])/(k * T)))
    for i in range(0, VBM_line): 
        conc_holes = conc_holes + ((DOS[i+1,0] - DOS[i,0]) * (func[i+1] + func[i])) #trapezoidal method
    conc_holes = conc_holes/2
    return conc_holes
# returns the concentration of conduction band electrons when Fermi level=Ef
def trapz_electrons(rows_DOS, CBM_line, DOS, Ef, E_VBM): 
    CB = np.empty((rows_DOS - CBM_line, 5),dtype=float)
    conc_electrons = 0
    CB = DOS[CBM_line:(rows_DOS), 0:6]
    func = np.empty((CB[:,0].size, 1), dtype=float)
    for i in range(0, CB[:,0].size): 
        func[i] = (CB[i,1] + CB[i,2])/(1 + m.exp((CB[i,0] - Ef - E_VBM)/(k * T)))
    for i in range(0, CB[:,0].size-1): 
        conc_electrons = conc_electrons + ((CB[i+1,0] - CB[i,0]) * (func[i+1] + func[i])) #trapezoidal method
    conc_electrons = conc_electrons/2
    return conc_electrons
"""
****************TO FIND DEFECT CONCENTRATIONS AT A GIVEN pO2*******************
"""
"""
To find chemical potentials
"""
# returning chemical potentials of O and Al over pO2 range: 1e-45 atm to 1 atm
# uAl=EDFT_Al @ pO2=6.5e-42 atm, hence this limit
def O_chempot(DFTE_O2, E_over, u0_O2, k, T, p0):
    p = np.empty((46,1), dtype=float)
    u_O = np.empty((46,1), dtype=float) 
    for i in range(0,46): 
        p[i] = m.pow(10,-(45-i))
        u_O[i] = (1/2) * (DFTE_O2 + E_over + u0_O2 + (k * T * m.log(p[i]/p0)))
    return u_O
def Al_chempot(DFTE_Al2O3, u_O):
    u_Al = np.empty((46,1),dtype=float) 
    for i in range(0,46): 
        u_Al[i] = (DFTE_Al2O3 - (3 * u_O[i]))/2
    return u_Al   
"""
To find defect formation energies at a given pO2
"""
# returns vector of formation energies of various defects at a given pO2, Fermi level Ef
def formn_E(KV, E_perf, u_O, u_Al, E_VBM, Ef):
    delE = np.empty((14,1),dtype=float)
    for i in range(0,14): #change!
        delE[i] = KV[i,4] - E_perf - (KV[i,2] * u_O) - (KV[i,3] * u_Al) + (KV[i,1] * (E_VBM + Ef)) + KV[i,6] 
    return delE
"""
To find defect concentrations at a given pO2 using Boltzmann approx
"""
# returns concentration of oxygen vacancies (all charge states)
def conc_VO(KV, delE, k, T):
    nD = KV[0:3,[5]] #N_site*N_config
    delEVO = delE[0:3] #copying formnE of VO from delE
    VO = np.empty((3,1),dtype=float)
    for i in range (0,3): 
        VO[i] = nD[i] * np.exp(-delEVO[i]/(k * T))
    return VO 
# returns concentration of aluminum vacancies (all charge states)
def conc_VAl(KV, delE, k, T):
    nD = KV[3:7,[5]] #N_site*N_config
    delEVAl = delE[3:7] #copying formnE of VAl from delE
    VAl = np.empty((4,1),dtype=float)
    for i in range (0,4): 
        VAl[i] = nD[i] * np.exp(-delEVAl[i]/(k * T))
    return VAl
# returns concentration of oxygen interstitials (all charge states)
def conc_IO(KV, delE, k, T):
    nD = KV[7:10,[5]] #N_site*N_config
    delEIO = delE[7:10] #copying formnE of IO from delE
    IO = np.empty((3,1),dtype=float)
    for i in range (0,3): 
        IO[i] = nD[i] * np.exp(-delEIO[i]/(k * T))
    return IO 
# returns concentration of aluminum interstitials (all charge states)
def conc_IAl(KV, delE, k, T):
    nD = KV[10:14,[5]] #N_site*N_config
    delEIAl = delE[10:14] #copying formnE of IAl from delE
    IAl = np.empty((4,1),dtype=float)
    for i in range (0,4): 
        IAl[i] = nD[i] * np.exp(-delEIAl[i]/(k * T))
    return IAl
"""
**************************BISECTION METHOD*************************************
"""
def samesign(fa, fb):
    return fa*fb>0
# calculates the value of the charge-neutrality function at given pO2 by 
# calculating the electronic and ionic defect concentrations at Ef 
def neut_fn(VBM_line, DOS, Ef, E_VBM, rows_DOS, CBM_line, KV, E_perf, u_O, u_Al, k, T):
    conc_holes = trapz_holes(VBM_line, DOS, Ef, E_VBM) 
    conc_electrons = trapz_electrons(rows_DOS, CBM_line, DOS, Ef, E_VBM) 
    delE = formn_E(KV, E_perf, u_O, u_Al, E_VBM, Ef) 
    VO = conc_VO(KV, delE, k, T) 
    VAl = conc_VAl(KV, delE, k, T) 
    IO = conc_IO(KV, delE, k, T) 
    IAl = conc_IAl(KV, delE, k, T) 
    q_VO = KV[0:3,[1]] 
    q_VAl = KV[3:7,[1]] 
    q_IO = KV[7:10,[1]] 
    q_IAl = KV[10:14,[1]] 
    qVO = np.empty((3,1),dtype=float) 
    qVAl = np.empty((4,1),dtype=float) 
    qIO = np.empty((3,1),dtype=float) 
    qIAl = np.empty((4,1),dtype=float) 
    for i in range(0,3):
        qVO[i] = q_VO[i] * VO[i]
    for i in range(0,4):
        qVAl[i] = q_VAl[i] * VAl[i]
    for i in range(0,3):
        qIO[i] = q_IO[i] * IO[i]
    for i in range(0,4):
        qIAl[i] = q_IAl[i] * IAl[i]            
    sum_qD = sum(qVO) + sum(qVAl) + sum(qIO) + sum(qIAl)
    f = sum_qD + conc_holes - conc_electrons
    return f
# this function calculates the Ef that solves the neutrality equation at a given pO2 
def bisection(VBM_line, DOS, Efa, Efb, E_VBM, rows_DOS, CBM_line, KV, E_perf, u_O, u_Al, k, T):
    fa = neut_fn(VBM_line, DOS, Efa, E_VBM, rows_DOS, CBM_line, KV, E_perf, u_O, u_Al, k, T)
    fb = neut_fn(VBM_line, DOS, Efb, E_VBM, rows_DOS, CBM_line, KV, E_perf, u_O, u_Al, k, T)
    assert not samesign(fa,fb)
    while Efb - Efa > 1e-10:
        Efc = (Efa + Efb)/2
        fc = neut_fn(VBM_line, DOS, Efc, E_VBM, rows_DOS, CBM_line, KV, E_perf, u_O, u_Al, k, T)
        if samesign(fa,fc):
            Efa = Efc
        elif samesign(fb,fc):
            Efb = Efc
    return Efc  
"""
************************FINAL Ef AND DEFECT CONCENTRATIONS*********************
"""
# this function uses bisection method to calculate f,Ef and [D] at various pO2 
def final_f_Ef_conc(VBM_line, DOS, Efa, Efb, E_VBM, rows_DOS, CBM_line, KV, E_perf, u_O, u_Al, k, T):
    f = np.empty((46,1),dtype=float) # to check value of neutrality equation using final Ef
    Ef = np.empty((46,1),dtype=float)
    delE = np.empty((14,46),dtype=float)
    final_h = np.empty((46,1),dtype=float)
    final_e = np.empty((46,1),dtype=float)
    final_VO = np.empty((3,46),dtype=float)
    final_VAl = np.empty((4,46),dtype=float)
    final_IO = np.empty((3,46),dtype=float)
    final_IAl = np.empty((4,46),dtype=float)
    for i in range(0,46):
        Ef[i] = bisection(VBM_line, DOS, Efa, Efb, E_VBM, rows_DOS, CBM_line, KV, E_perf, u_O[i], u_Al[i], k, T)
        f[i] = neut_fn(VBM_line, DOS, Ef[i], E_VBM, rows_DOS, CBM_line, KV, E_perf, u_O[i], u_Al[i], k, T)
        delE[:,[i]] = formn_E(KV, E_perf, u_O[i], u_Al[i], E_VBM, Ef[i])
        final_h[i] = trapz_holes(VBM_line, DOS, Ef[i], E_VBM)
        final_e[i] = trapz_electrons(rows_DOS, CBM_line, DOS, Ef[i], E_VBM)
        final_VO[:,[i]] = conc_VO(KV, delE[:,[i]], k, T)
        final_VAl[:,[i]] = conc_VAl(KV, delE[:,[i]], k, T)
        final_IO[:,[i]] = conc_IO(KV, delE[:,[i]], k, T)
        final_IAl[:,[i]] = conc_IAl(KV, delE[:,[i]], k, T)
    return (final_VO,final_VAl,final_IO,final_IAl,final_h,final_e,f,Ef,delE)
    
"""
****************************MAIN BODY******************************************
"""
DOS = np.loadtxt("dosperfect.txt") # DOS from VASP
DOS[:,1] = DOS[:,1]/24 # spin-up DOS per formula unit of Al2O3
DOS[:,2] = DOS[:,2]/24 # spin-down DOS per formula unit of Al2O3
KV = np.loadtxt("native_data.txt",skiprows=1) # array of the defect supercell data
# defining constants
k = 8.6173303e-05 # Boltzmann constant in eV
E_VBM = 5.795 # Valence band maximum in eV from VASP DOS
E_CBM = 11.638 # Conduction band minimum in eV from VASP DOS
rows_DOS = len(DOS) # number of rows in the array DOS
# Ef ranges from 0 to Eg in eV
Efa = 0
Efb = 5.84
VBM_line = find_VBM(rows_DOS, E_VBM)
CBM_line = find_CBM(rows_DOS, E_CBM)
# defining constants to calculate chemical potentials 
DFTE_Al2O3 = -37.4057 # DFT energy of Al2O3(s) per f.u. in eV
DFTE_O2 = -9.8591 # DFT energy of O2(g) in eV
E_perf = -897.74454 # DFT energy in eV of perfect supercell of Al2O3
E_over = 1.36 # O2 GGA overbinding correction
u0_O2 = -2.4534 # at 1100 K (Temperature correction to u_O)
p0 = 1 # standard pressure in atm
T = 1100 # temperature at which KV diagram plotted in K
# calculating chemical potentials
u_O = O_chempot(DFTE_O2, E_over, u0_O2, p0)
u_Al = Al_chempot(DFTE_Al2O3, u_O)
# calculating all final defect concentrations at all pO2 by using bisection method to solve
# charge-neutrality equation  
func = final_f_Ef_conc(VBM_line, DOS, Efa, Efb, E_VBM, rows_DOS, CBM_line, KV, E_perf, u_O, u_Al, k, T)
finalVO = func[0]
finalVAl = func[1]
finalIO = func[2]
finalIAl = func[3]
finalh = func[4]
finale = func[5]
f = func[6]
Ef = func[7]
delE = func[8]
p = np.array([[1e-45,1e-44,1e-43,1e-42,1e-41,1e-40,1e-39,
             1e-38,1e-37,1e-36,1e-35,1e-34,1e-33,1e-32,1e-31,1e-30,1e-29,1e-28,
             1e-27,1e-26,1e-25,1e-24,1e-23,1e-22,1e-21,1e-20,1e-19,1e-18,1e-17,
             1e-16,1e-15,1e-14,1e-13,1e-12,1e-11,1e-10,1e-09,1e-08,1e-07,1e-06,
             1e-05,1e-04,1e-03,1e-02,1e-01,1]])
logp = np.log10(p)
logVO = np.log10(finalVO)
logVAl = np.log10(finalVAl)
logIO = np.log10(finalIO)
logIAl = np.log10(finalIAl)
logh = np.log10(finalh)
loge = np.log10(finale)
#plt.figure(figsize=(8,6))
#plt.figure(figsize=(3.84,2.95))
axes = plt.gca()
axes.set_xlim([-45,0])
axes.set_ylim([-16,-4]) 
font = {'fontname':'Times New Roman','fontsize':7}
plt.xlabel('logpO2 (atm)',**font)
plt.ylabel('log[D] per f.u. Al2O3',**font)
plt.plot(logp[0,:],logVO[0,:],color='#1b9e77',marker='.',label='VOx')
plt.plot(logp[0,:],logVO[1,:],color='#1b9e77',marker='*',label='VO+1')
plt.plot(logp[0,:],logVO[2,:],color='#1b9e77',marker='p',label='VO+2')
plt.plot(logp[0,:],logVAl[0,:],color='#d95f02',marker='P',label='VAlx')
plt.plot(logp[0,:],logVAl[1,:],color='#d95f02',marker='+',label='VAl-1')
plt.plot(logp[0,:],logVAl[2,:],color='#d95f02',marker='x',label='VAl-2')
plt.plot(logp[0,:],logVAl[3,:],color='#d95f02',marker='o',label='VAl-3')
plt.plot(logp[0,:],logIO[0,:],color='#7570b3',marker='X',label='IOx')
plt.plot(logp[0,:],logIO[1,:],color='#7570b3',marker='h',label='IO-1')
plt.plot(logp[0,:],logIO[2,:],color='#7570b3',marker='H',label='IO-2')
plt.plot(logp[0,:],logIAl[0,:],color='#e7298a',marker='d',label='IAlx')
plt.plot(logp[0,:],logIAl[1,:],color='#e7298a',marker='D',label='IAl+1')
plt.plot(logp[0,:],logIAl[2,:],color='#e7298a',marker='s',label='IAl+2')
plt.plot(logp[0,:],logIAl[3,:],color='#e7298a',marker='^',label='IAl+3')
plt.plot(logp[0,:],logh[:,0],color='#66a61e',marker='>',label='h')
plt.plot(logp[0,:],loge[:,0],color='#e6ab02',marker='<',label='e')
pylab.legend(loc='upper left')
pylab.savefig('native_1100K.png')
