import numpy as np
import math as m
import matplotlib.pyplot as plt
from mpmath import * # for adjusting precision
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
****************TO FIND DEFECT CONCENTRATIONS AT A GIVEN pH2*******************
"""
"""
To find chemical potentials
"""
# over pH2=1e-15 to 1 atm
# including even lower values of pH2 will not make a difference, since T is so low
def H_chempot(lim, DFTE_H2, u0_H2, k, T, p0):
    pH2 = np.empty((lim,1),dtype=float) 
    u_H = np.empty((lim,1),dtype=float) 
    for i in range(0,lim): 
        pH2[i] = m.pow(10,-(lim-1-i))
        u_H[i] = (1/2) * (DFTE_H2 + u0_H2 + (k * T * m.log(pH2[i]/p0))) 
    return u_H
"""
To find defect formation energies at a given pH2
"""
# formation E of isolated H defects; order Hi-1,Hi0,Hi+1
def formn_E(KV_Hi, E_perf, u_H, E_VBM, Ef):
    delE=np.empty((3,1),dtype=float) 
    for i in range(0,3):  
        delE[i]=KV_Hi[i,5] - E_perf - (KV_Hi[i,4]*u_H) + (KV_Hi[i,1] * (E_VBM+Ef)) + KV_Hi[i,7]     
    return delE
# reading fixed native defect concentrations, at each charge state
def conc_VO_fix(fixednative):
    VO_fix = fixednative[0:3,[5]] # q=0,1,2
    return VO_fix
def conc_VAl_fix(fixednative):
    VAl_fix = fixednative[3:7,[5]] # q=0,-1,-2,-3
    return VAl_fix
def conc_IO_fix(fixednative):
    IO = fixednative[7:10,[5]] # q=0,-1,-2
    return IO
def conc_IAl_fix(fixednative):
    IAl = fixednative[10:14,[5]] # q=0,1,2,3
    return IAl
# calculating concentrations of isolated H defects using Boltzmann approx.
def conc_IH_eq(KV_Hi, delE, k, T):
    nD = KV_Hi[0:3,[6]] #N_site*N_config
    IH_eq = np.empty((3,1),dtype=float)
    for i in range(0,3):
        IH_eq[i] = nD[i] * np.exp(-delE[i]/(k * T)) # q=-1,0,1
    return IH_eq 
# calculating concentrations of H defect complexes from fixed native defect conc
# and equilibrium isolated H defect conc
# only those complexes w/ positive Eb and realizable charge considered
# Supplementary Info has detailed derivation
def conc_VOH(delE, BE, fixednative, KV_Hi, VO_fix, k, T):
    IH_eq = conc_IH_eq(KV_Hi, delE, k, T) #3x1 q=-1,0,1
    VOH = np.empty((2,1),dtype=float) #q=0,1
    Eb = BE[0:2,[11]] #0,1
    Nconf_VOH = BE[0:2,[10]] #0,1
    Nconf_VO = fixednative[0:3,[4]] #q=0,1,2
    nD_Hi = KV_Hi[0:3,[6]] #-1,0,1
    # first calculating all the configuration-related prefactors
    X0 = Nconf_VOH[0]/(Nconf_VO[1] * nD_Hi[0]) #VOH0=VO+1,Hi-1
    X1 = Nconf_VOH[1]/(Nconf_VO[0] * nD_Hi[2]) #VOH+1=VO0,Hi+1
    IH_eq0 = IH_eq[0]; IH_eq2 = IH_eq[2]; Eb0 = Eb[0]; Eb1 = Eb[1]
    mp.dps = 250
    IH_eq0 = mpf(IH_eq0[0]); IH_eq2 = mpf(IH_eq2[0]); Eb0 = mpf(Eb0[0]); Eb1 = mpf(Eb1[0]);
    VOH[0] = (VO_fix[1] * IH_eq0)/(IH_eq0 + ((1/X0) * m.exp(-Eb0/(k * T)))) #VOH0=VO+1,Hi-1
    VOH[1] = (VO_fix[0] * IH_eq2)/(IH_eq2 + ((1/X1) * m.exp(-Eb1/(k * T)))) #VOHp1=VO0,Hi+1
    return VOH
def conc_VAlH(delE, BE, fixednative, KV_Hi, VAl_fix, k, T):
    IH_eq = conc_IH_eq(KV_Hi, delE, k, T) #3x1 q=-1,0,1
    VAlH = np.empty((9,1),dtype=float) #q=-3,-2,-1,0,1; -1,0,1,2->VAlxH
    Eb = BE[2:11,[11]] #q=-3,-2,-1,0,1; -1,0,1,2->VAlxH
    Nconf_VAlH = BE[2:11,[10]] #q=-3,-2,-1,0,1; -1,0,1,2->VAlxH
    Nconf_VAl = fixednative[3:7,[4]] #q=0,-1,-2,-3
    nD_Hi = KV_Hi[0:3,[6]] #q=-1,0,+1
    # first calculating all the configuration-related prefactors
    X0 = Nconf_VAlH[0]/(Nconf_VAl[2] * nD_Hi[0]) #VAlHm3=VAl-2,Hi-1
    X1 = Nconf_VAlH[1]/(Nconf_VAl[3] * nD_Hi[2]) #VAlHm2=VAl-3,Hi+1
    X2 = Nconf_VAlH[2]/(Nconf_VAl[2] * nD_Hi[2]) #VAlHm1=VAl-2,Hi+1
    X3 = Nconf_VAlH[3]/(Nconf_VAl[1] * nD_Hi[2]) #VAlHm0=VAl-1,Hi+1
    X4 = Nconf_VAlH[4]/(Nconf_VAl[0] * nD_Hi[2]) #VAlHp1=VAl0,Hi+1   
    X5 = Nconf_VAlH[5]/(Nconf_VAl[3] * (nD_Hi[2] ** 2)) #VAl2Hm1=VAl-3,2*Hi+1
    X6 = Nconf_VAlH[6]/(Nconf_VAl[3] * (nD_Hi[2] ** 3)) #VAl3Hx=VAl-3,3*Hi+1
    X7 = Nconf_VAlH[7]/(Nconf_VAl[3] * (nD_Hi[2] ** 4)) #VAl4Hp1=VAl-3,4*Hi+1
    X8 = Nconf_VAlH[8]/(Nconf_VAl[3] * (nD_Hi[2] ** 5)) #VAl5Hp2=VAl-3,5*Hi+1
    # calculating the sum of NVAlHm2,NVAl2Hm1,NVAl3Hx,NVAl4Hp1,NVAl5Hp2 
    mult1 = IH_eq[2] * X1 * m.exp(Eb[1]/(k*T))
    mult5 = (IH_eq[2] ** 2) * X5 * m.exp(Eb[5]/(k * T))
    mult6 = (IH_eq[2] ** 3) * X6 * m.exp(Eb[6]/(k * T))
    mult7 = (IH_eq[2] ** 4) * X7 * m.exp(Eb[7]/(k * T))
    mult8 = (IH_eq[2] ** 5) * X8 * m.exp(Eb[8]/(k * T))
    mp.dps = 250 #setting precision to 250 decimal points
    mult1 = mpf(mult1[0]); mult5 = mpf(mult5[0]); mult6 = mpf(mult6[0])
    mult7 = mpf(mult7[0]); mult8 = mpf(mult8[0]) #changing the precision
    summed=VAl_fix[3] * (mult1 + mult5 + mult6 + mult7 + mult8)/(1 + (mult1 + mult5 + mult6 + mult7 + mult8))
    # finding concentrations of VAlHm2,VAl2Hm1,VAl3Hx,VAl4Hp1,VAl5Hp2 analytically
    VAlH[1] = (VAl_fix[3] - summed) * IH_eq[2] * X1 * m.exp(Eb[1]/(k * T)) #VAlHm2=VAl-3,Hi+1
    VAlH[5] = (VAl_fix[3] - summed) * (IH_eq[2] ** 2) * X5 * m.exp(Eb[5]/(k * T)) #VAl2Hm1=VAl-3,2*Hi+1
    VAlH[6] = (VAl_fix[3] - summed) * (IH_eq[2] ** 3) * X6 * m.exp(Eb[6]/(k * T)) #VAl3Hx=VAl-3,3*Hi+1
    VAlH[7] = (VAl_fix[3] - summed) * (IH_eq[2] ** 4) * X7 * m.exp(Eb[7]/(k * T)) #VAl4Hp1=VAl-3,4*Hi+1
    VAlH[8] = (VAl_fix[3] - summed) * (IH_eq[2] ** 5) * X8 * m.exp(Eb[8]/(k * T)) #VAl5Hp2=VAl-3,5*Hi+1
    # next, defect complexes whose VAlq is not shared
    IH_eq2 = IH_eq[2]; Eb3 = Eb[3]; Eb4 = Eb[4]
    IH_eq2 = mpf(IH_eq2[0]); Eb3 = mpf(Eb3[0]); Eb4 = mpf(Eb4[0])
    VAlH[3] = (VAl_fix[1] * IH_eq2)/(IH_eq2 + ((1/X3) * m.exp(-Eb3/(k * T)))) #VAlH0=VAl-1,Hi+1
    VAlH[4] = (VAl_fix[0] * IH_eq2)/(IH_eq2 + ((1/X4) * m.exp(-Eb4/(k * T)))) #VAlHp1=VAl0,Hi+1  
    # calculating the sum of NVAlHm3,NVAlHm1 
    mult0 = IH_eq[0] * X0 * m.exp(Eb[0]/(k * T))
    mult2 = IH_eq[2] * X2 * m.exp(Eb[2]/(k * T))
    mult0 = mpf(mult0[0]); mult2 = mpf(mult2[0])
    summed2 = VAl_fix[2] * (mult0 + mult2)/(1 + (mult0 + mult2))
    # finding concentrations of VAlHm3,VAlHm1 analytically
    VAlH[0] = (VAl_fix[2] - summed2) * IH_eq[0] * X0 * m.exp(Eb[0]/(k * T)) #VAlHm3=VAl-2,Hi-1
    VAlH[2] = (VAl_fix[2] - summed2) * IH_eq[2] * X2 * m.exp(Eb[2]/(k * T)) #VAlHm1=VAl-2,Hi+1
    return VAlH
"""
**************************BISECTION METHOD*************************************
"""
def samesign(fa,fb):
    return fa * fb > 0
# calculates charge density at each u_H and fixed Ef
def neut_fn(VBM_line, DOS, Ef, E_VBM, rows_DOS, CBM_line, KV_Hi, E_perf, u_H, k, T, fixednative, KV_iso, KV_complex):
    conc_holes = trapz_holes(VBM_line, DOS, Ef, E_VBM) 
    conc_electrons = trapz_electrons(rows_DOS, CBM_line, DOS, Ef, E_VBM) 
    delE = formn_E(KV_Hi, E_perf, u_H, E_VBM, Ef) #3x1
    # fixed concentrations of native defects
    VO_fix = conc_VO_fix(fixednative) #3x1
    VAl_fix = conc_VAl_fix(fixednative) #4x1
    IO_fix = conc_IO_fix(fixednative) #3x1
    IAl_fix = conc_IAl_fix(fixednative) #4x1
    # equilibrium concentrations of H defects
    IH_eq = conc_IH_eq(KV_Hi, delE, k, T) #3x1
    # concentrations of H complexes
    VOH = conc_VOH(delE, BE, fixednative, KV_Hi, VO_fix, k, T) #2x1
    VAlH = conc_VAlH(delE, BE, fixednative, KV_Hi, VAl_fix, k, T) #9x1
    # charges
    q_VO = KV_iso[0:3,1] #3x1
    q_VAl = KV_iso[3:7,1] #4x1
    q_IO = KV_iso[7:10,1] #3x1
    q_IAl = KV_iso[10:14,1] #4x1
    q_IH = KV_iso[14:17,1] #3x1
    q_VOH = KV_complex[1:3,1] #2x1 #0,1
    q_VAlH = KV_complex[6:15,1] #9x1 #-3,-2,-1,0,1; -1,0,1,2->VAlxH
    VO = np.empty((3,1),dtype=float) #3x1
    VAl = np.empty((4,1),dtype=float) #4x1
    IO = np.empty((3,1),dtype=float) #3x1
    IAl = np.empty((4,1),dtype=float) #4x1    
    qVO = np.empty((3,1),dtype=float) #3x1
    qVAl = np.empty((4,1),dtype=float) #4x1
    qIO = np.empty((3,1),dtype=float) #3x1
    qIAl = np.empty((4,1),dtype=float) #4x1
    qIH_eq = np.empty((3,1),dtype=float) #3x1
    qVOH = np.empty((2,1),dtype=float) #2x1
    qVAlH = np.empty((9,1),dtype=float) #9x1
    # current concentrations; ordering: 0,1,2; 0,-1,-2,-3
    VO[0] = VO_fix[0] - VOH[1]
    VO[1] = VO_fix[1] - VOH[0]
    VO[2] = VO_fix[2]
    VAl[0] = VAl_fix[0] - VAlH[4]
    VAl[1] = VAl_fix[1] - VAlH[3]
    VAl[2] = VAl_fix[2] - VAlH[2] - VAlH[0]
    VAl[3] = VAl_fix[3] - VAlH[1] - VAlH[5] - VAlH[6] - VAlH[7] - VAlH[8]    
    IO = IO_fix
    IAl = IAl_fix
    # charge*respective conc
    for i in range(0,3):
        qVO[i] = q_VO[i] * VO[i]
    for i in range(0,4):
        qVAl[i] = q_VAl[i] * VAl[i]
    for i in range(0,3):
        qIO[i] = q_IO[i] * IO[i]
    for i in range(0,4):
        qIAl[i] = q_IAl[i] * IAl[i]
    for i in range(0,3):
        qIH_eq[i] = q_IH[i] * IH_eq[i]
    for i in range(0,2):
        qVOH[i] = q_VOH[i] * VOH[i]            
    for i in range(0,9):
        qVAlH[i] = q_VAlH[i] * VAlH[i]
    sum_qD = sum(qVO) + sum(qVAl) + sum(qIO) + sum(qIAl) + sum(qIH_eq) + sum(qVOH) + sum(qVAlH)
    f = sum_qD + conc_holes - conc_electrons    
    return f
# to solve for Ef that achieves charge neutrality
def bisection(VBM_line, DOS, Efa, Efb, E_VBM, rows_DOS, CBM_line, KV_Hi, E_perf, u_H, k, T, fixednative, KV_iso, KV_complex):
    fa = neut_fn(VBM_line, DOS, Efa, E_VBM, rows_DOS, CBM_line, KV_Hi, E_perf, u_H, k, T, fixednative, KV_iso, KV_complex)
    #print(fa)
    fb = neut_fn(VBM_line, DOS, Efb, E_VBM, rows_DOS, CBM_line, KV_Hi, E_perf, u_H, k, T, fixednative, KV_iso, KV_complex)
    #print(fb)
    assert not samesign(fa,fb)
    while Efb - Efa > 1e-14:
        Efc = (Efa + Efb)/2
        fc = neut_fn(VBM_line, DOS, Efc, E_VBM, rows_DOS, CBM_line, KV_Hi, E_perf, u_H, k, T, fixednative, KV_iso, KV_complex)
        if samesign(fa,fc):
            Efa = Efc
        elif samesign(fb,fc):
            Efb = Efc
    return Efc
# evaluates everything across the pH2 range
def final_f_Ef_conc(lim, VBM_line, DOS, Efa, Efb, E_VBM, rows_DOS, CBM_line, KV_Hi, E_perf, u_H, k, T, fixednative, KV_iso, KV_complex):
    VO_fix = conc_VO_fix(fixednative) #3x1
    VAl_fix = conc_VAl_fix(fixednative) #4x1
    IO_fix = conc_IO_fix(fixednative) #3x1
    IAl_fix = conc_IAl_fix(fixednative) #3x1
    f = np.empty((lim,1),dtype=float) #to check value of neutrality equation using final Ef
    Ef = np.empty((lim,1),dtype=float)
    delE = np.empty((3,lim),dtype=float)
    final_h = np.empty((lim,1),dtype=float)
    final_e = np.empty((lim,1),dtype=float)
    final_VO = np.empty((3,lim),dtype=float) #current VO
    final_VAl = np.empty((4,lim),dtype=float) #current VAl
    final_IO = np.empty((3,lim),dtype=float)
    final_IAl = np.empty((4,lim),dtype=float)
    final_IH = np.empty((3,lim),dtype=float)
    final_VOH = np.empty((2,lim),dtype=float)    
    final_VAlH = np.empty((9,lim),dtype=float)
    for i in range(0,lim):
        #print(u_H[i])
        Ef[i] = bisection(VBM_line, DOS, Efa, Efb, E_VBM, rows_DOS, CBM_line, KV_Hi, E_perf, u_H[i], k, T, fixednative, KV_iso, KV_complex)
        #print(Ef[i])
        delE[:,[i]] = formn_E(KV_Hi, E_perf, u_H[i], E_VBM, Ef[i])
        f[i] = neut_fn(VBM_line, DOS, Ef[i], E_VBM, rows_DOS, CBM_line, KV_Hi, E_perf, u_H[i], k, T, fixednative, KV_iso, KV_complex)
        final_h[i] = trapz_holes(VBM_line, DOS, Ef[i], E_VBM)
        final_e[i] = trapz_electrons(rows_DOS, CBM_line, DOS, Ef[i], E_VBM)
        final_IH[:,[i]] = conc_IH_eq(KV_Hi, delE[:,[i]], k, T)
        final_VAlH[:,[i]] = conc_VAlH(delE[:,[i]], BE, fixednative, KV_Hi, VAl_fix, k, T)
        final_VOH[:,[i]] = conc_VOH(delE[:,[i]], BE, fixednative, KV_Hi, VO_fix, k, T)
        final_VO[0,[i]] = VO_fix[0] - final_VOH[1,i]
        final_VO[1,[i]] = VO_fix[1] - final_VOH[0,i]
        final_VO[2,[i]] = VO_fix[2]
        final_VAl[0,[i]] = VAl_fix[0] - final_VAlH[4,i]
        final_VAl[1,[i]] = VAl_fix[1] - final_VAlH[3,i]
        final_VAl[2,[i]] = VAl_fix[2] - final_VAlH[2,i] - final_VAlH[0,i]
        final_VAl[3,[i]] = VAl_fix[3] - final_VAlH[1,i] - final_VAlH[5,i] - final_VAlH[6,i] - final_VAlH[7,i] - final_VAlH[8,i]
        final_IO[:,[i]] = IO_fix
        final_IAl[:,[i]] = IAl_fix
    return (Ef,delE,f,final_h,final_e,final_IH,final_VAlH,final_VOH,final_VO,final_VAl,final_IO,final_IAl)
"""
****************************MAIN BODY******************************************
"""
DOS = np.loadtxt("dosperfect.txt") # DOS from VASP
DOS[:,1] = DOS[:,1]/24 # spin-up DOS per formula unit of Al2O3
DOS[:,2] = DOS[:,2]/24 # spin-down DOS per formula unit of Al2O3
# n,charge,num_O,num_Al,num_H,E_defect,NsiteNconfig,E_MP
KV_Hi = np.loadtxt("onlyHi.txt",skiprows=1) # 3x8, only Hi DFT details
# n,charge,num_O,num_Al,num_H,num_Mg,num_Fe,num_Ti,num_Cr,num_Si,E_defect,num_site,E_MP
KV_iso = np.loadtxt("isolated.txt",skiprows=1) # 30x13, all defect details
# n,charge,num_O,num_Al,num_H,num_Mg,num_Fe,num_Ti,num_Cr,num_Si,E_defect,num_site,E_MP
KV_complex = np.loadtxt("complex.txt",skiprows=1) # 36x13, all defect details
# n,charge,num_O,num_Al,Nconf,concentration
fixednative = np.loadtxt("fixed_native_conc.txt",skiprows=1) # 14x6, conc of native defects from 1100K
# n,charge,num_O,num_Al,num_H,num_Mg,num_Fe,num_Ti,num_Cr,num_Si,Nconf,BE
BE = np.loadtxt("binding_energies.txt",skiprows=1) # binding energies 11x12
# defining constants
k = 8.6173303e-05 # Boltzmann constant in eV
E_VBM = 5.795 # Valence band maximum in eV
E_CBM = 11.638 # Conduction band minimum in eV
rows_DOS = len(DOS) # number of rows in the array DOS
# Ef ranges from 0 to Eg in eV
Efa=0
Efb=5.84
VBM_line = find_VBM(rows_DOS, E_VBM)
CBM_line = find_CBM(rows_DOS, E_CBM)
# defining constants to calculate chemical potentials 
DFTE_H2 = -6.771 # DFT energy of H2 molecule in eV
E_perf = -897.74454 # DFT energy in eV of perfect supercell of Al2O3
u0_H2 = -0.31856664 # at 300 K (Temperature correction to u_H)
p0 = 1 # standard pressure in atm
T = 300 # K
lim = 16
u_H = H_chempot(lim, DFTE_H2, u0_H2, k, T, p0)
# calculating all final defect concentrations at all pH2 by using bisection method to solve
# charge-neutrality equation  
func = final_f_Ef_conc(lim, VBM_line, DOS, Efa, Efb, E_VBM, rows_DOS, CBM_line, KV_Hi, E_perf, u_H, k, T, fixednative, KV_iso, KV_complex)
Ef = func[0]
delE = func[1]
f = func[2]
finalh = func[3]
finale=func[4]
finalIH=func[5]
finalVAlH=func[6]
finalVOH=func[7]
finalVO=func[8]
finalVAl=func[9]
finalIO=func[10]
finalIAl=func[11]
p=np.array([[1e-15,1e-14,1e-13,1e-12,1e-11,1e-10,1e-09,1e-08,1e-07,1e-06,
             1e-05,1e-04,1e-03,1e-02,1e-01,1]])
logp=np.log10(p)
logh=np.log10(finalh)
loge=np.log10(finale)
logVAl=np.log10(finalVAl)
logIH=np.log10(finalIH)
logVAlH=np.log10(finalVAlH)
logVOH=np.log10(finalVOH)
logVO=np.log10(finalVO)
logVAl=np.log10(finalVAl)
logIO=np.log10(finalIO)
logIAl=np.log10(finalIAl)
#plt.figure(figsize=(3.84,2.95))
axes=plt.gca()
axes.set_xlim([-15,0])
axes.set_ylim([-18,-4])
font = {'fontname':'Times New Roman','fontsize':7}
plt.xlabel('logpH2 (atm)',**font)
plt.ylabel('log[D] per f.u. Al2O3',**font)
plt.plot(logp[0,:],logVO[0,:],color='#e41a1c',marker='.',label='VOx')
plt.plot(logp[0,:],logVO[1,:],color='#e41a1c',marker='*',label='VO+1')
plt.plot(logp[0,:],logVO[2,:],color='#e41a1c',marker='p',label='VO+2')
plt.plot(logp[0,:],logVAl[0,:],color='#377eb8',marker='P',label='VAlx')
plt.plot(logp[0,:],logVAl[1,:],color='#377eb8',marker='+',label='VAl-1')
plt.plot(logp[0,:],logVAl[2,:],color='#377eb8',marker='x',label='VAl-2')
plt.plot(logp[0,:],logVAl[3,:],color='#377eb8',marker='o',label='VAl-3')
plt.plot(logp[0,:],logIO[0,:],color='#4daf4a',marker='X',label='IOx')
plt.plot(logp[0,:],logIO[1,:],color='#4daf4a',marker='h',label='IO-1')
plt.plot(logp[0,:],logIO[2,:],color='#4daf4a',marker='H',label='IO-2')
plt.plot(logp[0,:],logIAl[0,:],color='#984ea3',marker='d',label='IAlx')
plt.plot(logp[0,:],logIAl[1,:],color='#984ea3',marker='D',label='IAl+1')
plt.plot(logp[0,:],logIAl[2,:],color='#984ea3',marker='s',label='IAl+2')
plt.plot(logp[0,:],logIAl[3,:],color='#984ea3',marker='^',label='IAl+3')
plt.plot(logp[0,:],logh[:,0],color='#ff7f00',marker='>',label='h')
plt.plot(logp[0,:],loge[:,0],color='#ffff33',marker='<',label='e')
plt.plot(logp[0,:],logIH[0,:],color='#a65628',marker='.',label='IH-1')
plt.plot(logp[0,:],logIH[1,:],color='#a65628',marker='*',label='IHx')
plt.plot(logp[0,:],logIH[2,:],color='#a65628',marker='p',label='IH+1')
plt.plot(logp[0,:],logVOH[0,:],color='#f781bf',marker='P',label='VOHx')
plt.plot(logp[0,:],logVOH[1,:],color='#f781bf',marker='+',label='VOH+1')
plt.plot(logp[0,:],logVAlH[0,:],color='#999999',marker='h',label='VAlH-3')
plt.plot(logp[0,:],logVAlH[1,:],color='#999999',marker='H',label='VAlH-2')
plt.plot(logp[0,:],logVAlH[2,:],color='#999999',marker='d',label='VAlH-1')
plt.plot(logp[0,:],logVAlH[3,:],color='#999999',marker='D',label='VAlHx')
plt.plot(logp[0,:],logVAlH[4,:],color='#999999',marker='s',label='VAlH+1')
plt.plot(logp[0,:],logVAlH[5,:],color='#984ea3',marker='H',label='VAl2H-1')
plt.plot(logp[0,:],logVAlH[6,:],color='#ff7f00',marker='^',label='VAl3Hx')
plt.plot(logp[0,:],logVAlH[7,:],color='#ff7f00',marker='x',label='VAl4H+1')
plt.plot(logp[0,:],logVAlH[8,:],color='#984ea3',marker='+',label='VAl5H+2')
plt.legend(loc='upper left')
pylab.savefig('with_hydrogen_300K.png')

'''

plt.plot(logp[0,:],logVAlH[6,:],color='#7570b3',label='$[V_{Al}-3H]^{x}$',linewidth=2)
plt.plot(logp[0,:],logVAlH[7,:],color='#000000',marker='>',markersize=5,label='$[V_{Al}-4H]^{.}')
plt.plot(logp[0,:],logVAlH[2,:],color='#e6ab02',label='$[V_{Al}-H]^{,}$',linewidth=2)
plt.plot(logp[0,:],logIH[2,:],color='#e41a1c',label='$H_i^."\u0387"$',linewidth=2)
plt.plot(logp[0,:],logVAlH[3,:],color='#a6761d',label='$[V_{Al}-H]^{x}$',linewidth=2)
ax=plt.subplot(111)
ax.tick_params(axis='both', which='major', labelsize=7)
pylab.savefig('undoped_H_300K.png')
concH=sum(finalIH[:,(lim-1)])+sum(finalVOH[:,(lim-1)])+sum(finalVAlH[:,(lim-1)])
logconcH=np.log10(concH)
"""
plt.plot(logp[0,:],logVO[0,:],color='#e41a1c',marker='.',label='VOx')
plt.plot(logp[0,:],logVO[1,:],color='#e41a1c',marker='*',label='VO+1')
plt.plot(logp[0,:],logVO[2,:],color='#e41a1c',marker='p',label='VO+2')
plt.plot(logp[0,:],logVAl[0,:],color='#377eb8',marker='P',label='VAlx')
plt.plot(logp[0,:],logVAl[1,:],color='#377eb8',marker='+',label='VAl-1')
plt.plot(logp[0,:],logVAl[2,:],color='#377eb8',marker='x',label='VAl-2')
plt.plot(logp[0,:],logVAl[3,:],color='#377eb8',marker='o',label='VAl-3')
plt.plot(logp[0,:],logIO[0,:],color='#4daf4a',marker='X',label='IOx')
plt.plot(logp[0,:],logIO[1,:],color='#4daf4a',marker='h',label='IO-1')
plt.plot(logp[0,:],logIO[2,:],color='#4daf4a',marker='H',label='IO-2')
plt.plot(logp[0,:],logIAl[0,:],color='#984ea3',marker='d',label='IAlx')
plt.plot(logp[0,:],logIAl[1,:],color='#984ea3',marker='D',label='IAl+1')
plt.plot(logp[0,:],logIAl[2,:],color='#984ea3',marker='s',label='IAl+2')
plt.plot(logp[0,:],logIAl[3,:],color='#984ea3',marker='^',label='IAl+3')
plt.plot(logp[0,:],logh[:,0],color='#ff7f00',marker='>',label='h')
plt.plot(logp[0,:],loge[:,0],color='#ffff33',marker='<',label='e')
plt.plot(logp[0,:],logIH[0,:],color='#a65628',marker='.',label='IH-1')
plt.plot(logp[0,:],logIH[1,:],color='#a65628',marker='*',label='IHx')
plt.plot(logp[0,:],logIH[2,:],color='#a65628',marker='p',label='IH+1')
plt.plot(logp[0,:],logVOH[0,:],color='#f781bf',marker='P',label='VOHx')
plt.plot(logp[0,:],logVOH[1,:],color='#f781bf',marker='+',label='VOH+1')
plt.plot(logp[0,:],logVAlH[0,:],color='#999999',marker='h',label='VAlH-3')
plt.plot(logp[0,:],logVAlH[1,:],color='#999999',marker='H',label='VAlH-2')
plt.plot(logp[0,:],logVAlH[2,:],color='#999999',marker='d',label='VAlH-1')
plt.plot(logp[0,:],logVAlH[3,:],color='#999999',marker='D',label='VAlHx')
plt.plot(logp[0,:],logVAlH[4,:],color='#999999',marker='s',label='VAlH+1')
plt.plot(logp[0,:],logVAlH[5,:],color='#984ea3',marker='H',label='VAl2H-1')
plt.plot(logp[0,:],logVAlH[6,:],color='#ff7f00',marker='^',label='VAl3Hx')
plt.plot(logp[0,:],logVAlH[7,:],color='#ff7f00',marker='x',label='VAl4H+1')
plt.plot(logp[0,:],logVAlH[8,:],color='#984ea3',marker='+',label='VAl5H+2')
plt.legend()
"""
"""
#pylab.legend(loc='upper left')

#ax=plt.subplot(111)

# Shrink current axis by 20%
#box = ax.get_position()
#ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
# Put a legend to the right of the current axis
#ax.legend(loc='upper right',bbox_to_anchor=(1, 1))
#ax.legend().draggable()
"""
'''