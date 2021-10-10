# Defect-equilibria
This repository contains the Python code and example input files to calculate defect equilibria in ionic crystals (also known as Kroger-Vink or Brouwer diagrams). <br/>
<br/>
This code was used to calculate the defect equilibria in undoped Al2O3 across a range of oxygen gas partial pressures at 1100 K and in undoped Al2O3 across a range of hydrogen gas partial pressures at 300 K. For the latter case, the concentrations of native defects in Al2O3 are kept fixed at the values obtained at 1100 K, high pO2 conditions. This is because the low temperature (300 K) and high migration barriers for native defects in Al2O3 prevents them from re-equilibrating with the environment and causes them to be kinetically trapped. <br/>
<br/>
Our paper (Somjit, Vrindaa, and Bilge Yildiz. "Doping α-Al2O3 to reduce its hydrogen permeability: Thermodynamic assessment of hydrogen defects and solubility from first principles." Acta Materialia 169 (2019): 172-183) and its supplementary informatio. has details on the the derivation of the different concentration terms, and interpretation of the results, etc.<br/>
<br/>
I wrote this code early on during my PhD when I was still learning the ropes, so it is not the most modularized. I hope to clean it up sometime, however, it is still easy to understand and use, and can be extended to other materials and other cases as well (such as doped systems).<br/>
