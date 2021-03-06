# Defect-equilibria
This repository contains the Python code and example input files to calculate defect equilibria in ionic crystals (also known as Kroger-Vink or Brouwer diagrams). <br/>
<br/>
This code was used to calculate the defect equilibria in undoped Al2O3 across a range of oxygen gas partial pressures at 1100 K and in undoped Al2O3 across a range of hydrogen gas partial pressures at 300 K. For the latter case, the concentrations of native defects in Al2O3 are kept fixed at the values obtained at 1100 K, high pO2 conditions. This is to reflect 'frozen in' defects from Al2O3 growth and subsequent hydrogen complex formation during cooldown. The low temperature (300 K) and high migration barriers for native defects in Al2O3 prevents them from re-equilibrating with the environment, causing them to be kinetically trapped. However, hydrogen is mobile enough to equilibrate with the environment and form complexes with the native defects. <br/>
<br/>
Our paper (Somjit, Vrindaa, and Bilge Yildiz. "Doping α-Al2O3 to reduce its hydrogen permeability: Thermodynamic assessment of hydrogen defects and solubility from first principles." Acta Materialia 169 (2019): 172-183) and its supplementary information has details on the method, derivation of the different concentration terms, and interpretation of the results, etc.<br/>
<br/>
I wrote this code early on during my PhD when I was still learning the ropes, so it is not the most modularized. I hope to clean it up sometime, however, it is still easy to understand and use, and can be extended to other materials and other cases as well (such as doped systems).<br/>
<br/>
This code self-consistently calculates the Fermi level at a given pO2 or pH2 at which charge neutrality is obtained. The electronic and ionic defect concentrations are calculated using this equilibrium Fermi level at a given pO2. Doing this over a range of pO2 gives us the Kroger-Vink diagram.
