#!/usr/local/bin/env python
# -*- coding: utf-8 -*-

"""
#######################################
#                                     #
#-- Fracture Simulation of Networks:  --#
## Tensile testing in x-direction with initial box rotated in any generalized angle theta along specificd hkl plane ###
#------  Generalization to hkl,theta rotation- Author: Devosmita Sen  --------#
#------  Original Author: Akash Arora  --------#
#                                     #
#######################################

 Overall Framework (Steps):
     1. Generate a Network following the algorithm published
        by AA Gusev, Macromolecules, 2019, 52, 9, 3244-3251
        if gen_net = 0, then it reads topology from user-supplied 
        network.txt file present in this folder
     
     2. Force relaxtion of network using Fast Inertial Relaxation Engine (FIRE) 
        to obtain the equilibrium positions of crosslinks (min-energy configuration)

     3. Compute Properties: Energy, Gamma (prestretch), and 
        Stress (all 6 componenets) 
     
     4. Deform the network (tensile) in desired direction by 
        strain format by supplying lambda_x, lambda_y, lambda_z

     5. Break bonds using Kintetic Theory of Fracture (force-activated KMC) 
        presently implemented algorithm is ispired by 
        Termonia et al., Macromolecules, 1985, 18, 2246

     6. Repeat steps 2-5 until the given extension (lam_total) is achived OR    
        stress decreases below a certain (user-specified) value 
        indicating that material is completey fractured.
"""

import time
import math
import random
##import netgen
import ioLAMMPS
import matplotlib
import numpy as np
from relax import Optimizer
from numpy import linalg as LA
from scipy.optimize import fsolve
from matplotlib import pyplot as plt
import param as p
from scipy.spatial.transform import Rotation
random.seed(a=None,version=2)
##random.seed(a=None, version=2)
print('First random number of this seed: %d'%(random.randint(0, 10000))) 
import os
# This is just to check whether different jobs have different seeds


def meanr2_fun(n_chains, chains, links, Lx, Ly, Lz):
    meanr2=0
    dist = np.zeros((n_chains,4))
    dist[:,0:3] = chains[:,1:]
    dist[:,3] = -1
        
    for i in range (0, n_chains):
        if(chains[i,2] !=-1):
      
          link_1 = chains[i,2]-1
          link_2 = chains[i,3]-1
          lk = links[link_1,:] - links[link_2,:]
          
          lk[0] = lk[0] - int(round(lk[0]/Lx))*Lx
          lk[1] = lk[1] - int(round(lk[1]/Ly))*Ly
          lk[2] = lk[2] - int(round(lk[2]/Lz))*Lz
                
          dist[i,3] = LA.norm(lk)
          meanr2=meanr2+(dist[i,3])**2
####          print(dist[i,3])
##          stop
##          print(((dist[i,3])**2)/(p.N_low*p.b_low**2))
          

    return meanr2


netgen_flag = 0
## read in orthogonal axis alligned simulation box (lattice)
if(netgen_flag==0):

   vflag = 0
   ##N = 12   
   print('--------------------------')   
   print('----Reading Network-------')   
   print('--------------------------')   
   [xlo, xhi, ylo, yhi, zlo, zhi, n_atoms, n_bonds, atoms, bonds,
           atom_types, bond_types, mass, loop_atoms] = ioLAMMPS.readLAMMPS_restart("network_final.txt",  vflag)
    
   #bond_types=n_bonds ##p.bond_types  ## allow n_bonds number of bonds, because theoretically, all chains can have a different N value which is sampled from the distribution
   ##perc_T2=p.perc_T2
   
   print('xlo, xhi',xlo, xhi) 
   print('ylo, yhi',ylo, yhi) 
   print('zlo, zhi',zlo, zhi) 
   print('n_atoms', n_atoms) 
   print('n_bonds', n_bonds) 
   print('atom_types = ', atom_types) 
   
   print('mass = ', mass) 
   print('primary loops = ', len(loop_atoms)) 
   print('--------------------------')
   n_chains=n_bonds
   n_links=n_atoms

elif(netgen_flag==1):

   func = p.func
   N    = p.N
   b=p.b
   rho  = p.rho
   l0   = 1
   prob = 1.0
   n_chains  = p.n_chains
   n_links   = int(2*n_chains/func)
##   L = 32.5984

   cR3=p.cR3
   conc=cR3/(N*b**2)**1.5 #(chains/nm3)
   ##n_chains=len(G.edges)
   L=(tot_chains/conc)**(1/3)
   netgen.generate_network(prob, func, N, L, l0, n_chains, n_links)

   [xlo, xhi, ylo, yhi, zlo, zhi, n_atoms, n_bonds, atoms, bonds, 
           atom_types, bond_types, mass, loop_atoms] = ioLAMMPS.readLAMMPS("network.txt", N, 0)


else:
   print('Invalid network generation flag')


############################  Assign chains their N values- which are sampled from a Gaussian distribution ################################3

mean_lognormal=np.log(p.mean_N**2/(np.sqrt(p.mean_N**2+p.sigma_N**2)))
sigma_log_normal=np.sqrt(np.log(1+p.sigma_N**2/p.mean_N**2))

N_arr=np.random.lognormal(mean=mean_lognormal, sigma=sigma_log_normal, size=n_chains)  ## generate n_chains number of samples from Gaussian distribution

bin_width = p.bin_width
min_val=np.min(N_arr) #########this gets added to Optimizer to track N values later ## this also needs to be written to file in order to track N values dring post processing

max_val=np.max(N_arr)

np.savetxt('min_max_val_N.txt',np.array([min_val, max_val]))




edges = np.arange(min_val, max_val + bin_width, bin_width)  ## the N data is converted to a linear form from min_val to max_val+bin_width; N value can thus be extracted later from this

bin_indices = np.digitize(N_arr, edges) 


for i in range(0,n_chains):
  bonds[i,1]=bin_indices[i]

n_bins=len(edges)-1

bond_types=n_bins
print('bond_types = ', bond_types) 

### to get back the N value ####

## bond type i
#N= min_val + ((i-1) * bin_width) + (bin_width / 2)  ## this is not the exact value that was chosen initially, but is rounded within the bin_width specified. smaller bin_width will give more accurate estimates

hkl=p.hkl

fstr=open('stress','w')
fstr.write('#Lx, Ly, Lz, lambda, FE, deltaFE, st[0], st[1], st[2], st[3], st[4], st[5]\n') 

##flen=open('strand_lengths','w')
##flen.write('#lambda, ave(R), max(R)\n') 

fkmc=open('KMC_stats','w')
fkmc.write('#lambda, init bonds, final bonds\n') 

fH=open('H_matrix','w')
fH.write('#iteration, H_matrix_flattened \n')



import numpy as np
def append_ovito_frame(filename, atoms, bonds, H, origin, iteration, species="C"):
    """
    Write an OVITO-compatible extended XYZ file with arbitrary H (column format)
    and origin, including up to 4 bond neighbors per atom.
    Appends each iteration as a new frame.
    """
    n_atoms = len(atoms)
    max_bonds = 4
    H_for_write = H.T

    # Initialize bonded neighbor lists
    bonded_neighbors = [[] for _ in range(n_atoms)]
    #print(np.min(bonds))

    if bonds is not None and len(bonds) > 0:
        for i, j in bonds[:, 2:]:
            bonded_neighbors[i - 1].append(j)
            bonded_neighbors[j - 1].append(i)

    # Pad or truncate to fixed number of bond columns
    padded_neighbors = []
    for lst in bonded_neighbors:
        lst = lst[:max_bonds]
        lst += [0] * (max_bonds - len(lst))
        padded_neighbors.append(lst)

    # Now write file
    with open(filename, "a") as f:
        f.write(f"{n_atoms}\n")
        f.write(
            f'Lattice="{H_for_write[0,0]} {H_for_write[0,1]} {H_for_write[0,2]} '
            f'{H_for_write[1,0]} {H_for_write[1,1]} {H_for_write[1,2]} '
            f'{H_for_write[2,0]} {H_for_write[2,1]} {H_for_write[2,2]}" '
            f'Origin="{origin[0]} {origin[1]} {origin[2]}" '
            f'Properties=species:S:1:pos:R:3'
            f':Bond1:I:1:Bond2:I:1:Bond3:I:1:Bond4:I:1 '
            f'Time={iteration}\n'
        )

        for i, pos in enumerate(atoms):
            nb = padded_neighbors[i]
            f.write(
                f"{species} {pos[0]:.8f} {pos[1]:.8f} {pos[2]:.8f} "
                f"{nb[0]} {nb[1]} {nb[2]} {nb[3]}\n"
            )
        '''  
        if bonds is not None and len(bonds) > 0:
          bonds_filename = os.path.splitext(filename)[0] + ".bonds"
          with open(bonds_filename, "w") as fb:
            fb.write(f"{len(bonds)}\n")
            fb.write("Bonds\n")
            for i, j in bonds[:, 2:]:
                fb.write(f"{i} {j}\n")
                
        '''

           
import numpy as np
import os

        
import gsd.hoomd
import gsd
import gsd.fl

def append_pdb_frame(filename, atoms, bonds, H, origin, iteration, species="C"):
    """
    Writes a simulation frame to a GSD trajectory file.

    This function correctly handles arbitrary box orientations (H matrix)
    using QR decomposition and saves dynamic bond information for each frame.
    It is designed to be called sequentially in a loop.

    Args:
        filename (str): The name of the GSD file (e.g., "trajectory.gsd").
        atoms (np.ndarray): Nx3 array of absolute particle positions.
        bonds (np.ndarray): Mx4 array, where columns 2 and 3 are the
                            1-based indices of bonded atom pairs.
        H (np.ndarray): 3x3 box matrix where rows are the box vectors (a, b, c).
        origin (np.ndarray): 1x3 vector of the box origin (corner).
        iteration (int): The current frame/iteration number.
        species (str or list): Either a single string (e.g., "C") for all
                               atoms, or a list of strings (length N)
                               specifying the type for each atom.
    """
    
    # --- Determine File Mode ---
    # 'wb' (write) for the first frame to create/overwrite the file.
    # 'ab' (append) for all subsequent frames.
    mode = 'w' if iteration == -2 else 'a'

    try:
        with gsd.hoomd.open(name=filename, mode=mode) as traj:
            s = gsd.hoomd.Frame()
            n_atoms = len(atoms)

            # --- 1. Box Orientation (QR Decomposition) ---
            # Your H is H_orig, with row vectors. We need to decompose its
            # transpose to get a lower-triangular matrix.
            H_orig = H
            # 1. Decompose H (which has column vectors)
            Q_temp, R_temp = np.linalg.qr(H)
            
            # 2. Force a canonical decomposition (positive diagonal on R)
            #    This fixes the "rotated backwards" ambiguity.
            signs = np.diag(np.sign(np.diag(R_temp)))
            Q_temp = Q_temp @ signs
            R_temp = signs @ R_temp
            
            # 3. Our GSD box matrix is the lower-triangular L = R_temp.T
            H_gsd = R_temp.T
            
            # 4. Our rotation matrix is Q_rot = Q_temp.T
            Q_rot = Q_temp.T
            
            #H_gsd = R_upper.T  # New lower-triangular box matrix

            # Extract the 6 GSD box parameters from H_gsd
            Lx = H_gsd[0, 0]
            Ly = H_gsd[1, 1]
            Lz = H_gsd[2, 2]
            
            # Handle potential division by zero if box is 2D (Lz=0)
            xy = H_gsd[1, 0] / Ly if Ly != 0 else 0.0
            xz = H_gsd[2, 0] / Lz if Lz != 0 else 0.0
            yz = H_gsd[2, 1] / Lz if Lz != 0 else 0.0

            s.configuration.box = [Lx, Ly, Lz, xy, xz, yz]

            # --- 2. Particle Positions ---
            # a. Get positions relative to the box origin
            P_rel_corner = atoms - origin
            
            # b. Apply the *inverse* rotation (Q) to the relative positions
            #    (Q is from H.T = Q @ R_upper, so P_new = P_orig @ Q)
            P_rotated = P_rel_corner @ Q_temp
            
            # c. Find center of the new GSD box
            box_center = 0.5 * np.array([Lx + xy*Ly + xz*Lz, Ly + yz*Lz, Lz])
            
            # d. Get positions relative to the new box center (as GSD expects)
            P_gsd = P_rotated - box_center
            
            s.particles.N = n_atoms
            s.particles.position = P_gsd.astype(np.float32)

            # --- 3. Particle Species ---
            if isinstance(species, str):
                # Case 1: Single species string for all atoms
                s.particles.types = [species]
                s.particles.typeid = np.zeros(n_atoms, dtype=np.uint32)
            elif isinstance(species, list) and len(species) == n_atoms:
                # Case 2: Per-atom list of species strings
                unique_types = sorted(list(set(species)))
                type_map = {name: i for i, name in enumerate(unique_types)}
                
                s.particles.types = unique_types
                s.particles.typeid = np.array([type_map[s] for s in species], dtype=np.uint32)
            else:
                raise ValueError("Species must be a single string or a list of strings of length N_atoms")

            # --- 4. Dynamic Bonds ---
            if bonds is not None and len(bonds) > 0:
                # We assume 'bonds' is Mx4, and columns 2 and 3 (i.e., [:, 2:4])
                # contain the 1-BASED atom indices, as implied by your
                # original function's logic.
                
                # Get M x 2 array of 1-based pairs
                bond_pairs_1_indexed = bonds[:, 2:4] 
                
                # Convert to 0-based indexing for GSD
                bond_pairs_0_indexed = bond_pairs_1_indexed - 1
                
                s.bonds.N = len(bond_pairs_0_indexed)
                s.bonds.group = bond_pairs_0_indexed.astype(np.uint32)
                
                # GSD requires at least one bond type name
                s.bonds.types = ['default_bond']
                s.bonds.typeid = np.zeros(s.bonds.N, dtype=np.uint32)
            else:
                s.bonds.N = 0

            # --- 5. Append Frame ---
            traj.append(s)
            
    except Exception as e:
        print(f"Error writing GSD frame {iteration}: {e}")
        # Optionally re-raise the exception if you want to stop the simulation
        # raise e
        
    filename=filename[0:-4]+'_atoms_only_correct_box_orient.xyz'
    append_ovito_frame(filename, atoms, bonds, H, origin, iteration, species="C")




#-------------------------------------#
#       Simulation Parameters         #
#-------------------------------------#

#N  = 12
#Nb = N
##N1=p.N1
##N2=p.N2
b=p.b
K  = p.K
r0 = 0.0
U0  = p.U0
tau = p.tau
del_t = p.del_t
erate = p.erate
lam_max = p.lam_max
tol = p.tol
max_itr = p.max_itr
write_itr = p.write_itr
wrt_step = p.wrt_step
angle_deg=p.angle_deg


############ in this code- the box dimensions in the lammps file are not changed- because those loose some information#######
##### all info is stored in the H matrix ################


#-------------------------------------#
#       First Force Relaxation        #
#-------------------------------------#

mymin = Optimizer(atoms, bonds, xlo, xhi, ylo, yhi, zlo, zhi, K, r0, min_val, bin_width,'Mao') ##min_val is the minimum value of N chosen in this simulation; bin_width is the width of the bin for the array of N values chosen


output_file = "full_trajectory.xyz"
# Make sure to delete the file from previous runs before starting the loop
if os.path.exists(output_file):
    os.remove(output_file)
'''    
with open(output_file, "w") as f:
    f.write("OVITO Data File\n")
    f.write("My dynamic bond simulation\n")
'''   

ioLAMMPS.writeLAMMPS('network_before_swelling.txt', mymin.xlo, mymin.xhi, mymin.ylo, mymin.yhi, mymin.zlo,
                                  mymin.zhi, mymin.atoms, mymin.bonds, atom_types, bond_types, mass, loop_atoms)
## first swell the network:
H_string = " ".join([f"{x:.8f}" for x in mymin.H.flatten()])
ite=-2
fH.write(f"{ite} {H_string}\n")

append_pdb_frame(output_file, mymin.atoms, mymin.bonds, mymin.H, np.array([mymin.xlo, mymin.ylo, mymin.zlo]), ite)


swell = 1
print(mymin.xhi-mymin.xlo,mymin.xhi-mymin.xlo,mymin.xhi-mymin.xlo)
cR3=p.cR3 ## dimless conc
conc=cR3/(p.mean_N*b**2)**1.5 #(chains/nm3)
V=(n_chains/conc)  ## this is the target volume of the network based on the dimless conc and mean N of the network

V_net=(xhi-xlo)*(yhi-ylo)*(zhi-zlo)## volume of network currently

#mean_r2=meanr2_fun(len(mymin.bonds), mymin.bonds, mymin.atoms, (mymin.xhi-mymin.xlo), (mymin.yhi-mymin.ylo), (mymin.zhi-mymin.zlo))
#gamma_curr=mean_r2/(len(mymin.bonds)*p.N_low*p.b_low**2)
#print('init_gamma=',gamma_curr)
gamma_desired=p.desired_gamma


#V_init=(mymin.xhi-mymin.xlo)*(mymin.yhi-mymin.ylo)*(mymin.zhi-mymin.zlo)
n_chains=len(mymin.bonds)

swell_factor=(V/V_net)**(1/3) ## perform isotropic swelling##(n_chains/V_init)*(gamma_curr/gamma_desired)*((N*b**2)**1.5/cR3) #np.sqrt(gamma_desired/gamma_curr)

#swell_factor=(V/V_net)**(1/3) ## perform isotropic swelling
scale_x=scale_y=scale_z=swell_factor



if(swell==1):
####   ioLAMMPS.writeLAMMPS('restart_network_01.txt', mymin.xlo, mymin.xhi, mymin.ylo, mymin.yhi, mymin.zlo,
####                                  mymin.zhi, mymin.atoms, mymin.bonds, atom_types, bond_types, mass, loop_atoms)
   # Swelling the network to V = 2
####   scale_x = 1.26
####   scale_y = 1.26
####   scale_z = 1.26
   mymin.change_box(scale_x, scale_y, scale_z)
   ioLAMMPS.writeLAMMPS('network_after_swelling_only.txt', mymin.xlo, mymin.xhi, mymin.ylo, mymin.yhi, mymin.zlo,
                                  mymin.zhi, mymin.atoms, mymin.bonds, atom_types, bond_types, mass, loop_atoms)
   #bondlengths_noPBC = mymin.bondlengths_noPBC()
   [e, Gamma] = mymin.fire_iterate(tol, max_itr, write_itr, 'log.txt')
####print((mymin.xhi-mymin.xlo)*(mymin.yhi-mymin.ylo)*(mymin.zhi-mymin.zlo), V)
####stop

ioLAMMPS.writeLAMMPS('network_after_swelling_and_relax.txt', mymin.xlo, mymin.xhi, mymin.ylo, mymin.yhi, mymin.zlo,
                                  mymin.zhi, mymin.atoms, mymin.bonds, atom_types, bond_types, mass, loop_atoms)
H_string = " ".join([f"{x:.8f}" for x in mymin.H.flatten()])
ite=-1
fH.write(f"{ite} {H_string}\n")
append_pdb_frame(output_file, mymin.atoms, mymin.bonds, mymin.H, np.array([mymin.xlo, mymin.ylo, mymin.zlo]), ite)

mean_r2=meanr2_fun(len(mymin.bonds), mymin.bonds, mymin.atoms, (mymin.xhi-mymin.xlo), (mymin.yhi-mymin.ylo), (mymin.zhi-mymin.zlo))
##gamma_new=mean_r2/(len(mymin.bonds)*p.N_low*p.b_low**2)
##print('new_gamma=',gamma_new)
V_new=(mymin.xhi-mymin.xlo)*(mymin.yhi-mymin.ylo)*(mymin.zhi-mymin.zlo)
conc_new=n_chains/V_new

##cR3_new=conc_new*(N*b**2)**1.5
##print('cR3_new',cR3_new)



## Rotate this box and atoms according to specified angle and theta direction

hkl = np.asarray(hkl, dtype=float)
if np.linalg.norm(hkl) == 0:
    raise ValueError("The hkl vector cannot be a zero vector.")
axis = hkl / np.linalg.norm(hkl)
angle_rad = np.deg2rad(angle_deg)
#rotation_vector = angle_rad * axis
#rotation = Rotation.from_rotvec(rotation_vector)

# 3. Create the rotation object from the axis-angle representation
# The 'rotation vector' is the axis multiplied by the angle in radians.
rotation_vector = angle_rad * axis
rotation = Rotation.from_rotvec(rotation_vector)
# Get the 3x3 rotation matrix from the same object
R_matrix = rotation.as_matrix()
# 4. Apply the rotation to the points
'''

origin_old = np.array([mymin.xlo, mymin.ylo, mymin.zlo])
box_vectors_sum = np.sum(mymin.H, axis=1) 
center_of_rotation = origin_old + 0.5 * box_vectors_sum
print(f"Center of rotation: {center_of_rotation}")

# 2. TRANSLATE-ROTATE-TRANSLATE THE ATOMS AND BOX ORIGIN
# First, translate the system to the origin
atoms_shifted = mymin.atoms - center_of_rotation
origin_shifted = origin_old - center_of_rotation

# Second, rotate the shifted system
atoms_rotated = rotation.apply(atoms_shifted)
origin_rotated = rotation.apply(origin_shifted)

# Third, translate the system back
atoms_new = atoms_rotated + center_of_rotation
origin_new = origin_rotated + center_of_rotation

# 3. ROTATE THE BOX VECTORS
# The box vectors (orientation/shape) are rotated directly.
# They are not affected by the translation.
H_new = R_matrix @ mymin.H##mymin.H @ R_matrix.T  ##R_matrix @ mymin.H

# 4. UPDATE YOUR OBJECT WITH THE NEW VALUES
mymin.atoms = atoms_new

mymin.H = H_new
mymin.xlo, mymin.ylo, mymin.zlo = origin_new
'''
# --- DEBUGGING BLOCK ---

# 1. Check Initial Values
origin_old = np.array([mymin.xlo, mymin.ylo, mymin.zlo])
print(f"DEBUG: Initial Origin (xlo, ylo, zlo) = {origin_old}")
print(f"DEBUG: Initial H Matrix:\n{mymin.H}")

# 2. Check Center Calculation
box_vectors_sum = np.sum(mymin.H, axis=1) 
center_of_rotation = origin_old + 0.5 * box_vectors_sum
print(f"DEBUG: Calculated Center of Rotation = {center_of_rotation}")
# --- For your box, this MUST be a non-zero value like [11.89, 11.89, 11.89] ---
# --- If it's [0,0,0], that is the source of the problem! ---

# 3. Check Translate-Rotate-Translate on a single atom
first_atom_initial = mymin.atoms[0].copy()
print(f"\nDEBUG: First atom initial pos = {first_atom_initial}")

atoms_shifted = mymin.atoms - center_of_rotation
print(f"DEBUG: First atom shifted = {atoms_shifted[0]}")

atoms_rotated = rotation.apply(atoms_shifted)
print(f"DEBUG: First atom rotated (but still shifted) = {atoms_rotated[0]}")

atoms_new = atoms_rotated + center_of_rotation
print(f"DEBUG: First atom FINAL pos = {atoms_new[0]}")

# 4. Check the box origin transformation
origin_shifted = origin_old - center_of_rotation
origin_rotated = rotation.apply(origin_shifted)
origin_new = origin_rotated + center_of_rotation
print(f"\nDEBUG: Final new origin = {origin_new}")

# 5. Check the final box vectors
H_new = R_matrix @ mymin.H
print(f"DEBUG: Final new H Matrix:\n{H_new}")


# 6. UPDATE YOUR OBJECT WITH THE NEW VALUES
mymin.atoms = atoms_new
mymin.H = H_new
mymin.xlo, mymin.ylo, mymin.zlo = origin_new

'''
import numpy as np, numpy.linalg as la

H = mymin.H.copy()
origin = np.array([mymin.xlo, mymin.ylo, mymin.zlo])
atoms_rot = mymin.atoms.copy()   # result after your rotation

# 1) Interpretation A: H is columns (H_col)
H_col = H.copy()
Hcol_inv = la.inv(H_col)
sA = (Hcol_inv @ (atoms_rot - origin).T).T
reconA = (H_col @ (sA % 1).T).T + origin
errA = np.max(la.norm(reconA - atoms_rot, axis=1))

# 2) Interpretation B: H is rows (H_row)
H_row = H.copy()
H_row_as_cols = H_row.T
Hrow_inv = la.inv(H_row_as_cols)
sB = (Hrow_inv @ (atoms_rot - origin).T).T
reconB = (H_row_as_cols @ (sB % 1).T).T + origin
errB = np.max(la.norm(reconB - atoms_rot, axis=1))

print("err assuming H is columns:", errA)
print("err assuming H is rows (converted to cols by transpose):", errB)
stop
'''
##stop

#box_vectors_sum = np.sum(mymin.H, axis=1) 
#center_of_rotation_new = origin_new + 0.5 * box_vectors_sum
#print(center_of_rotation,center_of_rotation_new)
#stop

'''
# 2. Apply the rotation to the old origin to find its new position
# The .apply() method is perfect for this.
origin_new = rotation.apply(origin_old)

# 3. Update the origin attributes in your object
mymin.xlo, mymin.ylo, mymin.zlo = origin_new


mymin.atoms = rotation.apply(mymin.atoms)

## rotate the bounding box- and calculate H accordingly
mymin.H=R_matrix @ mymin.H
'''

ioLAMMPS.writeLAMMPS('network_just after_rotation.txt', mymin.xlo, mymin.xhi, mymin.ylo, mymin.yhi, mymin.zlo,
                                  mymin.zhi, mymin.atoms, mymin.bonds, atom_types, bond_types, mass, loop_atoms)

[e, Gamma] = mymin.fire_iterate(tol, max_itr, write_itr, 'log.txt')
ioLAMMPS.writeLAMMPS('network_after_rotation_and_relax.txt', mymin.xlo, mymin.xhi, mymin.ylo, mymin.yhi, mymin.zlo,
                                  mymin.zhi, mymin.atoms, mymin.bonds, atom_types, bond_types, mass, loop_atoms)

ioLAMMPS.writeLAMMPS('restart_network_0.txt', mymin.xlo, mymin.xhi, mymin.ylo, mymin.yhi, mymin.zlo,
                                  mymin.zhi, mymin.atoms, mymin.bonds, atom_types, bond_types, mass, loop_atoms)

H_string = " ".join([f"{x:.8f}" for x in mymin.H.flatten()])
ite=0
fH.write(f"{ite} {H_string}\n")
fH.flush()
append_pdb_frame(output_file, mymin.atoms, mymin.bonds, mymin.H, np.array([mymin.xlo, mymin.ylo, mymin.zlo]), ite)

####stop
##dist = mymin.bondlengths()
Lx0 = mymin.xhi-mymin.xlo
BE0 = e
[pxx, pyy, pzz, pxy, pyz, pzx] = mymin.compute_pressure()
fstr.write('%7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f\n' \
                          %(mymin.xhi-mymin.xlo, mymin.yhi-mymin.ylo, mymin.zhi-mymin.zlo, 
                           (mymin.xhi-mymin.xlo)/Lx0, e, e-BE0, pxx, pyy, pzz, pxy, pyz, pzx)) 
fstr.flush()

##flen.write('%7.4f  %7.4f  %7.4f\n'%((mymin.xhi-mymin.xlo)/Lx0, np.mean(dist[:,3])/N, np.max(dist[:,3])/N))
##flen.flush()

fkmc.write('%7.4f  %5i  %5i\n'%((mymin.xhi-mymin.xlo)/Lx0, n_bonds, n_bonds))
fkmc.flush()


#-------------------------------------#
# Tensile deformation: lambda/scales  #
#-------------------------------------#
steps = int((lam_max-1)/(erate*del_t))
print('Deformation steps = ',steps)
begin_break = -1         # -1 implies that bond breaking begins right from start
#begin_break = n_steps   # implies bond breaking will begin after n_steps of deformation

for i in range(0,steps):

    scale_x = (1+(i+1)*erate*del_t)/(1+i*erate*del_t)
    scale_y = scale_z = 1.0/math.sqrt(scale_x)
    #lam=(1+(i+1)*erate*del_t)/(1+i*erate*del_t)#erate*del_t
    mymin.change_box(scale_x, scale_y, scale_z)
    #ioLAMMPS.writeLAMMPS_triclinic('after_changing_box.txt', mymin.xlo, mymin.xhi, mymin.ylo, mymin.yhi, mymin.zlo, mymin.zhi,mymin.xy, mymin.xz,mymin.yz,
                                           #mymin.atoms, mymin.bonds, atom_types, bond_types, mass, loop_atoms)
    #stop
    [e, Gamma] = mymin.fire_iterate(tol, max_itr, write_itr, 'log.txt')
    [pxx, pyy, pzz, pxy, pyz, pzx] = mymin.compute_pressure()
    fstr.write('%7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f\n' \
                                     %(mymin.xhi-mymin.xlo, mymin.yhi-mymin.ylo, mymin.zhi-mymin.zlo, 
                                  (mymin.xhi-mymin.xlo)/Lx0, e, e-BE0, pxx, pyy, pzz, pxy, pyz, pzx)) 
    fstr.flush()

    ##dist = mymin.bondlengths()
    ##flen.write('%7.4f  %7.4f  %7.4f\n'%((mymin.xhi-mymin.xlo)/Lx0, np.mean(dist[:,3])/N, np.max(dist[:,3])/N))
    ##flen.flush()
   
    if((i+1)%wrt_step==0): 
      filename = 'restart_network_%d.txt' %(i+1)
      ioLAMMPS.writeLAMMPS(filename, mymin.xlo, mymin.xhi, mymin.ylo, mymin.yhi, mymin.zlo, mymin.zhi,
                                           mymin.atoms, mymin.bonds, atom_types, bond_types, mass, loop_atoms)

    if(i > begin_break):
      # U0, tau, del_t, pflag, index
      [t, n_bonds_init, n_bonds_final] = mymin.KMCbondbreak(U0, tau, del_t, 0, i+1) 
      fkmc.write('%7.4f  %5i  %5i\n'%((mymin.xhi-mymin.xlo)/Lx0, n_bonds_init, n_bonds_final))
      fkmc.flush()
    H_string = " ".join([f"{x:.8f}" for x in mymin.H.flatten()])
    ite=i+1
    fH.write(f"{ite} {H_string}\n")
    fH.flush()
    append_pdb_frame(output_file, mymin.atoms, mymin.bonds, mymin.H, np.array([mymin.xlo, mymin.ylo, mymin.zlo]), ite)

#---------------------------------#
#     Final Network Properties    #
#---------------------------------#
[e, Gamma] = mymin.fire_iterate(tol, max_itr, write_itr, 'log.txt')
[pxx, pyy, pzz, pxy, pyz, pzx] = mymin.compute_pressure()
fstr.write('%7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f\n' \
                                 %(mymin.xhi-mymin.xlo, mymin.yhi-mymin.ylo, mymin.zhi-mymin.zlo, 
                              (mymin.xhi-mymin.xlo)/Lx0, e, e-BE0, pxx, pyy, pzz, pxy, pyz, pzx)) 
fstr.flush()

##dist = mymin.bondlengths()
##flen.write('%7.4f  %7.4f  %7.4f\n'%((mymin.xhi-mymin.xlo)/Lx0, np.mean(dist[:,3])/N, np.max(dist[:,3])/N))
##flen.flush()

fkmc.write('%7.4f  %5i  %5i\n'%((mymin.xhi-mymin.xlo)/Lx0, n_bonds_init, n_bonds_final))
fkmc.flush()
fH.flush()

filename = 'restart_network_%d.txt' %(i+1)
ioLAMMPS.writeLAMMPS(filename, mymin.xlo, mymin.xhi, mymin.ylo, mymin.yhi, mymin.zlo, mymin.zhi,
                                       mymin.atoms, mymin.bonds, atom_types, bond_types, mass, loop_atoms)

fstr.close()
##flen.close()
fkmc.close()
fH.close()