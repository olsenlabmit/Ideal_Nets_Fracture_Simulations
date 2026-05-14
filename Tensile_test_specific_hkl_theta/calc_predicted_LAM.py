###### Calculate Lambda_predicted, from the analytical model developed in this work ##############
import math
import random
##import netgen

import shutil




import ioLAMMPS ##_netgen as ioLAMMPS
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

from sklearn.cluster import DBSCAN

import numpy as np
from scipy.optimize import brentq

min_val, max_val = np.loadtxt('min_max_val_N.txt')
bin_width=p.bin_width

edges = np.arange(min_val, max_val + bin_width, bin_width)




def chain_locking_function(LAM, theta, l_max, l_init):
    """
    The difference between current chain stretch and its limit.
    Solving for where this equals 0.
    """
    # 3D Incompressible kinematics: lambda = sqrt( L^2 cos^2 + 1/L sin^2 )
    term1 = (LAM**2) * (np.cos(theta)**2)
    term2 = (1 / LAM) * (np.sin(theta)**2)
    ##current_lambda = np.sqrt(term1 + term2)
    
    func = term1 + term2- (l_max/l_init)**2 ##lambda_max / lambda_0
    return func
    
    

def solve_extensibility(theta, l_max, l_init):
            #individual_limit = []
    
    ##for t, l_max, l_0 in zip(thetas, lambda_maxes, lambda_0s):
        # The chain must have some room to stretch (capacity > 1)
        if l_max / l_init <= 1.0:
            individual_limit=1.0 ##.append(1.0) # Already locked
            return individual_limit##continue
            
        try:
            # We search for Lambda between 1 (no stretch) and a reasonable upper bound
            # brentq is a highly robust root-finder
            sol = brentq(chain_locking_function, 1.0, 100.0, args=(theta, l_max, l_init))
            individual_limit=sol ##.append(sol)
        except ValueError:
            # If no solution found in range
            individual_limit=np.inf ##.append(np.inf)

    # The network extensibility is the MINIMUM of all individual limits
    ##network_limit = np.nanmin(individual_limits)
        return individual_limit #network_limit 
    
    
    


# This is just to check whether different jobs have different seeds
import shlex  # Make sure to import this at the top of your script
def angle_between_vector_and_plane(vector, plane_normal):
    """
    Calculates the angle (in degrees) between a vector and a plane.
    """
    # Ensure inputs are numpy arrays
    v = np.array(vector)
    n = np.array(plane_normal)
    
    # Calculate the dot product
    # Use abs() to handle the acute angle
    dot_product = np.abs(np.dot(v, n))
    
    # Calculate the magnitudes (norms) of the vectors
    v_magnitude = np.linalg.norm(v)
    n_magnitude = np.linalg.norm(n)
    
    # Prevent division by zero if either vector is zero
    if v_magnitude == 0 or n_magnitude == 0:
        return 0.0 # Or raise an error
        
    # Calculate the sine of the angle
    # sin(alpha) = |v . n| / (|v| * |n|)
    sin_alpha = dot_product / (v_magnitude * n_magnitude)
    
    # Clamp the value to [-1, 1] to avoid potential floating point errors
    # with np.arcsin
    sin_alpha = np.clip(sin_alpha, -1.0, 1.0)
    
    # Calculate the angle in radians and convert to degrees
    alpha_radians = np.arcsin(sin_alpha)
    alpha_degrees = np.degrees(alpha_radians)
    
    return alpha_degrees#alpha_radians#alpha_degrees
    

def angle_between_vector_and_tensile_dir(vector, tensile_dir):
    """
    Calculates the angle (in degrees) between a vector and tensile direction.
    """
    # Ensure inputs are numpy arrays
    v = np.array(vector)
    n = np.array(tensile_dir)
    
    # Calculate the dot product
    # Use abs() to handle the acute angle
    dot_product = np.abs(np.dot(v, n))
    
    # Calculate the magnitudes (norms) of the vectors
    v_magnitude = np.linalg.norm(v)
    n_magnitude = np.linalg.norm(n)
    
    # Prevent division by zero if either vector is zero
    if v_magnitude == 0 or n_magnitude == 0:
        return 0.0 # Or raise an error
        
    # Calculate the sine of the angle
    # cos(alpha) = |v . n| / (|v| * |n|)
    cos_alpha = dot_product / (v_magnitude * n_magnitude)
    
    # Clamp the value to [-1, 1] to avoid potential floating point errors
    # with np.arcsin
    cos_alpha = np.clip(cos_alpha, -1.0, 1.0)
    
    # Calculate the angle in radians and convert to degrees
    alpha_radians = np.arccos(cos_alpha)
    alpha_degrees = np.degrees(alpha_radians)
    
    return alpha_degrees #alpha_radians #alpha_degrees#alpha_radians#alpha_degrees
    
    
    
def get_origin_from_file(filename, ite):
    """
    Parses a multi-frame file.
    
    It looks for lines starting with "Lattice=" and checks
    if their "Time=" value matches 'ite'. If it does,
    it returns the "Origin=" coordinates from that line.
    
    Args:
        filename (str): The path to the input file.
        ite (int): The iteration number to check against.
        
    Returns:
        list: A list of float coordinates [x, y, z] if a match is found,
              otherwise None.
    """
    
    try:
        with open(filename, 'r') as f:
            for line in f:
                # 1. Check if the line is a header line
                if line.startswith("Lattice="):
                    
                    # 2. Parse this header line into a dictionary
                    data = {}
                    try:
                        # --- THIS IS THE FIX ---
                        # Use shlex.split to correctly handle spaces inside quotes
                        parts = shlex.split(line)
                        
                        for part in parts:
                            # Split only on the first '='
                            key, value = part.split('=', 1)
                            # shlex already handles stripping quotes
                            data[key] = value
                            
                    except Exception as e:
                        # Skip this malformed header line
                        print(f"Warning: Skipping malformed header: {line.strip()}. Error: {e}")
                        continue

                    # 3. Check the "Time" condition
                    try:
                        # Handle potential trailing characters like '.' in "Time=1656."
                        time_val_str = data['Time'].rstrip('."') 
                        time_val = int(time_val_str)
                    except (KeyError, ValueError):
                        # This header didn't have a valid 'Time' key, so we skip it
                        continue
                        
                    # 4. If Time matches, get and return the "Origin"
                    if time_val == ite:
                        try:
                            origin_str = data['Origin']
                            # Convert '0.0 0.0 0.0' to a list of floats [0.0, 0.0, 0.0]
                            origin_coords = [float(x) for x in origin_str.split()]
                            return origin_coords  # Success!
                        except (KeyError, ValueError, IndexError) as e:
                            print(f"Error: Time matched, but 'Origin' was invalid. Error: {e}")
                            return None # Found the right line, but it's broken

        # If we get here, the file ended without a match
        return None

    except FileNotFoundError:
        print(f"Error: File not found at {filename}")
        return None
        
        
def get_bondlength_idx(i):
            lnk_1 = mymin.bonds[i,2]-1
            lnk_2 = mymin.bonds[i,3]-1
            delr = mymin.atoms[lnk_1,:] - mymin.atoms[lnk_2,:]
            delr_0=delr.copy()
            
            #delr[0] = delr[0] - int(round(delr[0]/Lx))*Lx
            #delr[1] = delr[1] - int(round(delr[1]/Ly))*Ly
            #delr[2] = delr[2] - int(round(delr[2]/Lz))*Lz
            
            r1 = mymin.atoms[lnk_1,:]
            r2 = mymin.atoms[lnk_2,:]


            # fractional coords
            s = H_inv @ delr

            # wrap into [-0.5, 0.5)
            s_wrapped = s - np.round(s)

            # back to Cartesian
            delr= mymin.H @ s_wrapped

            #if(np.allclose(delr,delr_0)==False):
                #print(delr_0,delr)
                #print('PBC')
            

                 
            r = LA.norm(delr)
            r_arr[i]=r
            
            origin = np.array([mymin.xlo, mymin.ylo, mymin.zlo])
            #print('origin',origin)
          

            #origin = np.array([self.xlo, self.ylo, self.zlo])
            #frac = (self.atoms - origin) @ H_inv.T
          
            #atoms=self.atoms
            atoms_relative = r1 - origin
            s1 = (H_inv @ atoms_relative.T).T
            
            atoms_relative = r2 - origin
            s2 = (H_inv @ atoms_relative.T).T
            #s_wrapped=s1%1.0-s2%1.0
            
            #'''
            
            ####print('lnk_1',lnk_1,'lnk_2',lnk_2, 'delr_0',delr_0,'delr',delr, 's',s,'s_wrapped', s_wrapped,'r',r)
            ####print('mymin.atoms[lnk_1,:]',mymin.atoms[lnk_1,:],'mymin.atoms[lnk_2,:]',mymin.atoms[lnk_2,:])
            ####print( s1,s2,s1%1.0,  s2%1.0)
            ####print((mymin.H @ (s1%1.0).T).T+origin,(mymin.H @ (s2%1.0).T).T+origin)
            #'''
            yz_plane_normal = (1, 0, 0)
            tensile_dir = (1, 0, 0)
            v=delr
            angle = angle_between_vector_and_tensile_dir(v, tensile_dir)
            #angle_normal = angle_between_vector_and_plane(v, yz_plane_normal)
            #print(angle,angle_normal)
            #stop
            return delr,r, angle
ite_arr=[0]#,10,20,30,40,50,100,500,700]#np.arange(0,1600,100)

ite=0 ## analyze only the starting iteration
#nets_arr=['bmn','bod','boe','nbo-a','sgn','srs','srs-a','sxt','utb']
#hkl_theta_arr=[['010',0],['010',30],['010',45],['010',60],['010',90],['001',0],['001',30],['001',45],['001',60],['001',90],['110',0],['110',30],['110',45],['110',60],['110',90],['111',0],['111',30],['111',45],['111',60],['111',90]]

net_cnt=0

all_Jm_arr=[]
all_avg_order_param_arr=[]

'''
for net in nets_arr: 
        net_cnt=net_cnt+1
        r_avg=[]
        angle_avg=[]
        sine_angle_avg=[]
        avg_order_param=[]
        r_unique_all_ite=[]
        angle_unique_all_ite=[]
        
        non_lin_elas_Jm_arr=[]  ## exponential factor B
    
        for hkl_theta in hkl_theta_arr:
'''


#run_arr=np.arange(1,6)
if(True):
            

            netgen_flag = 0
            ## read in orthogonal axis alligned simulation box (lattice)
            if(netgen_flag==0):

                vflag = 0
                N = 12   
                print('--------------------------')   
                print('----Reading Network-------')   
                print('--------------------------')   
                
                #print('./cR3=3/'+net+'110/0/Run1/restart_network_'+str(ite)+'.txt')
                [xlo, xhi, ylo, yhi, zlo, zhi, n_atoms, n_bonds, atoms, bonds,
                        atom_types, bond_types, mass, loop_atoms] = ioLAMMPS.readLAMMPS_restart('./restart_network_'+str(ite)+'.txt',vflag)##"network_after_swelling_and_relax.txt",vflag)##"restart_network_"+str(ite)+".txt",vflag)##"network_after_swelling_only.txt",  vflag)
                
                print('xlo, xhi',xlo, xhi) 
                print('ylo, yhi',ylo, yhi) 
                print('zlo, zhi',zlo, zhi) 
                print('n_atoms', n_atoms) 
                print('n_bonds', n_bonds) 
                print('atom_types = ', atom_types) 
                print('bond_types = ', bond_types) 
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
                        atom_types, bond_types, mass, loop_atoms] = ioLAMMPS.readLAMMPS("network_after_swelling_and_relax.txt")#"restart_network_"+str(ite)+".txt", N, 0)


            else:
                print('Invalid network generation flag')


            Nb = N
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
            
            ### theoretical prediction of macroscopic extension derived from a scaling analysis- considering the point of chain lock-up to be the contour length


            [xlo,ylo,zlo]=get_origin_from_file('./full_trajectory_atoms_only_correct_box_orient.xyz', ite)
            #print('xlo,ylo,zlo',xlo,ylo,zlo)
            print('xlo,ylo,zlo',[xlo,ylo,zlo])
            mymin = Optimizer(atoms, bonds, xlo, xhi, ylo, yhi, zlo, zhi, K, r0, min_val,bin_width, 'Mao')
            print(mymin.xlo,mymin.ylo,mymin.zlo)
            hkl_=p.hkl

            H_data=np.loadtxt('./H_matrix', skiprows=1)
            H_flat=H_data[ite+2,1:]
            H=H_flat.reshape((3, 3))
            print(H)
            mymin.H=H
            H_inv=np.linalg.inv(H)
            r_arr=np.zeros(n_bonds)
            angle_arr=np.zeros(n_bonds)
            delr_arr=np.zeros((n_bonds,3))

            volume = np.abs(np.linalg.det(mymin.H))
                    #print('VOLUME',volume)
            inv_volume = 1.0 / volume

            print('volume',volume)
            #'''
            LAM_ind=np.zeros(n_bonds) ## 3cos^2theta-1 ## theta is with tensile axis  ## P2
            #align_costheta=np.zeros(n_bonds) ### cos(theta)
            #align_rcostheta=np.zeros(n_bonds)
            
            #align_cos2theta=np.zeros(n_bonds)
            
            for i in range(0, n_bonds):

                        delr,r, angle=get_bondlength_idx(i) ## this angle is with the noral to tensile axis
                        theta= math.radians(angle) ## in radians  #### ## this angle is with the tensile axis
                        
                        r_arr[i]=r
                        delr_arr[i,:]=delr
                        angle_arr[i]=angle
                        
                        k = int(bonds[i, 1])

                        # Option 1: Midpoint of the bin (best estimate)
                        N_recovered = (edges[k - 1] + edges[k]) / 2.0
                        
                        l_max=1.4*N_recovered#Nb 
                        
                        LAM_ind[i]=solve_extensibility(theta, l_max, r) ## gives the predicted LAM value for each chain
                        
                        
                        ##order_param[i]=0.5*(3*(np.cos(np.deg2rad(angle_with_tensile_axis)))**2-1)##(np.cos(np.deg2rad(angle_with_tensile_axis)))**2##
                        ##align_costheta[i]=delr[0]/r  ##np.cos(np.deg2rad(angle_with_tensile_axis))
                        ##align_rcostheta[i]=(delr[0])**2##r*np.cos(np.deg2rad(angle_with_tensile_axis))
                        ##align_cos2theta[i]=(r**2)*((delr[0]/r)**4)#3np.cos(np.deg2rad(angle_with_tensile_axis))**4
                        

            #print(r_arr)
            #print('min and max',max(r_arr),min(r_arr))
            ##print('r_arr,angle_arr',r_arr,angle_arr)
            ##print('avg_order_param',np.mean(order_param))
            #print(np.unique(angle_arr))##np.mean(order_param))##len(np.where(order_param!=0)[0]))
            
            print('LAM_ind',LAM_ind)
            print('max_LAM ',np.max(LAM_ind),' min_LAM ',np.min(LAM_ind))
            
            #stop
            
            LAM_network=np.min(LAM_ind) ## 
            r_avg=np.mean(r_arr)
            angle_avg=np.mean(angle_arr)
            
            np.savetxt('./predicted_LAM_network.txt',np.array([LAM_network]))
            
            #sine_angle_arr=np.sin(np.deg2rad(np.array(angle_arr)))
            #sine_angle_avg.append(np.mean(sine_angle_arr))
            
            ##avg_order_param=np.mean(order_param)
            
            ##avg_costheta=np.mean(align_costheta)
            ##avg_rcostheta=1/np.sqrt(np.mean(align_rcostheta))
            
            ##avg_cos2theta=np.mean(align_cos2theta)
            
            ##np.savetxt('./avg_P2.txt',np.array([avg_order_param]))
            ##np.savetxt('./avg_costheta.txt',np.array([avg_costheta]))
            ##np.savetxt('./avg_rcostheta.txt',np.array([avg_rcostheta]))
            
            ##np.savetxt('./avg_r2cos2theta.txt',np.array([avg_cos2theta]))
            
            
            
        