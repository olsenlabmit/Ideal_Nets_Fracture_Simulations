cR3=3
desired_gamma=0.3
U0=10
hkl=[0, 1, 0]
angle_deg=0

#hkl=[1,1,0]
#angle_deg=45
#U0=10
#cR3=3



mean_N=12 ## mean N

sigma_N=2 ## std_dev in N distribution

bin_width=1.0  ## the bin width for N vaues histogram- sampled from Gaussian distribution
b=1.0

#N_low=N1
#N_high=N2
b_low=b
b_high=b
##bond_types=2 ## number of types of atoms
##perc_T2=50 ## percentage of chains that are of T2 typ2

#cR3=3##3 # dimless conc
##n_chains=10000

####a=9#8

####num_unit_cells=(a+1)**3
####tot_nodes=8*num_unit_cells # in each unit cell- only 14 nodes are associated with connections, rest will get repeated
####tot_chains=2*tot_nodes
####n_chains=tot_chains

##conc=cR3/(N*b**2)**1.5 #(chains/nm3)
##L=(n_chains/conc)**(1/3)
##C_mM=conc/0.6022 # conc in mM
##print('L',L)
##print('C_mM', C_mM)

# other things here are not required 
# will be modifying code to remove these
lam_max=30
del_t=0.002
erate=5



rho=3
K=1.0
tau=1.0


fit_param=1.0

E_b=1200.0

func=3

tol=0.01
max_itr = 100000
write_itr = 10000
wrt_step = 10 ####################

##########################
########################
##############

##############


#####
