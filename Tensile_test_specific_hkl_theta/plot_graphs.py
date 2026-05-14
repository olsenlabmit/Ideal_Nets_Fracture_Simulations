import io
import matplotlib
####matplotlib.use('Agg') 
from matplotlib import pyplot as plt
import numpy as np
import sys
import ioLAMMPS
from numpy import linalg as LA
import param as p
min_val, max_val = np.loadtxt('min_max_val_N.txt')
bin_width=p.bin_width
edges = np.arange(min_val, max_val + bin_width, bin_width)


mean_N=p.mean_N


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
          
          
          # For a given bin index k = int(bonds[i, 1]):
          k = int(chains[i, 1])

          # Option 1: Midpoint of the bin (best estimate)
          N_recovered = (edges[k - 1] + edges[k]) / 2.0
          
          
          meanr2=meanr2+(dist[i,3])**2/mean_N##(N_recovered**2/mean_N)

          


####          print(dist[i,3])
##          stop
##          print(((dist[i,3])**2)/(p.N_low*p.b_low**2))
          

    return meanr2



temps = []
with io.open("stress", mode="r") as f:
    next(f)
    for line in f:
        temps.append(line.split())

Lx=[float(i[0]) for i in temps]
Ly=[float(i[1]) for i in temps]
Lz=[float(i[2]) for i in temps]

lam=[i[3] for i in temps]
lam=[float(i) for i in lam]


FE=[i[4] for i in temps]
FE=[float(i) for i in FE] #free energy stored in chain
deltaFE=[i[5] for i in temps]
deltaFE=[float(i) for i in deltaFE]

st0=[i[6] for i in temps]
st0=[float(i) for i in st0]
st1=[i[7] for i in temps]
st1=[float(i) for i in st1]
st2=[i[8] for i in temps]
st2=[float(i) for i in st2]
st3=[i[9] for i in temps]
st3=[float(i) for i in st3]
st4=[i[10] for i in temps]
st4=[float(i) for i in st4]
st5=[i[11] for i in temps]
st5=[float(i) for i in st5]
factor=4.11
st0=np.array(st0)*factor
st1=np.array(st1)*factor
st2=np.array(st2)*factor
st3=np.array(st3)*factor
st4=np.array(st4)*factor
st5=np.array(st5)*factor
##st6=np.array(st6)

##stop


##size=sys.getsizeof(st1)
##factor=np.zeros((1,size))
##mylist = list(xrange(10))
##factor=np.zeros((1,size))
##a=st1+st2;
##a=[0.5*i for i in a]
stress=st0-0.5*(st1+st2)

vflag=0
[xlo, xhi, ylo, yhi, zlo, zhi, n_atoms, n_bonds, atoms, bonds,
           atom_types, bond_types, mass, loop_atoms] = ioLAMMPS.readLAMMPS_restart("network_final.txt",  vflag)
V_init=(xhi-xlo)*(yhi-ylo)*(zhi-zlo)

##cR3=p.cR3 ## dimless conc
##conc=cR3/(p.N*p.b**2)**1.5 #(chains/nm3)
##V=(n_bonds/conc)

##stress=(stress/V)*V_init



'''
plt.plot(lam,st0,label='pxx')
plt.plot(lam,st1,label='pyy')
plt.plot(lam,st2,label='pzz')
plt.plot(lam,st3,label='pxy')
plt.plot(lam,st4,label='pyz')
plt.plot(lam,st5,label='pzx')

plt.xlabel('lambda')
plt.ylabel('pressure/stress components')
plt.legend()

plt.figure()
plt.plot(lam,deltaFE,label='deltaFE')
plt.xlabel('lambda')
plt.ylabel('delta Free Energy')
plt.legend()
'''
#print(lam)
#print(deltaFE)

plt.figure()
xaxis=np.arange(0,len(Lx))#[(x)/Lx[0] for x in Lx]
plt.plot(xaxis,-stress,'b.-',label='stress')
plt.xlabel('lambda')
plt.ylabel('Stress (MPa)')
plt.legend()

file1=open("data.txt","w")
for i in range(len(xaxis)):
    file1.write("{:7} {:7} \n".format(xaxis[i],-stress[i]))

plt.savefig("stress_lambda_x.png")

area=np.trapz(-stress, xaxis,0.001)
file2=open("area.txt","w")
file2.write(str(area))


filename = "restart_network_0.txt"
##      file_path = os.path.join(directory, filename)
##      if not os.path.isdir(directory):
##         os.mkdir(directory)
####G=nx.Graph()
[xlo, xhi, ylo, yhi, zlo, zhi, n_atoms, n_bonds, atoms, bonds, 
      atom_types, bond_types, mass, loop_atoms] = ioLAMMPS.readLAMMPS_restart(filename, 0)#:#, G,folder)


mean_r2=meanr2_fun(n_bonds, bonds, atoms, (xhi-xlo), (yhi-ylo), (zhi-zlo)) ## this is already normalized by N
gamma=mean_r2/(n_bonds*p.b_low**2)
#print('gamma',gamma, 'sqrt(mean_r2)',np.sqrt(mean_r2/n_bonds),'sqrt(<R^2>)', np.sqrt(p.N)*p.b)
file3=open("gamma.txt","w")
file3.write(str(gamma))          
plt.show()
