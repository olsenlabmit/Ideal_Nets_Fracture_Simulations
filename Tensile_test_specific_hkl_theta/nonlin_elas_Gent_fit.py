## Fit the stress-strain curve with the Gent model #####
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

# 1. Define the Exponential (Fung) Model
def exponential_hardening(lam, A, B):
    """
    Fung-type exponential model.
    lam: stretch ratio (lambda)
    A:   Scale parameter (stress units)
    B:   Stiffening rate (dimensionless)
    """
    # Equation: sigma = A * (exp(B * strain) - 1)
    # We use (lam - 1) to represent strain
    return A * (np.exp(B * (lam - 1)) - 1)

# 2. Input Your Data Here

def get_target_index(y_data):
    """
    Identifies the index of the first peak. 
    If data is monotonic (no peaks found), returns the index of the max value.
    """
    y = np.array(y_data)
    
    # 1. Try to find peaks
    # height=0 ensures we don't pick up negative peaks if that matters to you
    peaks, _ = find_peaks(y)
    
    # 2. Check if any peaks were found
    if len(peaks) > 0:
        # Return the index of the very first peak found
        return peaks[0]
    else:
        # 3. If monotonic (no local maxima), return index of global max
        return np.argmax(y)
        

def Gent_model(lam, E, Jm):
    """
    Fung-type exponential model.
    lam: stretch ratio (lambda)
    A:   Scale parameter (stress units)
    B:   Stiffening rate (dimensionless)
    """
    # Equation: sigma = A * (exp(B * strain) - 1)
    # We use (lam - 1) to represent strain
    return (lam**2-1/lam)*(E/(3*(1-(lam**2+2/lam-3)/Jm)))##A * (np.exp(B * (lam - 1)) - 1)

def Neo_Hook_model(lam, E):
    """
    Fung-type exponential model.
    lam: stretch ratio (lambda)
    A:   Scale parameter (stress units)
    B:   Stiffening rate (dimensionless)
    """
    # Equation: sigma = A * (exp(B * strain) - 1)
    # We use (lam - 1) to represent strain
    return (lam**2-1/lam)*(E/(3))##*(1-(lam**2+2/lam-3)/Jm)))##A * (np.exp(B * (lam - 1)) - 1)
    
# 2. Input Your Data Here

data=np.genfromtxt('mean_ss_data.txt')
e_rate=5
del_t=0.002 
lam=data[0:,0]*e_rate*del_t+1
y=data[0:,1]
##print(np.where(y==np.max(y)))
#Jm=lam[np.where(y==np.max(y))[0][0]]**2-3
#print('Jm',Jm)
#x=(1/(1-(lam**2+2/lam-3)/Jm))

##print(((lam**2+2/lam-3)/Jm))
#print(x,y)

###### Neo-Hookean fitting in linear regime lam=1 to 1.1 ################33
idx=int(0.1/(e_rate*del_t))## upto lam=1.1  ##get_target_index(y) ##int(np.where(y==np.max(y))[0][0])
print(idx)

idx=[i for i in range(0,idx)][0:]
lam_data=lam[idx]
sigma_data=y[idx]

lam_max=np.max(lam_data)
#Jm_lam_max=lam_max**2+2/lam_max-3
#print('lam_max=',lam_max, ' Jm_lam_max=',Jm_lam_max)

# 3. Fit the Model
# Initial guesses (p0) are important for exponentials. 
# A is usually small, B is usually between 1 and 10.
initial_guess = [0.1] 

try:
    popt, pcov = curve_fit(Neo_Hook_model, lam_data, sigma_data, p0=initial_guess)
    E_fit = popt[0]
except RuntimeError:
    print("Error: Optimization failed. Try adjusting the initial_guess p0=[A, B].")
    A_fit, B_fit = 1, 1

# 4. Calculate R-squared (Goodness of Fit)
residuals = sigma_data - Neo_Hook_model(lam_data, *popt)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((sigma_data - np.mean(sigma_data))**2)
r_squared = 1 - (ss_res / ss_tot)
print('r_squared',r_squared)
print('E_fit from Nwo-Hookean',E_fit)

E_fixed=E_fit


############# now fit with Gent model with E_fixed from Neo-Hookean fitting ##################

idx=get_target_index(y) ##int(np.where(y==np.max(y))[0][0])
print(idx)

idx=[i for i in range(0,idx)][0:]
lam_data=lam[idx]
sigma_data=y[idx]

lam_max=np.max(lam_data)
Jm_lam_max=lam_max**2+2/lam_max-3
print('lam_max=',lam_max, ' Jm_lam_max=',Jm_lam_max)

# 3. Fit the Model
# Initial guesses (p0) are important for exponentials. 
# A is usually small, B is usually between 1 and 10.
initial_guess = [50]

def Gent_Jm_only(lam, Jm):
    return Gent_model(lam, E_fixed, Jm)

try:
    popt, pcov = curve_fit(Gent_Jm_only, lam_data, sigma_data, p0=initial_guess)
    Jm_fit = popt[0]
    print('Jm_fit',Jm_fit)
except RuntimeError:
    print("Error: Optimization failed. Try adjusting the initial_guess p0=[A, B].")
    A_fit, B_fit = 1, 1

# 4. Calculate R-squared (Goodness of Fit)
residuals = sigma_data - Gent_Jm_only(lam_data, *popt)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((sigma_data - np.mean(sigma_data))**2)
r_squared = 1 - (ss_res / ss_tot)
print('r_squared',r_squared)
# 5. Plot Results
lam_smooth = np.linspace(min(lam_data), max(lam_data), 100)
sigma_smooth = Gent_Jm_only(lam_smooth, *popt)

plt.figure(figsize=(7, 5))
plt.scatter(lam_data, sigma_data, color='black', label='Experimental Data')
plt.plot(lam_smooth, sigma_smooth, 'r-', linewidth=2, label=f'Fit (E (Neo Hookean)={E_fixed:.2f}  Jm (Gent) ={Jm_fit:.2f})')

plt.title(f"Gent_model  Fit ($R^2={r_squared:.4f}$)")
plt.xlabel(r"Stretch Ratio ($\lambda$)")
plt.ylabel("Stress")
plt.legend()
plt.grid(True, alpha=0.3)

# Display Parameters on Plot
text_str = '\n'.join((
    
    f'E = {E_fixed:.4f}',
    f'Jm = {Jm_fit:.4f}'))##r'$\sigma = A(e^{B(\lambda-1)} - 1)$',
plt.text(0.05, 0.75, text_str, transform=plt.gca().transAxes, 
         bbox=dict(facecolor='white', alpha=0.8))

plt.savefig('Gent_fit')

print(f"Fitted E from Neo-Hookean: {E_fixed:.5f}")
print(f"Fitted Jm: {Jm_fit:.5f}")

lam_m=np.sqrt(Jm_fit+3)## approx
print('Strain_hardening_parameter lam_m=',lam_m)


np.savetxt('exp_fit_E_Jm_lam_max.txt',np.transpose(np.array([E_fixed,Jm_fit, lam_max])))



redo=False
if(Jm_fit<Jm_lam_max):
  print('PROBLEM!! Jm_fit<Jm_lam_max; Jm_fit= ',str(Jm_fit),' Jm_lam_max= ',str(Jm_lam_max))
  redo=True
  lam_max=np.sqrt(Jm_fit+3)-1

#stop


if(redo==True):
  idx=int((lam_max-1)/(e_rate*del_t))#[i for i in range(0,idx)][0:]
  print(idx)

  idx=[i for i in range(0,idx)][0:]
  lam_data=lam[idx]
  sigma_data=y[idx]

  lam_max=np.max(lam_data)
  Jm_lam_max=lam_max**2+2/lam_max-3
  print('lam_max=',lam_max, ' Jm_lam_max=',Jm_lam_max)

# 3. Fit the Model
# Initial guesses (p0) are important for exponentials. 
# A is usually small, B is usually between 1 and 10.
  initial_guess = [40] 

  try:
    popt, pcov = curve_fit(Gent_Jm_only, lam_data, sigma_data, p0=initial_guess)
    Jm_fit = popt[0]
  except RuntimeError:
    print("Error: Optimization failed. Try adjusting the initial_guess p0=[A, B].")
    A_fit, B_fit = 1, 1

  # 4. Calculate R-squared (Goodness of Fit)
  residuals = sigma_data - Gent_Jm_only(lam_data, *popt)
  ss_res = np.sum(residuals**2)
  ss_tot = np.sum((sigma_data - np.mean(sigma_data))**2)
  r_squared = 1 - (ss_res / ss_tot)
  
  print('r_squared',r_squared)

  # 5. Plot Results
  lam_smooth = np.linspace(min(lam_data), max(lam_data), 100)
  sigma_smooth = Gent_Jm_only(lam_smooth, *popt)

  plt.figure(figsize=(7, 5))
  plt.scatter(lam_data, sigma_data, color='black', label='Experimental Data')
  plt.plot(lam_smooth, sigma_smooth, 'r-', linewidth=2, label=f'Exp Fit (E={E_fit:.2f}  Jm={Jm_fit:.2f})')

  plt.title(f"Gent_model  Fit ($R^2={r_squared:.4f}$)")
  plt.xlabel(r"Stretch Ratio ($\lambda$)")
  plt.ylabel("Stress")
  plt.legend()
  plt.grid(True, alpha=0.3)

# Display Parameters on Plot
  text_str = '\n'.join((
    
    f'E = {E_fit:.4f}',
    f'Jm = {Jm_fit:.4f}'))##r'$\sigma = A(e^{B(\lambda-1)} - 1)$',
  plt.text(0.05, 0.75, text_str, transform=plt.gca().transAxes, 
         bbox=dict(facecolor='white', alpha=0.8))

  plt.savefig('Gent_fit')

  print(f"Fitted E (Neo Hookean): {E_fixed:.5f}")
  print(f"Fitted Jm (Gent): {Jm_fit:.5f}")

  lam_m=np.sqrt(Jm_fit+3)## approx
  print('Strain_hardening_parameter lam_m=',lam_m)


  np.savetxt('exp_fit_E_Jm_lam_max.txt',np.transpose(np.array([E_fit,Jm_fit, lam_max])))