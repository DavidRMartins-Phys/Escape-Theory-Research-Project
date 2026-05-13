
"""
@author: david
"""


import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng()
from scipy.signal import savgol_filter
from numba import njit
from scipy.stats import linregress
#Global parameters
k_B =1
m=1

#Potential Parameters
a = 18
b = -12.5
c = 1.8

#ensure values below match the ADA script
runs = 100
ensemble_num = 25
# %%
"""## General Potential Properties"""

@njit
def V(a,b,c,x, degree):
  if degree == 0:
    V=1/2 * a*x**2+ 1/3*b*x**3+1/4*c*x**4
  elif degree ==1:
    V = a*x + b*x**2 + c*x**3
  elif degree ==2:
    V = a + 2*b*x + 3 *c * x**2
  return V

@njit
def Barrier(a,b,c):
  "This function returns the position of the barrier, and its height. Also returns the position of the metastable well, which\
  if a>0 is at x=0. Throughout, assume this is true. Outputs in order x-meta; x_b; E_b; omega_M; omega_b. "
  e2 = 1/(2*c)*(-b+(b**2-4*a*c)**0.5)
  e3 = 1/(2*c)*(-b-(b**2-4*a*c)**0.5)
  Ce2 = V(a,b,c,e2, 2)
  Ce3 = V(a,b,c,e3, 2)
  omega_M = a
  if Ce2>0:
    BH = V(a,b,c,e3, 0)
    omega_b = 1 * abs(Ce3)
    return 0, e3, BH, omega_M, omega_b
  else:
    BH = V(a,b,c,e2,0)
    omega_b = 1 * abs(Ce2)
    return 0, e2, BH, omega_M, omega_b 
# %%
x_meta, x_b, E_b, omega_M_sq, omega_b_sq = Barrier(a,b,c) #potential properties

#definition of lambda_3 and lambda_4
lambda_3 = 1/m * (2*b + 6*c*x_b)
lambda_4= 6* c

# %%

@njit
def filter_nan(arr):
    count = 0 #array filter that I had before via array indexing does not work with jit. Manual filter instead
    for v in arr:
        if not np.isnan(v):
            count += 1
    result = np.empty(count)
    j = 0
    for v in arr:
        if not np.isnan(v):
            result[j] = v
            j += 1
    return result
# %%


"""Cell below combines the 3 rate extraction methods into 1 function, in order to use the same escape array for the calculation.

The statistical rate function then determines the mean rate for all ensembles, along with the total error on the value.
"""

def Rates(escape_time, gamma, T):
  #determination of P_meta
  t_max = 2/Theory_LO(gamma, omega_b_sq, omega_M_sq, T, E_b) #to make averaging across multiple runs valid. Beforehand has an array that dependent on the maximum escape time
  t_array = np.linspace(0, t_max,500)
  #determining P_meta
  sorted_times = np.sort(escape_time)
  escaped_number = np.searchsorted(sorted_times, t_array)
  P_meta = 1 - escaped_number / runs
  log_P = np.log(P_meta)

  #Rate from mean escape time
  Mean_tau = np.mean(escape_time)
  std_tau = np.std(escape_time)
  Rate_from_mean = Mean_tau**-1
  Error_from_mean = std_tau * (Mean_tau**2 * runs**0.5)**-1 #from report

  #Calculation of Rate as a function of time using a Savitzky-Golay filter
  Gamma = - savgol_filter(log_P, 60, polyorder = 3, deriv = 1, delta= t_array[1] - t_array[0])
  #average rate as a function of time
  Rate_from_filter = np.mean(Gamma)
  std_filter = np.std(Gamma)
  Error_from_filter = std_filter / len(Gamma)**0.5


  #Calculation of rate from linreg fit
  start_idx = len(log_P)//5 #skipping first fifth of data
  end_idx = len(log_P)*4//5 #removes last fifth of data. Therefore only work with meaningful middle 60%
  slope, _,_,_,Error_from_linreg= linregress(t_array[start_idx:end_idx], log_P[start_idx:end_idx])
  Rate_from_linreg = -slope

  return Rate_from_mean, Error_from_mean, Rate_from_filter, Error_from_filter, Rate_from_linreg, Error_from_linreg
# %%

def Rate_Stats(escape_matrix, Ensembles, gamma, T):
  R_mean, E_mean = np.zeros(Ensembles), np.zeros(Ensembles)
  R_filter, E_filter = np.zeros(Ensembles), np.zeros(Ensembles)
  R_linreg, E_linreg = np.zeros(Ensembles), np.zeros(Ensembles)

  for i in range(Ensembles):
    escape_time = filter_nan(escape_matrix[i,:])
    R_mean[i], E_mean[i], R_filter[i], E_filter[i], R_linreg[i], E_linreg[i] = Rates(escape_time, gamma,T)

  #Calculation of mean rates
  R_mean_avg, R_filter_avg, R_linreg_avg = np.mean(R_mean), np.mean(R_filter), np.mean(R_linreg)

  #Mean internal errors
  E_mean_internal = 1/Ensembles * (np.sum(E_mean**2))**0.5
  E_filter_internal = 1/Ensembles * (np.sum(E_filter**2))**0.5
  E_linreg_internal = 1/Ensembles * (np.sum(E_linreg**2))**0.5

  #SEM of the rates (etrenal error)
  SEM_mean = np.std(R_mean)/Ensembles**0.5
  SEM_filter = np.std(R_filter)/Ensembles**0.5
  SEM_linreg = np.std(R_linreg)/Ensembles**0.5

  #Combination of internal and external errors
  Error_mean = (SEM_mean**2 + E_mean_internal**2)**0.5
  Error_filter = (SEM_filter**2 + E_filter_internal**2)**0.5
  Error_linreg = (SEM_linreg**2 + E_linreg_internal**2)**0.5

  return R_mean_avg, Error_mean, R_filter_avg, Error_filter, R_linreg_avg, Error_linreg
# %%
"""## Theoretical Rates"""

#Leading order rate:
@njit
def Theory_LO(gamma, omega_b_sq, omega_M_sq, T, E_b):
    Value = ((gamma**2 /4 + omega_b_sq)**0.5 - gamma/2)*(omega_M_sq/omega_b_sq)**0.5 *np.exp(-E_b/T)/(2*np.pi)
    return Value

#NLO Rate with 2-loop corrections
@njit
def Theory_NLO(gamma, omega_b_sq, omega_M_sq, T, E_b):
    C = gamma/2 + (gamma**2/4 + omega_b_sq)**0.5
    F = 6330*C**9 - 34431*C**8*gamma - 12086*C**8 + 78837*C**7*gamma**2 + 42747*C**7*gamma - 98286*C**6*gamma**3 - 59117*C**6*gamma**2 + 7936*C**6 + 71880*C**5*gamma**4 + 39849*C**5*gamma**3 - 26624*C**5*gamma - 30759*C**4*gamma**5 - 13037*C**4*gamma**4 + 36752*C**4*gamma**2 + 7113*C**3*gamma**6 + 1644*C**3*gamma**5 - 26608*C**3*gamma**3 - 684*C**2*gamma**7 + 10596*C**2*gamma**4 - 2184*C*gamma**5 + 180*gamma**6
    f = k_B*T*gamma**2/(8*m*(2*C-gamma)**2)
    I1 = lambda_4/omega_b_sq**2
    I2A = lambda_3**2/(9*omega_b_sq**3)
    I2B = F/((3*C-2*gamma)**2*(3*C-gamma)**2*(4*C-3*gamma)*(4*C-gamma))
    Gamma_0 = Theory_LO(gamma, omega_b_sq, omega_M_sq, T, E_b)
    Correction = 1- f*(I1 - I2A*I2B)
    Value = Gamma_0*Correction
    return Value
# %%
damping_array = np.load('damping array.npy')
T_array = np.load('T_array.npy')
Ratio = np.load('Ratio_array.npy')

#Set up the arrays for the rate and error.
Rate = np.zeros((len(T_array), len(damping_array)))
Error = np.zeros((len(T_array), len(damping_array)))
TheoryNLO = np.zeros((len(T_array), len(damping_array)))
TheoryLO = np.zeros((len(T_array), len(damping_array)))
for i in range(len(T_array)):
    for j in range(len(damping_array)):
        mat = np.load(f"escape_gamma_{j}_temp_{i}.npy")
        #using the linear regression method
        _,_,_,_,Rate[i,j], Error[i,j]= Rate_Stats(mat, ensemble_num, damping_array[j], T_array[i])
        TheoryNLO[i,j] = Theory_NLO(damping_array[j], omega_b_sq, omega_M_sq, T_array[i], E_b)
        TheoryLO[i,j] = Theory_LO(damping_array[j],omega_b_sq, omega_M_sq, T_array[i], E_b)
# %%
'Want to add the chi^2 value of each one onto the plot. Rather than plttong against temperature, plot against\
    inverse ratio. Add a plot of chi^2 vs inverse ratio as well.'
ind = np.searchsorted(damping_array, 2)
R = Rate[:, ind:]
E = Error[:,ind:]
Theory = TheoryNLO[:, ind:]
chi_sq = np.zeros(len(T_array))
N = len(damping_array[ind:])

for i in range(len(T_array)):
    chi_sq[i] = np.sum((R[i,:]- Theory[i,:])**2 / E[i,:]**2) / N
# %%
    
Ratio_inv = 1 / Ratio
# %%
Deviation = np.zeros((len(T_array),len(R[i,:])))
Errorbars = np.zeros((len(T_array),len(R[i,:])))
for i in range(len(T_array)):
    Deviation[i,:] = abs((R[i,:] - Theory[i,:])/Theory[i,:] * 100)
    Errorbars[i,:] = E[i,:]/Theory[i,:] * 100

# %%
#plotting the first two temperatures together:
plt.rcParams.update({'font.size': 12})
'Adding the triple plot'
fig, ([ax1, ax2],[ax3, ax4], [ ax5, ax6]) = plt.subplots(3, 2, figsize = (12,12))
colours = np.array(['darkblue', 'cornflowerblue', 'lightskyblue', 'lightsalmon','tomato','red'])

#Set 1
ax1.plot(damping_array, TheoryNLO[0,:], color = colours[0], label = rf'Theoretical NLO Rate for T={T_array[0]}')
ax2.plot(damping_array, TheoryNLO[1,:], color = colours[1], label = rf'Theoretical NLO Rate for T={T_array[1]}')
ax1.errorbar(damping_array, Rate[0,:], yerr = Error[0,:], color = colours[0], fmt = '.', label = rf'Numerical Rate for T = {T_array[0]}, $\chi^2$ = {chi_sq[0]:.2f}')
ax2.errorbar(damping_array, Rate[1,:], yerr = Error[1,:], color = colours[1], fmt = '.', label = rf'Numerical Rate for T = {T_array[1]}, $\chi^2$ = {chi_sq[1]:.2f}')
#Set 2
ax3.plot(damping_array, TheoryNLO[2,:], color = colours[2], label = rf'Theoretical NLO Rate for T={T_array[2]}')
ax4.plot(damping_array, TheoryNLO[3,:], color = colours[3], label = rf'Theoretical NLO Rate for T={T_array[3]}')
ax3.errorbar(damping_array, Rate[2,:], yerr = Error[2,:], color = colours[2], fmt = '.', label = fr'Numerical Rate for T = {T_array[2]}, $\chi^2$ = {chi_sq[2]:.2f}')
ax4.errorbar(damping_array, Rate[3,:], yerr = Error[3,:], color = colours[3], fmt = '.', label = fr'Numerical Rate for T = {T_array[3]}, $\chi^2$ = {chi_sq[3]:.2f}')

#Set 3
ax5.plot(damping_array, TheoryNLO[4,:], color = colours[4], label = rf'Theoretical NLO Rate for T={T_array[4]}')
ax6.plot(damping_array, TheoryNLO[5,:], color = colours[5], label = rf'Theoretical NLO Rate for T={T_array[5]}')
ax5.errorbar(damping_array, Rate[4,:], yerr = Error[4,:], color = colours[4], fmt = '.', label = rf'Numerical Rate for T = {T_array[4]}, $\chi^2$ = {chi_sq[4]:.2f}')
ax6.errorbar(damping_array, Rate[5,:], yerr = Error[5,:], color = colours[5], fmt = '.', label = rf'Numerical Rate for T = {T_array[5]}, $\chi^2$ = {chi_sq[5]:.2f}')




ax1.set(xlabel = r'$\gamma$', ylabel = r'$\Gamma$')
ax2.set(xlabel = r'$\gamma$', ylabel = r'$\Gamma$')
ax3.set(xlabel = r'$\gamma$', ylabel = r'$\Gamma$')
ax1.legend()
ax2.legend()
ax3.legend()
ax4.set(xlabel = r'$\gamma$', ylabel = r'$\Gamma$')
ax5.set(xlabel = r'$\gamma$', ylabel = r'$\Gamma$')
ax6.set(xlabel = r'$\gamma$', ylabel = r'$\Gamma$')
ax4.legend()
ax5.legend()
ax6.legend()
plt.tight_layout()
plt.savefig(f"triple_plot_T_runs{runs}_ens{ensemble_num}.png")
# %%
plt.figure(figsize = (12,8))
for i in range(len(T_array)):
    plt.scatter(damping_array[ind:], Deviation[i,:], color = colours[i],label = fr'T = {T_array[i]}, $\chi^2$ = {chi_sq[i]:.2f}')
    avg_dev = np.mean(Deviation[i,:])
    plt.plot(damping_array[ind:], avg_dev * np.ones(len(damping_array[ind:])), color = colours[i], label= fr'Mean deviation is {avg_dev:.2f} %')
plt.xlabel('$\gamma$')
plt.ylabel('Deviation from Theory (in %)')
plt.legend()
plt.savefig('deviation_vs_damping')
# %%
slope, intercept,_,_,_ = linregress(Ratio_inv, chi_sq)
plt.figure()
plt.scatter(Ratio_inv, chi_sq,color = 'orange', label = f'$\chi^2(k_BT/E_b)$')
plt.xlabel(f'Inverse Ratio $(k_B T / E_b)$')
plt.ylabel(f'$\chi^2$')
plt.plot(Ratio_inv,intercept + slope*Ratio_inv, color = 'darkblue', linestyle = '--', label = 'Linear regression line fit' )
plt.legend()
plt.savefig('chi_squared_vs_temp')
#depending on how plot looks, might need to make more temperature runs.
# %%
plt.figure(figsize = (8,5))
difference = (TheoryLO - TheoryNLO)/TheoryLO*100
for i in range(len(T_array)):
    plt.plot(damping_array, difference[i],color = colours[i], label = fr'$T =${T_array[i]} ')
plt.legend()
plt.xlabel(f'$\gamma$')
plt.ylabel(r'$(\Gamma_{\mathrm{LO}} - \Gamma_{\mathrm{NLO}})/\Gamma_{\mathrm{LO}}$ in $\%$')
plt.tight_layout()
plt.show()
