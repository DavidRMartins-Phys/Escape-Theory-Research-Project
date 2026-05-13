
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
T = 2.5
beta = 1/(T*k_B)
#Potential Parameters
a = 18
b = -12.5
c = 1.8

#ensure values below match the ADA script
runs = 500
ensemble_num = 50

# %%
"""Need to change the function names to simplify this. Also change the sampling to the Boltzmann using the MH algortihm, rather than the Kramers distribution"""

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

# %%


"""Cell below determines an appropriate sink position based on a return probability after crossing the barrier of e^-7"""

from scipy.optimize import brentq
V_b = V(a,b,c,x_b,0)
def equation(x, a, b, c, V_b, T):
    return (a/2)*x**2 + (b/3)*x**3 + (c/4)*x**4 - (V_b - 7*T)
upper = x_b + 1.0
while equation(upper, a, b, c, V_b, T) * equation(x_b, a, b, c, V_b, T) > 0:
    upper += 0.1
# solve with constraint x > x_b
x_sink = brentq(equation, x_b, upper, args=(a, b, c, V_b, T))

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

def Rates(escape_time, gamma):
  #determination of P_meta
  t_max = 2/Theory_LO(gamma, omega_b_sq, omega_M_sq, T, E_b) #to make averaging across multiple runs valid. Beforehand has an array that dependent on the maximum escape time
  t_array = np.linspace(0, t_max,500)
  #determining P_meta
  sorted_times = np.sort(escape_time)
  escaped_number = np.searchsorted(sorted_times, t_array)
  P_meta = np.maximum(1 - escaped_number / runs, 1e-10)
  log_P = np.log(P_meta)

  #Rate from mean escape time
  Mean_tau = (len(escape_time))/runs * np.mean(escape_time)
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

def Rate_Stats(escape_matrix, Ensembles, gamma):
  R_mean, E_mean = np.zeros(Ensembles), np.zeros(Ensembles)
  R_filter, E_filter = np.zeros(Ensembles), np.zeros(Ensembles)
  R_linreg, E_linreg = np.zeros(Ensembles), np.zeros(Ensembles)

  for i in range(Ensembles):
    escape_time = filter_nan(escape_matrix[i,:])
    R_mean[i], E_mean[i], R_filter[i], E_filter[i], R_linreg[i], E_linreg[i] = Rates(escape_time, gamma)

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
'Calculating the theoretical rates in 1d'
damping_array = np.load('damping array.npy')

TheoryLO = Theory_LO(damping_array, omega_b_sq, omega_M_sq, T, E_b)
TheoryNLO = Theory_NLO(damping_array, omega_b_sq, omega_M_sq, T, E_b)
# %%
'Loading of datasets and extracting the rates and errors'
gamma_num = len(damping_array)
R_mean, E_mean, R_filter, E_filter, R_linreg, E_linreg = np.zeros(gamma_num),np.zeros(gamma_num),\
    np.zeros(gamma_num),np.zeros(gamma_num),np.zeros(gamma_num),np.zeros(gamma_num)

for i in range(gamma_num):
    matrix = np.load(f"escape_gamma_{i}_T_2.5.npy")
    R_mean[i], E_mean[i], R_filter[i], E_filter[i], R_linreg[i], E_linreg[i] = Rate_Stats(matrix, ensemble_num, damping_array[i])
# %%
"Determination of chi^2 for all methods"
ind = np.searchsorted(damping_array, 2)
R1 = R_mean[ind:]
E1 = E_mean[ind:]
R2 = R_filter[ind:]
E2 = E_filter[ind:]
R3 = R_linreg[ind:]
E3 = E_linreg[ind:]

R_array = np.array([R1, R2, R3])
E_array =np.array([E1, E2, E3])

T1 = TheoryNLO[ind:]
T2 = TheoryLO[ind:]

N = len(damping_array[ind:])

chi_LO = np.zeros(3)
chi_NLO = np.zeros(3)
for i in range(3):
    chi_LO[i] = 1/N *np.sum((R_array[i,:]- T2)**2 / E_array[i,:]**2)
    chi_NLO[i] = 1/N * np.sum((R_array[i,:]- T1)**2 / E_array[i,:]**2)

chi_mean = chi_NLO[0]
chi_filter = chi_NLO[1]
chi_linreg = chi_NLO[2]
# %%
plt.rcParams.update({'font.size': 12})
'Adding the plots, and saving them with specific values named (runs, ensembles, temperature'
plt.figure(figsize=(9,4))
plt.errorbar(damping_array, R_mean, yerr = E_mean, fmt = 'y.' ,label = 'Rate from mean escape time')
plt.errorbar(damping_array, R_filter, yerr = E_filter,fmt = 'b.', label = 'Rate from Savitzky-Golay filter')
plt.errorbar(damping_array, R_linreg, yerr = E_linreg,fmt = 'r.', label = 'Rate from linear regression')
plt.plot(damping_array, TheoryNLO, color = 'green', label = 'Theoretical NLO Rate')
plt.plot(damping_array, TheoryLO,color = 'darkgreen', linestyle = '-', label = 'Theoretical LO Rate')
plt.xlabel(r'$\gamma$')
plt.ylabel(r'$\Gamma(\gamma)$')
plt.legend()
plt.savefig(f"rates_T{T}_runs{runs}_ens{ensemble_num}.png", dpi=150, bbox_inches='tight')

# %%


'Adding the triple plot'
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (10,3), sharey = True)
#plots of theory
ax1.plot(damping_array, TheoryNLO, color = 'green', label = '_nolegend_')
ax1.plot(damping_array, TheoryLO,color = 'darkgreen', linestyle = '-', label = '_nolegend_')
ax2.plot(damping_array, TheoryNLO, color = 'green', label='_nolegend_')
ax2.plot(damping_array, TheoryLO,color = 'darkgreen', linestyle = '-', label='_nolegend_')
ax3.plot(damping_array, TheoryNLO, color = 'green',label='_nolegend_')
ax3.plot(damping_array, TheoryLO,color = 'darkgreen', linestyle = '-', label='_nolegend_')

ax1.errorbar(damping_array, R_mean, yerr = E_mean, fmt = 'y.' ,label=rf'$\langle\tau\rangle$:{chi_mean:.2f}/{chi_LO[0]:.2f}(LO)')
ax2.errorbar(damping_array, R_filter, yerr = E_filter,fmt = 'b.', label=rf'SG:{chi_filter:.2f}/{chi_LO[1]:.2f}(LO)')
ax3.errorbar(damping_array, R_linreg, yerr = E_linreg,fmt = 'r.', label=rf'Linreg:{chi_linreg:.2f}/{chi_LO[2]:.2f}(LO)')

ax1.set(xlabel=r'$\gamma$', ylabel=r'$\Gamma$')
ax2.set(xlabel=r'$\gamma$')
ax3.set(xlabel=r'$\gamma$')
ax1.legend()
ax2.legend()
ax3.legend()
plt.tight_layout()
plt.savefig(f"triple_plot_T{T}_runs{runs}_ens{ensemble_num}.png")

