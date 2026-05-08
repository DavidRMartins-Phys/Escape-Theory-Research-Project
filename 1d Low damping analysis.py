
import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng()
from numba import njit
from scipy.stats import linregress
from scipy.optimize import curve_fit
#Global parameters
k_B =1
m=1
T = 2 #set to match simulation
beta = 1/(T*k_B)
#Potential Parameters
a = 18
b = -12.5
c = 1.8

#ensure values below match simulation ones
runs = 10000
Ensembles = 50

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
#Leading order rate:
@njit
def Theory_LO(gamma, omega_b_sq, omega_M_sq, T, E_b):
    Value = ((gamma**2 /4 + omega_b_sq)**0.5 - gamma/2)*(omega_M_sq/omega_b_sq)**0.5 *np.exp(-E_b/T)/(2*np.pi)
    return Value

# %%

def Gamma_of_t(escape_time, gamma, window):
    #determination of P_meta
    t_max = 1/gamma #to make averaging across multiple runs valid. Beforehand has an array that dependent on the maximum escape time
    t_array = np.linspace(0, t_max,500)
    #determining P_meta
    sorted_times = np.sort(escape_time)
    escaped_number = np.searchsorted(sorted_times, t_array)
    P_meta = 1 - escaped_number / runs
    log_P = np.log(P_meta)
    index2 = np.arange(window//2 - 1, 500, window//2)
    index_array = np.concatenate(([0], index2))
    half = window // 2
    time= t_array[index_array]
    Rate_of_t = np.zeros(len(index_array))
    Error_in_rate = np.zeros(len(index_array))
    for i, idx in enumerate(index_array):
        lo, hi = max(0, idx - half), min(500, idx + half)
        slope, _, _, _, Error_in_rate[i] = linregress(t_array[lo:hi], log_P[lo:hi])
        Rate_of_t[i] = -slope
    return time, Rate_of_t, Error_in_rate

def Gamma_of_t_ensembles(escape_matrix,Ensembles, gamma, window=24):
    Rate_of_t_matrix = []
    Error_matrix = []
    for i in range(Ensembles):
      escape_time = filter_nan(escape_matrix[i,:])
      time, Rate_array, error_array = Gamma_of_t(escape_time, gamma, window)
      length = len(Rate_array)
      Rate_of_t_matrix.append(Rate_array)
      Error_matrix.append(error_array)
    Error_array = np.array(Error_matrix)
    rate_of_t_array=np.array(Rate_of_t_matrix)
      
    Gamma_t = np.zeros(length)
    Error_t = np.zeros(length)
    for i in range(length):
      Gamma_t[i] = np.mean(rate_of_t_array[:,i])
      Mean_internal_error = np.mean(Error_array[:,i])
      SEM_of_t = np.std(rate_of_t_array[:,i])/np.sqrt(Ensembles)
      Error_t[i] = (SEM_of_t **2 + Mean_internal_error**2)**0.5 
    return Gamma_t, Error_t, time
      
# %%

damping = np.load('damping.npy')
G_of_t = np.zeros((len(damping), 42))
for i in range(len(damping)):
    mat = np.load(f'escape_gamma_{i}.npy')
    G_of_t[i,:], _, t_array = Gamma_of_t_ensembles(mat, Ensembles, damping[i])

import matplotlib.cm as cm
colors = cm.coolwarm(np.linspace(0, 1, len(damping)))
plt.figure(figsize = (6,4))
def exponential(t, A, B, C):
    return A*np.exp(-B *t) + C
for i in range(5):
    plt.plot(t_array, G_of_t[i, :], color = colors[i], label = fr'$\gamma =${damping[i]:.2f}')
    y = Theory_LO(damping[i], omega_b_sq, omega_M_sq, T, E_b)
    print(fr'Theory is {y}')
    plt.hlines(y, 0, max(t_array), color = colors[i], linestyle = '--')
    bounds = ([0, 0, 0],        # lower: all positive
          [np.inf, np.inf, np.inf])  # upper: unconstrained
    fit_vals ,_= curve_fit(exponential, t_array, G_of_t[i], bounds = bounds, maxfev = 10000)
    print(fr'$\gamma$ ={damping[i]:.2f} yields fits {fit_vals}')
    plt.plot(t_array, exponential(t_array, *fit_vals), color = colors[i], linestyle = ':')
plt.legend()
plt.xlabel(r'$t$')
plt.ylabel(r'$\Gamma(t)$')

