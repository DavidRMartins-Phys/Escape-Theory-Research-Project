
import numpy as np
import matplotlib.pyplot as plt
from numba import njit

#Global parameters
k_B =1
m=1
T = 2
beta = 1/(T*k_B)
#Potential Parameters
a = 18
b = -12.5
c = 1.8

#ensure values below match the ADA script
runs = 10000
Ensembles = 50
#import of damping array
damping = np.load('damping.npy')
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
'This analysis includes the outputs above for multiple temperatures, namely 2,3,4' 
#nans input in the arrays by hand, considering the outliers. 
T_array = np.array([2,3,4]) 
A_array = np.array(([3.96292698e-03,4.49119582e-03,np.nan,np.nan,0.00456362],\
                   [1.98976106e-02,0.02310031,0.02331222,np.nan,0.02200481],\
                       [0.04471448,0.05246773,0.0533674,0.05255551,0.05173328]))
B_array = np.array(([1.83976093e+00,1.04463941e+00,np.nan,np.nan,0.46406465],\
                   [1.91220247e+00,1.09762117,0.76608386,np.nan,0.4683383],\
                       [1.9622996,1.13835581,0.80434731,0.60899796,0.49040941]))
C_array = np.array(([3.11578772e-04,5.55185570e-04,np.nan,np.nan,0.00112653],\
                   [1.21467495e-03,0.00216192,0.00303328,np.nan,0.00450524],\
                       [0.00237906,0.004317,0.00601055,0.00755529,0.00889057]))
Initial_conds = A_array+C_array
Theory_array = np.zeros((len(T_array), len(damping)))
# %%


for i in range(len(T_array)):
    T = T_array[i]
    for j in range(len(damping)):
        gamma = damping[j]
        Theory_array[i,j] = Theory_LO(gamma, omega_b_sq, omega_M_sq, T, E_b)
Diff = (Theory_array - Initial_conds)/Theory_array * 100
print(abs(Diff))
print(np.nanmean(abs(Diff)))
print(np.nanstd(abs(Diff)))
fig, (ax1, ax2) = plt.subplots(1,2,figsize = (8,6), sharey=True)
for i in range(len(T_array)):
    ax1.scatter(damping, B_array[i,:], label = f'T = {T_array[i]}')
ax1.legend()
plt.figure()
for i in range(len(damping)):
    ax2.scatter(T_array,B_array[:,i], label = f'damping = {damping[i]}' )
ax2.legend()
ax1.set_xlabel(r'$\gamma$')
ax2.set_xlabel(r'$T$')
ax1.set_ylabel('Fitted Variable B')
plt.tight_layout()


# %%
'Ansatz, given the small T dependence and clear 1/gamma dependence is that\
    B = T/alpha*gamma'
alpha_array = np.zeros((len(T_array), len(damping)))
for i in range(len(T_array)):
    for j in range(len(damping)):
        B = B_array[i,j]
        if not np.isnan(B):
            alpha_array[i,j] = 1/(B * damping[j])
        else:
            alpha_array[i,j] = np.nan
print(alpha_array)
            
# %%
exponents = np.linspace(0.5,1, 100)
alpha_prop= np.zeros((len(T_array), len(damping)))
standarddev = np.zeros(len(exponents))
mean_alpha = np.zeros(len(exponents))
for p in range(len(exponents)):
    exponent = exponents[p]
    for i in range(len(T_array)):
        for j in range(len(damping)):
            B = B_array[i,j]
            if not np.isnan(B):
                alpha_array[i,j] = 1/(B * damping[j]**exponent)
            else:
                alpha_array[i,j] = np.nan
    mean_alpha[p] = np.nanmean(alpha_array)
    standarddev[p] = np.nanstd(alpha_array)
#finding the minimum std index:
ind = np.argmin(standarddev)
#choosing alpha and exponent for minimum std:
exponent_min = exponents[ind]
alpha= mean_alpha[ind]
plt.rcParams.update({'font.size': 13})
fig, (ax1, ax2) = plt.subplots(2,figsize = (6,4.5), sharex = True)
ax1.plot(exponents, mean_alpha)
ax1.hlines(alpha, min(exponents), max(exponents), color = 'red', linestyle = '--', label=rf'$\langle\alpha\rangle = $ {alpha:.2f}')
ax1.vlines(exponent_min, min(mean_alpha), max(mean_alpha),color = 'red', linestyle = '--', label = rf'$n$ in $B = 1/\alpha\gamma^n =${exponent_min:.2f}')
ax2.plot(exponents, standarddev)
ax2.hlines(min(standarddev), min(exponents), max(exponents),color = 'red', linestyle = '--', label = rf'minimum $\sigma = ${min(standarddev):.2f}')
ax2.vlines(exponent_min, min(standarddev), max(standarddev),color = 'red', linestyle = '--')
ax1.set_ylabel(r' $\langle \alpha\rangle$')
ax2.set_ylabel(r' $\sigma$')
ax2.set_xlabel(r'Exponent of $\gamma$')
ax1.legend()
ax2.legend()
plt.tight_layout()
# %%
alpha_std = min(standarddev)
internal_error_n = abs(alpha_std / alpha) * 12**-0.5
n_pos = np.zeros((len(T_array), len(damping)))
for i in range(len(T_array)):
    for j in range(len(damping)):
        n_pos[i,j] = -np.log(alpha * B_array[i,j]) / np.log(damping[j])
std_n = np.nanstd(n_pos)
avg_n = np.nanmean(n_pos)
external_error_n = std_n * 12**-0.5
error_n = (internal_error_n**2 + external_error_n**2)**0.5
print(rf'$n$ = {avg_n:.2f} \pm {error_n:.2f}')
