
"""
@author: david
"""


import numpy as np
from numba import njit, prange
#Global parameters
k_B =1
m=1
#Potential Parameters
a = 18
b = -12.5
c = 1.8

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
@njit
def Theory_LO(gamma, omega_b_sq, omega_M_sq, T, E_b):
    Value = ((gamma**2 /4 + omega_b_sq)**0.5 - gamma/2)*(omega_M_sq/omega_b_sq)**0.5 *np.exp(-E_b/T)/(2*np.pi)
    return Value
# %%


"""Cell below determines an appropriate sink position based on a return probability after crossing the barrier of e^-7"""

from scipy.optimize import brentq
V_b = V(a,b,c,x_b,0)
def equation(x, a, b, c, V_b, T):
    return (a/2)*x**2 + (b/3)*x**3 + (c/4)*x**4 - (V_b - 7*T)
upper = x_b + 1.0
while equation(upper, a, b, c, V_b, 3) * equation(x_b, a, b, c, V_b, 3) > 0:
    upper += 0.1
# solve with constraint x > x_b
x_sink = brentq(equation, x_b, upper, args=(a, b, c, V_b, 3))

# %%
@njit
def initial_sampling(a,b,c,runs, T, gamma, omega_b_sq, omega_M_sq):
  x_initial = np.empty(runs)
  beta = 1/T
  #std of gaussian distributions
  sigma_x = 2*(beta*omega_M_sq*m)**-0.5 #multiply the std by 2 to sample the distribution faster, removing burn-in requirement
  sigma_p = (m/beta)**0.5

  x_proposal = 0
  #Determination of x position from true Boltzmann
  i=0
  while i <runs:
    x_prop = np.random.normal(x_proposal, sigma_x)
    if x_prop<x_b:
      log_accept = -beta*(V(a,b,c,x_prop,0) - V(a,b,c,x_proposal,0)) #comparison between true boltzmann and gaussian approx
      if np.log(np.random.uniform(1e-300,1))<log_accept:
        x_proposal = x_prop
    x_initial[i] = x_proposal
    i+=1
  #momentum
  p_initial = np.random.normal(0, sigma_p, runs)
  return x_initial, p_initial
# %%
#the function below defines the adaptive timestep calculation, based on the value of gamma, ensuring clear separation of timescales and optimization of runtime.
@njit
def adaptive_t(gamma):
  relaxation_time = 1/gamma
  dt_base = 0.01
  dt_min = 1e-3
  t_step = 0.5 * min(dt_base, 0.01*relaxation_time)
  t_step = 0.5 * max(t_step, dt_min)
  return t_step

# %%
"""## Langevin Dynamics"""

@njit
def Step(p,x,exp_term, noise_scale, t_step):
    p_half = p - t_step * V(a, b, c, x, 1)
    x_new  = x + t_step*2 * p_half
    p_half = p_half * exp_term + noise_scale * np.random.normal()
    p_new  = p_half - t_step * V(a, b, c, x_new, 1)
    return x_new,p_new


@njit(parallel = True)
def Evolution(a,b,c,runs,gamma, x_b, omega_b_sq,T):
  SINK = x_sink
  #sampling of initial conditions
  x_initial, p_initial = initial_sampling(a,b,c,runs, T, gamma, omega_b_sq, omega_M_sq)
  t_step = adaptive_t(gamma)
  maxtime = 100/Theory_LO(gamma, omega_b_sq, omega_M_sq, T, E_b)
  escape_time = np.full(runs, np.nan) #sets up the escape time array with nan that gets replaced when it escapes.
  delta_t = 2*t_step
  exp_term = np.exp(-gamma * delta_t)
  noise_scale = np.sqrt((1 - exp_term**2) * T)
  for i in prange(runs):
      x = x_initial[i]
      p = p_initial[i]
      time = 0.0
      prev_x = x
      pos_esc = np.nan
      while x <= SINK and time < maxtime:
          x, p = Step(p, x, exp_term, noise_scale, t_step)
          time += delta_t
          if prev_x < x_b and x >= x_b:
              pos_esc = time
          if prev_x >=x_b and x<x_b:
              pos_esc = np.nan
          prev_x = x
      escape_time[i] = pos_esc
  return escape_time

@njit(parallel = True)
def escape_matrix(runs, Ensembles, gamma,T):
  escape = np.zeros((Ensembles, runs))
  for i in prange(Ensembles):
    escape[i,:] = Evolution(a,b,c,runs,gamma, x_b, omega_b_sq,T)
  return escape
# %%
end_gamma = 15 #change this depending on what we want as the final gamma
gamma_num = 50 #change this depending on how many damping values we want
damping = np.linspace(0.01, end_gamma, gamma_num)
#save the damping matrix, to remove chances of errors when statistics are being calculated (mismatch due to forgetting)
np.save('damping array.npy', damping)
#size of escape matrix per gamma
runs = 100
ensemble_num = 25

#Temperature and ratio arrays
T_array = np.linspace(1, 4, 15)
np.save('T_array.npy', T_array)
Ratio_array = E_b/T_array
np.save('Ratio_array.npy', Ratio_array)
#%%
for j in prange(len(T_array)):
    T = T_array[j]
    for i in range(gamma_num):
        mat = escape_matrix(runs, ensemble_num, damping[i],T)
        np.save(f"escape_gamma_{i}_temp_{j}.npy", mat)
        print(f'{(i+1)/gamma_num * (j+1)/15 * 100} % Complete')
    #need to add the specified directory when access ADA