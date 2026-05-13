
import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from sympy import symbols, diff, lambdify, solve, Matrix
from scipy.stats import linregress


k_b = 1
m=1
a = 4.5
b=1
lambda_3x = -3.6
lambda_4x = 0.8
lambda_4y = 1
g = 0.3
f=0
runs = 50
Ensembles = 10

"""##Properties of the potential"""

#Potenial in 2d
#Using sympy for the differentiation:
def Potential(a,b,lambda_3x,lambda_4x,lambda_4y,g,f,x_val,y_val, x_deg, y_deg):
  #definition of symbolic potential
  x,y = symbols('x,y')
  V_sym = a*0.5 * x**2 + b *0.5*y**2 + lambda_3x/6 * x**3 + lambda_4x/24 * x**4 + lambda_4y/24*y**4 + 0.5*f*x*y**2+ g*0.25 * x**2*y**2
  #differentiation according to degrees specified
  fun = diff(diff(V_sym, x, x_deg), y, y_deg)
  #conversion to python from symbolic
  V_function = lambdify((x,y), fun)
  #evaluation at the specified x and y vals
  V_val = V_function(x_val,y_val)
  return V_val

#defining an njit compatible potential function:
@njit
def V(a,b,lambda_3x,lambda_4x,lambda_4y,g,f,x,y):
  Value = a*0.5 * x**2 + b *0.5*y**2 + lambda_3x/6 * x**3 + lambda_4x/24 * x**4 + lambda_4y/24*y**4 + g*0.25 * x**2*y**2 + 0.5 * f*x*y**2
  return Value
#defining njit compatible first derivatives of the potential, for the integrator
@njit
def V_x(a,lambda_3x,lambda_4x,g,f,x,y):
  Value = a*x + lambda_3x*0.5*x**2 + lambda_4x/6 * x**3 + g*0.5 * x*y**2 + 0.5*f*y**2
  return Value

@njit
def V_y(b,lambda_4y,g,f,x,y):
  Value = b*y + lambda_4y/6 * y**3 + g*0.5 * x**2*y + f*x*y**2
  return Value

def Critical_point(a,b,lambda_3x,lambda_4x,lambda_4y,g,f):
  x,y = symbols('x,y')
  V_sym = a*0.5 * x**2 + b *0.5*y**2 + lambda_3x/6 * x**3 + lambda_4x/24 * x**4 + lambda_4y/24*y**4 + g*0.25 * x**2*y**2+0.5*f*x*y**2
  dVdx = diff(V_sym, x)
  dVdy = diff(V_sym, y)
  critical_points = solve([dVdx, dVdy], [x,y])
  real_points = []
  for point in critical_points:
    if point[0].is_real and point[1].is_real:
      real_points.append(point)
  return real_points

def Point_type(a,b,lambda_3x,lambda_4x,lambda_4y,g,f):
  x,y= symbols('x,y')
  V_sym = a*0.5 * x**2 + b *0.5*y**2 + lambda_3x/6 * x**3 + lambda_4x/24 * x**4 + lambda_4y/24*y**4 + g*0.25 * x**2*y**2+0.5*f*x*y**2
  Hessian = Matrix([[diff(V_sym,x,2), diff(diff(V_sym, x),y)], [diff(diff(V_sym, y), x), diff(V_sym, y, 2)]])
  critical = Critical_point(a,b,lambda_3x, lambda_4x, lambda_4y, g,f)
  minima = []
  maxima = []
  saddle= []
  for point in critical:
    Hessian_val = Hessian.subs([(x,point[0]), (y,point[1])])
    determinant = Hessian_val.det()
    Value = Hessian_val[0,0]
    if determinant > 0:
      if  Value> 0:
        minima.append(point)
      if Value<0:
        maxima.append(point)
    elif  determinant<0:
      saddle.append(point)
    else:
      print(f'Point {point} is undetermined')
  return minima, maxima, saddle
#in the potential type we are consideering, the maxima are irrelevant. Should get 2 minima and 1 saddlepoint, ideally.


def Barrier(a,b,lambda_3x,lambda_4x,lambda_4y,g,f):
  minima, _, saddle = Point_type(a,b,lambda_3x,lambda_4x,lambda_4y,g,f) #both are lists of tuples
  minimum1 = minima[0]
  minimum2 = minima[1]
  sad = saddle[0]
  #extraction of critical point coords
  x_min1, y_min1 = minimum1[0], minimum1[1]
  x_min2, y_min2 = minimum2[0], minimum2[1]
  x_sad, y_sad = sad[0], sad[1]

  #Determination of which is stable and which is metastable:
  V_min1 = Potential(a,b,lambda_3x,lambda_4x,lambda_4y,g,f,x_min1,y_min1,0,0)
  V_min2 = Potential(a,b,lambda_3x,lambda_4x,lambda_4y,g,f,x_min2,y_min2,0,0)
  V_sad = Potential(a,b,lambda_3x,lambda_4x,lambda_4y,g,f,x_sad,y_sad,0,0)
  if V_min1>V_min2:
    x_meta, y_meta = x_min1, y_min1
    x_stable, y_stable = x_min2, y_min2
  else:
    x_meta, y_meta = x_min2, y_min2
    x_stable, y_stable = x_min1, y_min1

  #Metastable and saddlepoints as tuples of the coords:
  metastable = (x_meta, y_meta)
  saddlepoint = (x_sad, y_sad)
  #Determination of Barrier Height:
  E_b = V_sad - Potential(a,b,lambda_3x,lambda_4x,lambda_4y,g,f,x_meta,y_meta,0,0)

  #For the omega matrices at the saddlepoint and metastable well
  w_meta = np.zeros((2,2))
  w_saddle = np.zeros((2,2))
  for i in range(2):
    for j in range(2):
      x_degree = 2 - i- j
      y_degree = i+j
      w_meta[i,j] = float(1/m * Potential(a,b,lambda_3x,lambda_4x,lambda_4y,g,f,x_meta,y_meta,x_degree,y_degree))
      w_saddle[i,j] =float(1/m * Potential(a,b,lambda_3x,lambda_4x,lambda_4y,g,f,x_sad,y_sad,x_degree,y_degree))

  return E_b, metastable, saddlepoint, w_meta, w_saddle

test = Point_type(a,b,lambda_3x, lambda_4x, lambda_4y, g,f)

test2 = Barrier(a,b,lambda_3x, lambda_4x, lambda_4y, g,f)

E_b, metastable, saddlepoint, w_meta, w_saddle = Barrier(a,b,lambda_3x,lambda_4x,lambda_4y,g,f)

x_s, y_s = float(saddlepoint[0]), float(saddlepoint[1])
x_meta, y_meta = float(metastable[0]), float(metastable[1])
E_b = float(E_b)
"""Determining of an approapriate sink position"""

Minima = test[0]
stable = Minima[1]
x_stable, y_stable = float(stable[0]), float(stable[1])
V_stable = V(a,b,lambda_3x, lambda_4x, lambda_4y, g,f, x_stable, y_stable)

V_s = V(a,b,lambda_3x, lambda_4x, lambda_4y, g,f, x_s, y_s)

V_meta = V(a,b,lambda_3x, lambda_4x, lambda_4y, g,f, x_meta, y_meta)



@njit#(fastmath=True)
def filter_nan(arr):
    count = 0
    for v in arr:
        if v == v:
            count += 1
    result = np.empty(count)
    j = 0
    for v in arr:
        if v == v:
            result[j] = v
            j += 1
    return result

def Rates(escape_time, gamma, runs, lambda_plus, beta):
  #determination of P_meta
  t_max = 2/Theory_LO(gamma, det_w_s, det_w_m, lambda_plus, beta, E_b) #to make averaging across multiple runs valid. Beforehand has an array that dependent on the maximum escape time
  t_array = np.linspace(0, t_max,500)
  #determining P_meta
  sorted_times = np.sort(escape_time)
  escaped_number = np.searchsorted(sorted_times, t_array)
  P_meta = np.maximum(1 - escaped_number / runs, 1e-10)
  log_P = np.log(P_meta)

  #Calculation of rate from linreg fit
  start_idx = len(log_P)//5 #skipping first fifth of data
  end_idx = len(log_P)*4//5 #removes last fifth of data. Therefore only work with meaningful middle 60%
  slope, _,_,_,Error_from_linreg= linregress(t_array[start_idx:end_idx], log_P[start_idx:end_idx])
  Rate_from_linreg = -slope

  return Rate_from_linreg, Error_from_linreg
# %%

def Rate_Stats(escape_matrix, Ensembles, gamma, lambda_plus, runs, beta):
  R_linreg, E_linreg = np.zeros(Ensembles), np.zeros(Ensembles)

  for i in range(Ensembles):
    escape_time = filter_nan(escape_matrix[i,:])
    R_linreg[i], E_linreg[i] = Rates(escape_time, gamma,runs, lambda_plus, beta)

  #Calculation of mean rates
  R_linreg_avg = np.mean(R_linreg)

  #Mean internal errors
  E_linreg_internal = 1/Ensembles * (np.sum(E_linreg**2))**0.5

  #SEM of the rates (etrenal error)
  SEM_linreg = np.std(R_linreg)/Ensembles**0.5

  #Combination of internal and external errors
  Error_linreg = (SEM_linreg**2 + E_linreg_internal**2)**0.5

  return R_linreg_avg, Error_linreg

#defining a function that determines the determinant of any 2x2 matrix
@njit
def determinant(matrix):
  a, b, c, d = matrix[0,0], matrix[0,1], matrix[1,0], matrix[1,1]
  return a*d-b*c
@njit
def Trace(matrix):
  a,b,c,d  = matrix[0,0], matrix[0,1], matrix[1,0], matrix[1,1]
  return a+d
det_w_s = determinant(w_saddle)
det_w_m = determinant(w_meta)
trace_w_s = Trace(w_saddle)
@njit
def lambda_plus_finder(trace_w_s, det_w_s, gamma):
  root = 2*(trace_w_s**2 - 4 * det_w_s)**0.5
  term1 = gamma**2 - 2*trace_w_s
  frac = -gamma/2
  pos1 = frac + 0.5*(term1 - root)**0.5
  pos2 = frac - 0.5*(term1 - root)**0.5
  pos3 = frac + 0.5*(term1 + root)**0.5
  pos4 = frac - 0.5 * (term1 + root)**0.5
  if pos1>0:
    return pos1
  elif pos2>0:
    return pos2
  elif pos3>0:
    return pos3
  else:
    return pos4

@njit(fastmath = True)
def Theory_LO(gamma, det_w_s, det_w_m, lambda_plus, beta, E_b):
  f = lambda_plus/(2*np.pi) * np.exp(-E_b * beta) * (det_w_m/abs(det_w_s))**0.5
  return f
# %%
'extraction of damping array:'
damping = np.load('damping array.npy')
T_array = np.load('T_array.npy')
Theory = np.zeros((len(T_array),len(damping)))
lambda_array = np.zeros(len(damping))
for i in range(len(damping)):
    lambda_array[i] = lambda_plus_finder(trace_w_s, det_w_s, damping[i])

# %%
beta = 1/ T_array
for i in range(len(T_array)):
    for j in range(len(damping)):
        Theory[i, j] = Theory_LO(damping[j], det_w_s, det_w_m, lambda_array[j], beta[i], E_b)

R, E = np.zeros((len(T_array),len(damping))), np.zeros((len(T_array),len(damping)))
for i in range(len(T_array)):
    for j in range(len(damping)):
        mat = np.load(f'escape_gamma_{j}_temp_{i}.npy')
        R[i,j],E[i,j] = Rate_Stats(mat, Ensembles, damping[j], lambda_array[j], runs, beta[i])


# %%
'Want to add the chi^2 value of each one onto the plot. Rather than plttong against temperature, plot against\
    inverse ratio. Add a plot of chi^2 vs inverse ratio as well.'
ind = np.searchsorted(damping, 2)
R1 = R[:, ind:]
E1 = E[:,ind:]
Theory1 = Theory[:, ind:]
chi_sq = np.zeros(len(T_array))
N = len(damping[ind:])

for i in range(len(T_array)):
    chi_sq[i] = np.sum((R1[i,:]- Theory1[i,:])**2 / E1[i,:]**2) / N
# %%
#plotting the first two temperatures together:
plt.rcParams.update({'font.size': 12})
'Adding the triple plot'
fig, ([ax1, ax2],[ax3, ax4], [ ax5, ax6]) = plt.subplots(3, 2, figsize = (12,12))
colours = np.array(['darkblue', 'cornflowerblue', 'lightskyblue', 'lightsalmon','tomato','red'])

#Set 1
ax1.plot(damping, Theory[0,:], color = colours[0], label = rf'LO Rate for T={T_array[0]:.2f}')
ax2.plot(damping, Theory[1,:], color = colours[1], label = rf'LO Rate for T={T_array[1]:.2f}')
ax1.errorbar(damping, R[0,:], yerr = E[0,:], color = colours[0], fmt = '.', label = rf'Numerical Rate for T = {T_array[0]:.2f}, $\chi^2$ = {chi_sq[0]:.2f}')
ax2.errorbar(damping, R[1,:], yerr = E[1,:], color = colours[1], fmt = '.', label = rf'Numerical Rate for T = {T_array[1]:.2f}, $\chi^2$ = {chi_sq[1]:.2f}')
#Set 2
ax3.plot(damping, Theory[2,:], color = colours[2], label = rf'LO Rate for T={T_array[2]:.2f}')
ax4.plot(damping, Theory[3,:], color = colours[3], label = rf'LO Rate for T={T_array[3]:.2f}')
ax3.errorbar(damping, R[2,:], yerr = E[2,:], color = colours[2], fmt = '.', label = fr'Numerical Rate for T = {T_array[2]:.2f}, $\chi^2$ = {chi_sq[2]:.2f}')
ax4.errorbar(damping, R[3,:], yerr = E[3,:], color = colours[3], fmt = '.', label = fr'Numerical Rate for T = {T_array[3]:.2f}, $\chi^2$ = {chi_sq[3]:.2f}')

#Set 3
ax5.plot(damping, Theory[4,:], color = colours[4], label = rf'LO Rate for T={T_array[4]:.2f}')
ax6.plot(damping, Theory[5,:], color = colours[5], label = rf'LO Rate for T={T_array[5]:.2f}')
ax5.errorbar(damping, R[4,:], yerr = E[4,:], color = colours[4], fmt = '.', label = rf'Numerical Rate for T = {T_array[4]:.2f}, $\chi^2$ = {chi_sq[4]:.2f}')
ax6.errorbar(damping, R[5,:], yerr = E[5,:], color = colours[5], fmt = '.', label = rf'Numerical Rate for T = {T_array[5]:.2f}, $\chi^2$ = {chi_sq[5]:.2f}')




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
