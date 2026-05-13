"""
@author: david
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from sympy import symbols, diff, lambdify, solve, Matrix
from scipy.stats import linregress


k_b = 1
m=1
T = 1.5
beta = float(1/T)
a = 4.5
b=1
lambda_3x = -3.6
lambda_4x = 0.8
lambda_4y = 1
g = 0.3
f=0
runs = 200
Ensembles = 50

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

"""Determining of an approapriate sink position"""

Minima = test[0]
stable = Minima[1]
x_stable, y_stable = float(stable[0]), float(stable[1])
V_stable = V(a,b,lambda_3x, lambda_4x, lambda_4y, g,f, x_stable, y_stable)

V_s = V(a,b,lambda_3x, lambda_4x, lambda_4y, g,f, x_s, y_s)

V_meta = V(a,b,lambda_3x, lambda_4x, lambda_4y, g,f, x_meta, y_meta)


from scipy.optimize import brentq
V_s = V(a,b,lambda_3x, lambda_4x, lambda_4y, g,f, x_s, y_s)
def equation(x, a, lambda_3x, lambda_4x, V_s, T):
    return (a/2)*x**2 + (lambda_3x/6)*x**3 + (lambda_4x/24)*x**4 - (V_s - 7*T)
upper = x_s + 0.1
while equation(upper, a, lambda_3x, lambda_4x, V_s, T) * equation(x_s, a, lambda_3x, lambda_4x, V_s, T) > 0:
    upper += 0.1
# solve with constraint x > x_b
x_sink = brentq(equation, x_s, upper, args=(a, lambda_3x, lambda_4x, V_s, T))
print(f"x_sink = {x_sink:.6f}")
print(f'x_b = {x_s:.6f}')

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

def Rates(escape_time, gamma, runs, lambda_plus):
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

def Rate_Stats(escape_matrix, Ensembles, gamma, lambda_plus, runs):
  R_linreg, E_linreg = np.zeros(Ensembles), np.zeros(Ensembles)

  for i in range(Ensembles):
    escape_time = filter_nan(escape_matrix[i,:])
    R_linreg[i], E_linreg[i] = Rates(escape_time, gamma,runs, lambda_plus)

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
damping = np.load('damping_array.npy')
Theory = np.zeros(len(damping))
lambda_array = np.zeros(len(damping))
for i in range(len(damping)):
    lambda_array[i] = lambda_plus_finder(trace_w_s, det_w_s, damping[i])
for i in range(len(damping)):
    Theory[i] = Theory_LO(damping[i], det_w_s, det_w_m, lambda_array[i], beta, E_b)

R, E = np.zeros(len(damping)), np.zeros(len(damping))

for i in range(len(damping)):
    mat = np.load(f'escape_gamma_{i}.npy')
    R[i],E[i] = Rate_Stats(mat, Ensembles, damping[i], lambda_array[i], runs)
# %%
'Determining the chi^2 of the fit, for gamma>2'
ind = np.searchsorted(damping, 2)
R1 = R[ind:]
E1 = E[ind:]
T = Theory[ind:]
N = len(damping[ind:])

chi_sq = 1/N * np.sum((R1-T)**2/E1**2)
# %%

'Adding the plot'
plt.figure(figsize = (12,8))
plt.errorbar(damping, R, yerr = E, color = 'red', fmt = '.', label = 'Numerical')
plt.plot(damping,Theory, color = 'green', label = rf'LO Theory Rate, $\chi^2$ = {chi_sq:.2f}')
plt.xlabel(r'$\gamma$')
plt.ylabel(r'$\Gamma$')
plt.legend()

