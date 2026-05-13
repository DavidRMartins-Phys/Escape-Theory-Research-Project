# -*- coding: utf-8 -*-
"""
Created on Fri May  1 13:42:53 2026

@author: david
"""

import numpy as np
from numba import njit, prange
from sympy import symbols, diff, lambdify, solve, Matrix

k_b = 1
m=1
T = 0.8
beta = 1/T
a = 4.5
b=1
lambda_3x = -3.6
lambda_4x = 0.8
lambda_4y = 1
g = 0.3
f=0

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

"""## Sampling of initial conditions

Develop the MH algorithm in 2d for Boltzmann distribution sampling

In the gaussian approximation, the two coordinates are independent. Therefore, sample x,y, p_x and p_y from gaussian distibutions and then have the acceptance probability for the position coordinates based on the potential properties.
"""

@njit
def Boltzmann_Sampling(a,b,lambda_3x,lambda_4x,lambda_4y,g,f, beta, x_meta, y_meta, x_s, w_meta, runs):
  #setting up the arrays
  x_initial = np.zeros(runs)
  y_initial = np.zeros(runs)

  #Standard deviations for gaussian distributions
  sigma_x = 2*(beta*w_meta[0,0]*m)**-0.5
  sigma_y=2*(beta*w_meta[1,1]*m)**-0.5
  sigma_p = (m/beta)**0.5 #The momenta have the same standard deviation

  #Starting point for metropolis Hasting as the metastable well (removes burn-in requirement)
  x_proposal, y_proposal = x_meta, y_meta
  i=0
  while i <runs:
    x_prop = np.random.normal(x_proposal, sigma_x)
    y_prop = np.random.normal(y_proposal, sigma_y)
    if x_prop< x_s:
      log_accept = -beta * (V(a,b,lambda_3x, lambda_4x, lambda_4y, g,f, x_prop, y_prop)-V(a,b,lambda_3x, lambda_4x, lambda_4y, g,f, x_proposal, y_proposal))
      if np.log(np.random.uniform(1e-30,1))<log_accept:
        x_proposal = x_prop
        y_proposal = y_prop

        x_initial[i] = x_proposal
        y_initial[i] = y_proposal
        i+=1
  px_initial = np.random.normal(0, sigma_p, runs)
  py_initial = np.random.normal(0, sigma_p, runs)

  return x_initial, y_initial, px_initial, py_initial

"""## The Synchronous Leapfrog in 2d

For the integrator, require multiple steps:


*   Evolution step (complete)
*   Core Evolution loop, which takes damping, t_step and intial position arrays as input (complete)
*   Outer Evolution function, which determines the timestep and initial conditions to be used. (complete)
*   Need to introduce an arbitrary maxtime as placeholder, since I dont have the theoretical rate yet (complete)
*   Need a way of determining the sink position, based on the backreaction probability having log(back) ~ -7*beta
    Decided to scap the idea of determining a sink position, instead comparing the probability of backreaction directly with e^-7 β. (complete)
"""

@njit(fastmath = True)
def Step(x, y, px, py, t_step, exp_term, x_noise, y_noise, dVdx, dVdy):
  #intial momentum updates
  px_half = px - t_step*dVdx
  py_half = py - t_step * dVdy
  #Position updates
  x = x + 2*t_step*px_half
  y = y + 2*t_step*py_half
  #Momentum noise updates
  px_half = px_half*exp_term + x_noise
  py_half = py_half*exp_term + y_noise
  #Update of gradient components
  dVdx = V_x(a,lambda_3x,lambda_4x,g,x,y)
  dVdy = V_y(b, lambda_4y, g, x,y)
  #final momentum update
  px = px_half - t_step*dVdx
  py = py_half - t_step * dVdy
  return x,y,px,py, dVdx, dVdy


@njit(parallel=True, fastmath = True)
def Evolution_core(x_initial, y_initial, px_initial, py_initial, runs, gamma):
  t_step = 1e-3 #need to add adaptive timestep here!
  delta_t = 2*t_step
  maxtime = 10**8 * delta_t
  #Defining the variables for the step
  exp_term = np.exp(-gamma * delta_t)
  noise_scale = ((1 - exp_term**2) * T)**0.5
  #setting up the escape time array
  escape_time = np.full(runs, np.nan)
  for i in prange(runs):
    #sampling of noise values for the entire runtime:
    batchsize = 50000
    Noise_x=noise_scale * np.random.normal(0,1, batchsize)
    Noise_y = noise_scale * np.random.normal(0,1,batchsize)
    possible = np.nan
    x = x_initial[i]
    y = y_initial[i]
    px = px_initial[i]
    py = py_initial[i]
    time = 0.0
    prev_x = x
    j=0
    dVdx = V_x(a,lambda_3x,lambda_4x,g,x,y)
    dVdy = V_y(b, lambda_4y, g, x,y)
    while time<maxtime and x<x_sink:
      if j>=batchsize:
        Noise_x=noise_scale * np.random.normal(0,1, batchsize)
        Noise_y = noise_scale * np.random.normal(0,1,batchsize)
        j=0
      x_noise = Noise_x[j]
      y_noise = Noise_y[j]
      x,y,px,py, dVdx, dVdy = Step(x, y, px, py, t_step, exp_term, x_noise, y_noise, dVdx, dVdy)
      time += delta_t
      if prev_x < x_s and x >= x_s:
        possible = time
      if x<x_s and prev_x>=x_s: #recrosses and resets the escape time. Without this, if it crosses and recrosses (never reaching the sink) an escape time is still stored, which is not correct.
        possible = np.nan
      prev_x = x #stores previous x and compares it to the new one, in order to update or reject the update
      j+=1
    escape_time[i] = possible
  return escape_time


@njit
def Evolution(a,b,lambda_3x, lambda_4x, lambda_4y, g,f, runs, gamma, x_s, y_s, x_meta, y_meta, w_meta, T, x_sink):
  #initial conditions
  x_initial, y_initial, px_initial, py_initial = Boltzmann_Sampling(a,b,lambda_3x,lambda_4x,lambda_4y,g, beta, x_meta, y_meta, x_s, w_meta, runs)
  escape_time = Evolution_core(x_initial, y_initial, px_initial, py_initial, runs, gamma)
  return escape_time

@njit()
def Escape_matrix(Ensembles,a,b,lambda_3x, lambda_4x, lambda_4y, g,f, runs, gamma, x_s, y_s, x_meta, y_meta, w_meta, T, x_sink ):
    matrix = np.zeros((Ensembles, runs))
    for i in range(Ensembles):
        matrix[i,:] = Evolution(a,b,lambda_3x, lambda_4x, lambda_4y, g,f, runs, gamma, x_s, y_s, x_meta, y_meta, w_meta, T, x_sink)
    return matrix


Ensembles = 50
runs = 200
damping = np.linspace(0.01, 15, 150)
np.save('damping_array.npy', damping)
gamma_num = len(damping)

for i in range(gamma_num):
    mat = Escape_matrix(Ensembles, a,b,lambda_3x, lambda_4x, lambda_4y,g,f,damping[i], x_s,y_s,x_meta, y_meta, w_meta, T,x_sink)
    np.save(f"escape_gamma_{i}.npy", mat)
    print(f'{(i+1)/gamma_num * 100:.2f} % Complete')