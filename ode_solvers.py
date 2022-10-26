# ODE Solves
import numpy as np
from scipy import optimize

def solve_euler(f,tmesh,y0):
    """
    Forward Euler Method ODE Solver:
        - f: Right hand side of ODE f(t,y)
        - y0: Initial condition
        - tmesh: array of times
    """
    num_pts = len(tmesh)
    try:
        dim = len(y0)
    except TypeError:
        dim = 1
    soln = np.zeros((dim,num_pts))
    soln[:,0] =  y0
        
    for n in range(1,num_pts):
        h = tmesh[n] - tmesh[n-1]
        t = tmesh[n-1]
        soln[:,n] = soln[:,n-1] + h*f(t,soln[:,n-1]) 
    return soln

def heuns_2nd_order_rk(f, tmesh, y0):
    """
    Heun's 2nd Order Method ODE Solver:
        - f: Right hand side of ODE f(t,y)
        - y0: Initial condition
        - tmesh: array of times
    """
    num_pts = len(tmesh)
    try:
        dim = len(y0)
    except TypeError:
        dim = 1
    soln = np.zeros((dim,num_pts))
    soln[:,0] =  y0
    
    for n in range(1,num_pts):
        h = tmesh[n] - tmesh[n-1]
        t = tmesh[n-1]
        soln[:,n] = soln[:,n-1] + (h/2)*(f(t,soln[:,n-1]) + f(t+h, soln[:,n-1]+h*f(t,soln[:,n-1])))
    return soln
    
def heuns_3rd_order(f, tmesh, y0):
    """
    Heun's 3rd Order ODE Solver:
        - f: Right hand side of ODE f(t,y)
        - y0: Initial condition
        - tmesh: array of times
    """
    num_pts = len(tmesh)
    try:
        dim = len(y0)
    except TypeError:
        dim = 1
    soln = np.zeros((dim,num_pts))
    soln[:,0] =  y0
    
    # Set constants from Butcher table
    c2 = 1/3
    c3 = 2/3
    b1 = 1/4 
    b2 = 0
    b3 = 3/4
    a21 = 1/3
    a31 = 0
    a32 = 2/3
    
    for n in range(1,num_pts):
        h = tmesh[n] - tmesh[n-1]
        t = tmesh[n-1]
        k1 = f(t,soln[:,n-1])
        k2 = f(t + c2*h, soln[:,n-1]+h*a21*k1)
        k3 = f(t + c3*h, soln[:,n-1]+h*a31*k1 + h*a32*k2)
        soln[:,n] = soln[:,n-1] + h*(b1*k1 + b2*k2 + b3*k3)
    return soln

def backward_euler(f,tmesh,y0):
    """
    Backwards Euler Method ODE Solver:
        - f: Right hand side of ODE f(t,y)
        - y0: Initial condition
        - tmesh: array of times
    """
    num_pts = len(tmesh)  
    
    try:
        dim = len(y0)
    except TypeError:
        dim = 1
    soln = np.zeros((dim,num_pts))
    soln[:,0] =  y0   
    
    y_old = np.copy(y0)
    for n in range(1,num_pts):
        h = tmesh[n] - tmesh[n-1]
        t = tmesh[n]
        
        nonlin_func = lambda y: -y + y_old + h*f(t,y)
        soln_root = optimize.root(nonlin_func, y_old)
        soln[:,n] = soln_root.x       
        y_old = soln[:,n]
    return soln

def trapz(f,tmesh,y0):
    """
    Trapezoidal Method ODE Solver:
        - f: Right hand side of ODE f(t,y)
        - y0: Initial condition
        - tmesh: array of times
    """
    num_pts = len(tmesh)  
    try:
        dim = len(y0)
    except TypeError:
        dim = 1
    soln = np.zeros((dim,num_pts))
    soln[:,0] =  y0  
    y_old = np.copy(y0)
    for n in range(1,num_pts):
        h = tmesh[n] - tmesh[n-1]
        nonlin_func = lambda y: -y + y_old + 0.5*h*(f(tmesh[n],y) + f(tmesh[n-1],y_old))
        soln_root = optimize.root(nonlin_func, y_old)
        soln[:,n] = soln_root.x       
        y_old = soln[:,n]
    return soln

def rk4(f, tmesh, y0):
    """
    Runge-Kutta 4th Order ODE Solver:
        - f: Right hand side of ODE f(t,y)
        - y0: Initial condition
        - tmesh: array of times
    """
    num_pts = len(tmesh)
    try:
        dim = len(y0)
    except TypeError:
        dim = 1
    soln = np.zeros((dim,num_pts))
    soln[:,0] =  y0
    
    for n in range(1,num_pts):
        h = tmesh[n] - tmesh[n-1]
        t = tmesh[n-1]
        k1 = soln[:,n-1]
        k2 = soln[:,n-1] + (h/2)*f(t,k1)
        k3 = soln[:,n-1] + (h/2)*f(t + h/2, k2)
        k4 = soln[:,n-1] + h*f(t + h/2, k3)
        soln[:,n] = soln[:,n-1] + (h/6)*(f(t,k1) + 2*f(t+h/2, k2)+2*f(t+h/2, k3)+f(t+h,k4))
    return soln

def ab_two_step(f, tmesh, y0, y1):
    """
    Two-Step Adams-Bashforth ODE Solver:
        - f: Right hand side of ODE f(t,y)
        - y0: Initial condition t0
        - y1: 2nd Initial Condition t1
        - tmesh: array of times
    """
    num_pts = len(tmesh)
    try:
        dim = len(y0)
    except TypeError:
        dim = 1
    soln = np.zeros((dim,num_pts))
    soln[:,0] = y0
    soln[:,1] = y1
    
    for n in range(2,num_pts):
        h = tmesh[n] - tmesh[n-1]
        t = tmesh[n-1]
        soln[:,n] = soln[:,n-1] + (h/2)*(3*f(t,soln[:,n-1]) - f(tmesh[n-2],soln[:,n-2]))
    return soln

def ab_three_step(f, tmesh, y0, y1, y2):
    """
    Three-Step Adams-Bashforth ODE Solver:
        - f: Right hand side of ODE f(t,y)
        - y0: Initial condition t0
        - y1: 2nd Initial Condition t1
        - y2: 3rd Initial Condition t2
        - tmesh: array of times
    """
    num_pts = len(tmesh)
    dim = len(y0)
    soln = np.zeros((dim,num_pts))
    soln[:,0] =  y0
    soln[:,1] = y1
    soln[:,2] = y2
    
    for n in range(3,num_pts):
        h = tmesh[n] - tmesh[n-1]
        soln[:,n] = soln[:,n-1] + (h/12)*(23*f(tmesh[n-1], soln[:,n-1]) - 16*f(tmesh[n-2], soln[:,n-2]) + 5*f(tmesh[n-3], soln[:,n-3]))
    return soln

def am_two_step(f, tmesh, y0, y1):
    """
    Two-Step Adams-Moulton ODE Solver:
        - f: Right hand side of ODE f(t,y)
        - y0: Initial condition t0
        - y1: 2nd Initial Condition t1
        - tmesh: array of times
    """
    num_pts = len(tmesh)
    try:
        dim = len(y0)
    except TypeError:
        dim = 1
    soln = np.zeros((dim,num_pts))
    soln[:,0] =  y0
    soln[:,1] = y1
    y_old1 = np.copy(y0)
    y_old = np.copy(y1)
    
    for n in range(2,num_pts):
        h = tmesh[n] - tmesh[n-1]
        nonlin_func = lambda y: -y + y_old + (1/12)*h*(5*f(tmesh[n], y) + 8*f(tmesh[n-1], y_old)-f(tmesh[n-2], y_old1))
        soln_root = optimize.root(nonlin_func, y_old)
        soln[:,n] = soln_root.x
        y_old = soln[:,n]
        y_old1 = soln[:,n-1]
    return soln

def bdf3(f, tmesh, y0, y1, y2):
    """
    BDF3 Method ODE Solver:
        - f: Right hand side of ODE f(t,y)
        - y0: Initial condition t0
        - y1: 2nd Initial Condition t1
        - y2: 3rd Initial Condition t2
        - tmesh: array of times
    """
    b = 6/11
    a0 = 18/11
    a1 = -9/11
    a2 = 2/11
    
    num_pts = len(tmesh)
    try:
        dim = len(y0)
    except TypeError:
        dim = 1
    soln = np.zeros((dim,num_pts))
    soln[:,0] = y0
    soln[:,1] = y1
    soln[:,2] = y2
    y_old2 = np.copy(y0)
    y_old1 = np.copy(y1)
    y_old = np.copy(y2)
    
    for n in range(3,num_pts):
        h = tmesh[n] - tmesh[n-1]
        nonlin_func = lambda y: -y + a0*y_old + a1*y_old1 + a2*y_old2 + h*b*f(tmesh[n], y)
        soln_root = optimize.root(nonlin_func, y_old)
        soln[:,n] = soln_root.x
        y_old = soln[:,n]
        y_old1 = soln[:,n-1]
        y_old2 = soln[:, n-2]
    return soln
    
    
def two_stage_gauss(f, tmesh, y0):
    """
    Two-Stage Gauss Method ODE Solver:
        - f: Right hand side of ODE f(t,y)
        - y0: Initial condition t0
        - tmesh: array of times
    """
    num_pts = len(tmesh)
    try:
        dim = len(y0)
    except TypeError:
        dim = 1
    soln = np.zeros((dim,num_pts))
    soln[:,0] =  y0
    
    # Set constants from Butcher table
    tau1 = (1/2) - (1/6)*np.sqrt(3) 
    tau2 = (1/2) + (1/6)*np.sqrt(3)
    a11 = 1/4
    a12 = (3-2*np.sqrt(3))/12
    a21 = (3+2*np.sqrt(3))/12
    a22 = 1/4
    
    def nonlin_sys(p):
        yn1, yn2 = p[0], p[1]
        F = (-yn1 + y_old + h*(a11*f(t+tau1*h, yn1) + a12*f(t+tau2*h, yn2)), 
                      -yn2 + y_old + h*(a21*f(t+tau1*h, yn1) + a22*f(t+tau2*h, yn2)))
        F = np.reshape(F, (2,))
        return F
    
    y_old = np.copy(y0)    
    for n in range(1,num_pts):
        h = tmesh[n] - tmesh[n-1]
        t = tmesh[n-1]
        yns = optimize.root(nonlin_sys, [y_old, y_old])
        yn1, yn2 = yns.x[0], yns.x[1]
        soln[:,n] = y_old + (h/2)*(f(t+tau1*h, yn1)+f(t+tau2*h, yn2))
        y_old = soln[:,n]
        
    return soln
