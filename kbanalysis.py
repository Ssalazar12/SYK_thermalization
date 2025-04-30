import numpy as np
import math
import itertools
import re
import cmath
from scipy.integrate import odeint, simpson
from  scipy.optimize import curve_fit
import gc
import h5py

from tqdm import tqdm

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ---------------------------------------------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------------------------------------------

N = 100 # The coefficients c_0, ..., c_N will be used
Nx = 500# number of points in the normalized imaginary time axis X
# We stop the search when err_N <= 10^(err_max) 


# ---------------------------------------------------------------------------------------------------------------
# READ DATA 
# ---------------------------------------------------------------------------------------------------------------

def initialize_data(data_route,str_name, use_h5=False):

    # loads the data and initializes the parameters
    # if data is saved as a numpy array
    if use_h5 == False:
        gdata = np.load(data_route+str_name)
    # if data is saved as hsf5 file
    else:
        h5read = h5py.File(data_route+str_name,'r')
        # transpose because the julia indexing used to generate this is weird
        gdata = h5read['values'][:,:].T
        h5read.close()

    # check for missing values
    if np.isnan(gdata).any():
        print("found nans in array this may yield problems ...")
    
    params = [float(s) for s in re.findall(r"[-+]?(?:\d*\.*\d+)", str_name)]
    
    j2 = params[1]
    j = params[2]
    Dt = params[3]
    origin_index =  int(params[4]) # chamge from julia indexing
    Mint = -origin_index*Dt # since sometimes we don't save the whole matrix to save space
    Maxt = float(params[6])
    # get several temperatures since they are different
    Sigma = params[0]
    
    jq = np.ones(len(gdata))
    jq[origin_index-1:] = j
    j2_list  = np.zeros(len(gdata))
    j2_list[origin_index-1:] = j2
    
    return gdata, Sigma, j2_list, jq , Dt, origin_index, Mint, Maxt

# ---------------------------------------------------------------------------------------------------------------
# ENERGY CALCULATIONS
# ---------------------------------------------------------------------------------------------------------------

    
def eq_g(time1,time2, Sigma):
    # equilibrium solutions
    Theta = np.arccos(Sigma/J)
    return 2*np.log(Sigma/(J*np.cosh(Theta*I + Sigma*(time1-time2))))

def calculate_energy(nint, gn, jq, j2_list,Dt,sig):
    # nint: integer which tells the time index where the integrations begin
    # gn: array representing the two-time green's function 
    # jq, j2: floats, couplings 

    Et1 = np.zeros(len(gn), complex)
    # track the kinetic term contriubtion
    Ekin = np.zeros(len(gn), complex)
    # track the interaction term contribution
    Eq = np.zeros(len(gn), complex)
    
    for t1 in range(nint, len(Et1)):
        # integration time we need to shift it to get the proper indeximg
        tn = t1 + 1 # time has to lead here due to python convention with array slicing
        gg = gn[0:tn,t1]
        qintegrand = jq[t1]*jq[0:tn]*(np.exp(gg) - np.exp(np.conj(gg))) 
        kintegrand = j2_list[t1]*j2_list[0:tn]*(gg - np.conj(gg))
        
        # calculate the energy at each timestep t1. We integrate along the columns up to the time t
        integrand = qintegrand + kintegrand

        Et1[t1] = -1j*0.5*Dt*(0.5*integrand[0] + np.sum(integrand[1:-1]) + 0.5*integrand[-1] )
        Ekin[t1] = -1j*0.5*Dt*(0.5*kintegrand[0] + np.sum(kintegrand[1:-1]) + 0.5*kintegrand[-1] )
        Eq[t1] = -1j*0.5*Dt*(0.5*qintegrand[0] + np.sum(qintegrand[1:-1]) + 0.5*qintegrand[-1] )

    # calculates the prequench energy
    pre_e = - jq[0] * np.sqrt(1-(sig/jq[0])**2)
    Et1[0:nint+1] = complex(pre_e)
    Eq[0:nint+1] = complex(pre_e)
        
    return Et1.real, Ekin, Eq
    
def thermal_energy(gg, j2, jq,dt):
    # calculates the energy of the real time equilibrium solution
    # gg: 1d array containing the real  time thermal green's function
    # j2, jq: 1d arrays containing the timedependent coupling
    
    # flip it so we get the order of starting from 0 to +infinity
    g_thermal = np.flip(gg)

    k_integrand = j2[-1]*j2[-1]*g_thermal
    q_integrand = jq[-1]*jq[-1]*np.exp(g_thermal) 
    t = np.linspace(0, len(g_thermal)*dt,len(g_thermal))

    #ek = np.imag( simpson(k_integrand,t) )
    #eq = np.imag( simpson(q_integrand,t) )

    ek = dt*np.imag( 0.5*k_integrand[0] + np.sum(k_integrand[1:-1]) + 0.5*k_integrand[-1] )
    eq = dt*np.imag( 0.5*q_integrand[0] + np.sum(q_integrand[1:-1]) + 0.5*q_integrand[-1] )
    
    return ek, eq, ek+eq

def img_time_qenergy(gg,B,BETA,dx):

    q_integrand = np.exp(gg) 
    eq =  -(B/(8*BETA))*simpson(q_integrand, dx=dx)
    return eq


# ---------------------------------------------------------------------------------------------------------------------
# FOR THE TEMPERATURE ESTIMATION
# ---------------------------------------------------------------------------------------------------------------------


def fit_temperature(g_slice, maxt, NeqE ,dt,Jq,J2,Nbeta=300):
    # fits a time slice of the Kadanoff-Baym solution to a equilibrium solutions for different temperatures
    # and finds the best beta which estimates the final temperature of the system. 
    # gslice: Array. A COLUMN of the matrix that represents the ooe green's function in the two-timeplane
    # NeqE: FLoat representing the nonequilibrium energy
    # Jq, J2: Floats. THe couplings of the model
      
    # for low J2  
    # beta_list =  list(np.linspace(0.001, 2.0 ,Nbeta))  + list(np.linspace(8, 12 ,10)) # Nbeta = 1000
    # for high J2
    beta_list =  list(np.linspace(0.0001, 0.005 ,Nbeta)) # NBETA = 1000

    tsteps = len(g_slice)
    thermal_list = []
    therm_E = []
    # kinetic energy
    therm_kin = []
    # interaction energy
    therm_q  = []
    
    for beta in tqdm(beta_list, desc ="Fitting final beta"):
        # SET UP THE imaginary time solution for a give beta
        a = (2*beta*J2[-1])**2
        b = (2*beta*Jq[-1])**2
        _,_, cn_list, err_ = find_g(a, b)
        # calculate the derivative 
        alpha_x = get_deriv(cn_list, a,b)
        # relate it to the real time version. We choose the negative solution because that is the one
        # related to positive t=t1-t2 which is what we are looking at here
        ialpha = -1j*alpha_x/beta

        g_thermal, _,_ = solution_to_thermal(tsteps, 0, ialpha, dt, Jq[-1],J2[-1])
        # calculate energy for the current thermal solution
        th_kin, th_q, th_E = thermal_energy(g_thermal, J2, Jq,dt)
        
        therm_E.append(th_E)
        therm_kin.append(th_kin)
        therm_q.append(th_q)
        thermal_list.append(g_thermal)
        
    # find the solution closest to the "postquench" energy
    therm_E = np.asarray(therm_E)
    E_diffs = np.abs((NeqE - therm_E))
    minimum_index = np.asarray(E_diffs).argmin()

    best_beta = beta_list[minimum_index]
    best_g = thermal_list[minimum_index]
    best_thE = therm_E[minimum_index]
    best_thkin = therm_kin[minimum_index]
    best_thq = therm_q[minimum_index]
    best_diff = E_diffs[minimum_index]

    # find the mean squared error between the thermal and kb solutions
    mse = np.mean(np.abs(best_g - g_slice)**2)
    
    
    return best_beta, best_g, [best_thE, best_thkin, best_thq] ,mse

def fit_temperature_wigner(Nt,origin_index ,maxt, NeqE ,dt,Jq,J2,Nbeta=300):
    # fits a wigner transformed equilibrium solutions for different temperatures
    # and finds the beta which has the same energy of the initial state 
    # gslice: Array. A COLUMN of the matrix that represents the ooe green's function in the two-timeplane
    # NeqE: FLoat representing the nonequilibrium energy
    # Jq, J2: Floats. THe couplings of the model
      
    # for low J2  
    beta_list =  list(np.linspace(0.001, 2.0 ,Nbeta))  + list(np.linspace(8, 12 ,10))
    # for high J2
    #beta_list =  list(np.linspace(0.0001, 0.005 ,Nbeta)) 

    tsteps = int(Nt/2)
    thermal_list = []
    therm_E = []
    # kinetic energy
    therm_kin = []
    # interaction energy
    therm_q  = []
    
    for beta in tqdm(beta_list, desc ="Fitting final beta"):
        # SET UP THE imaginary time solution for a give beta
        a = (2*beta*J2[-1])**2
        b = (2*beta*Jq[-1])**2
        _,_, cn_list, err_ = find_g(a, b)
        # calculate the derivative 
        alpha_x = get_deriv(cn_list, a,b)
        # relate it to the real time version. We choose the negative solution because that is the one
        # related to positive t=t1-t2 which is what we are looking at here
        ialpha = -1j*alpha_x/beta

        g_thermal, _,_ = solution_to_thermal(tsteps, 0, ialpha, 2*dt, Jq[-1],J2[-1])
        # wigner transform the thermal solution
        g_rot = np.zeros(Nt, complex)
        g_rot = np.repeat(g_thermal, 2)
        # calculate the thermal E(t1) at the maximum t1 
        kintegrand = (J2[-1]**2)*(g_rot - np.conj(g_rot))
        qintegrand = (Jq[-1]**2)*(np.exp(g_rot) - np.exp(np.conj(g_rot))) 
       
        th_kin = -1j*0.5*dt*(0.5*kintegrand[0] + np.sum(kintegrand[1:-1]) + 0.5*kintegrand[-1] )
        th_q = -1j*0.5*dt*(0.5*qintegrand[0] + np.sum(qintegrand[1:-1]) + 0.5*qintegrand[-1] )
        th_E = th_q + th_kin
        
        therm_E.append(th_E)
        therm_kin.append(th_kin)
        therm_q.append(th_q)
        thermal_list.append(g_thermal)

        gc.collect()
  
    # find the solution closest to the "postquench" energy
    therm_E = np.asarray(therm_E)
    E_diffs = np.abs((NeqE - therm_E))
    minimum_index = np.asarray(E_diffs).argmin()

    best_beta = beta_list[minimum_index]
    best_g = thermal_list[minimum_index]
    best_thE = therm_E[minimum_index]
    best_thkin = therm_kin[minimum_index]
    best_thq = therm_q[minimum_index]
    best_diff = E_diffs[minimum_index]

    return best_beta, best_g, [best_thE, best_thkin, best_thq]


def get_gx(coef_list, nsamples):
    # Calculates the imaginary time g(x) using the estimated coefficients from find_g
    # coef_list: array containing the coeffecients from the power series expansion
    # nsamples: int the number of points in the x axis
    X = np.linspace(0,1,nsamples) # always between 0 and 1 due to periodicity
    GX = np.zeros(len(X))
    
    for i in range(0,len(X)):
        res = 0
        for n in range(0,len(coef_list)):
            res = res + coef_list[n]*(X[i]-0.5)**(2*n)
        GX[i] = res
        
    return X, GX

def get_deriv(coef_list,a,b):
    # gets the initial condition of the first derivative of g(x) from the coefficients
    # there are two solutions here 
    return cmath.sqrt( a+b-4*coef_list[1]-a*coef_list[0] )

def fit_temperature_imag(g_slice, maxt, NeqEq ,dt,Jq,J2,Nbeta=300):
    # fits a time slice of the Kadanoff-Baym solution to a equilibrium solutions for different temperatures
    # and finds the best beta which estimates the final temperature of the system. Using the imaginary time energy
    # gslice: Array. A COLUMN of the matrix that represents the ooe green's function in the two-timeplane
    # NeqE: FLoat representing the nonequilibrium energy
    # Jq, J2: Floats. THe couplings of the model
        
    # for low J2
    beta_list =  list(np.linspace(0.0007, 2 ,Nbeta)) + list(np.linspace(2, 12 ,250))
    # at med to high J2 
    #beta_list =  list(np.linspace(0.000001, 0.001 ,Nbeta))

    # beta_list = beta_med_list + beta_low_list

    tsteps = len(g_slice)
    thermal_list = []
    # interaction energy
    therm_q  = []
    
    for beta in tqdm(beta_list, desc ="Fitting final beta"):
        # SET UP THE imaginary time solution for a give beta
        a = (2*beta*J2[-1])**2
        b = (2*beta*Jq[-1])**2
        _,_, cn_list, err_ =find_g(a, b)
        x, gx = get_gx(cn_list, 200)
        dx = x[1]-x[0]
        # calculate energy for the current thermal solution
        eq_im = img_time_qenergy(gx,b,beta,dx)

        # calculate the derivative 
        alpha_x = get_deriv(cn_list, a,b)
        # relate it to the real time version. We choose the negative solution because that is the one
        # related to positive t=t1-t2 which is what we are looking at here
        ialpha = -1j*alpha_x/beta

        g_thermal, _,_ = solution_to_thermal(tsteps, 0, ialpha, dt, Jq[-1],J2[-1])
        
        therm_q.append(eq_im)
        thermal_list.append(g_thermal)
        
    # find the solution closest to the "postquench" energy
    therm_E = np.asarray(therm_q)
    E_diffs = np.abs((NeqEq - therm_E))
    minimum_index = np.asarray(E_diffs).argmin()

    best_beta = beta_list[minimum_index]
    best_g = thermal_list[minimum_index]
    best_thq = therm_q[minimum_index]
    best_diff = E_diffs[minimum_index]

    # find the mean squared error between the thermal and kb solutions
    mse = np.mean(np.abs(best_g - g_slice)**2)
    
    return best_beta, best_g, best_thq ,mse


def wingerize_thermal(gth, Nsize, do_all=True):
    # Put the thermal solution gth into two time plane by using Wigner coordinates
    # gth: 1d array containng the thermal solution
    # Nsize: integer, size of the 2-time matrix to be created 2 times bigger than the max len of gth
    # do_all: boolean, if we fill all of the matrix or only half
    A = np.zeros((Nsize,Nsize),dtype = complex)

    anti_arr = np.fliplr(A)

    val_ = np.conjugate(gth)
    val_conj = np.flip(np.conjugate(val_))
    if do_all==True:
        end_index = len(A)
    else:
        end_index = 1

    # fill the anti diagonals of the matrix with the values of g_thermal
    for i in range(-len(A)+1, end_index):
        dindices = kth_diag_indices(A,i)
        # length of the diagonal
        l_diag  = np.shape(dindices)[1]
        # half length of the diagonal
        l_diag_half = int(l_diag/2)

        # there are three cases to consider
        if l_diag==1:
            anti_vals = 0
        # when even
        elif l_diag%2==0:
            anti_vals = list(val_conj[-l_diag_half:]) + list(val_[:l_diag_half])
        # when odd
        else:
            anti_vals = list(val_conj[-l_diag_half:]) + [0] + list(val_[:l_diag_half]) 

        anti_arr[dindices] = anti_vals

    return np.fliplr(anti_arr)


# ---------------------------------------------------------------------------------------------------------------
# FOR RUNGE KUTTA METHOD 
# --------------------------------------------------------------------------------------------------------------- 

def ode(t,u1,u2,Jq,J2):
    ode_1 = u2
    ode_2 = -2*(Jq**2)*np.exp(u1) - 2*(J2**2)
    return np.array([ode_1,ode_2])

def solution_to_thermal(time_steps, g0, dg0, Dt, Jq,J2):
    # solves the thermalized equations using Runge Kutta 4. Uses scipy ODE
    # times_steps (int): number of steps the simulations will have
    # g0, dg0 (complex float): initial condition for the function and derivative respectively
    # Dt (float): step size for each timestep

    # initialize the initial conditions
    t0 = 0.0
    y0 = g0
    dy0 = dg0
    N = time_steps
    h = Dt
    tf = h*N # final time

    # For RK4 we transform to a system com 1st order coupled ODE's
    t = np.empty(N)
    y = np.empty(N,dtype=complex) # the actual function we want
    u = np.empty(N,dtype=complex) # the derivative
    
    # here the idea is g'' = -2*(J**2)*np.exp(g) - 2*J2**2
    # so we transform to: u=y' and y=g s.t. u'=  -2*(J**2)*np.exp(g) - 2*J2**2 and y'=u

    t[0] = t0
    y[0] = y0; u[0] = dy0 
    # RK4
    for i in range(0,N-1,1):
        # Estimate derivative at the start of the interval
        k11 = h*ode(t[i],y[i],u[i],Jq,J2)[0]
        k12 = h*ode(t[i],y[i],u[i],Jq,J2)[1]

        # Estimate derivatie at the middle of the interval
        k21 = h*ode(t[i]+(h/2),y[i]+(k11/2),u[i]+(k12/2),Jq,J2)[0]
        k22 = h*ode(t[i]+(h/2),y[i]+(k11/2),u[i]+(k12/2),Jq,J2)[1]
        
        # Again Estimate derivatie at the middle of the interval
        k31 = h*ode(t[i]+(h/2),y[i]+(k21/2),u[i]+(k22/2),Jq,J2)[0]
        k32 = h*ode(t[i]+(h/2),y[i]+(k21/2),u[i]+(k22/2),Jq,J2)[1]

        # Estimate derivatie at the end of the interval
        k41 = h*ode(t[i]+h,y[i]+k31,u[i]+k32, Jq,J2)[0]
        k42 = h*ode(t[i]+h,y[i]+k31,u[i]+k32, Jq,J2)[1]
        
        # calculate the solutions at the next time step
        y[i+1] = y[i] + ((k11+2*k21+2*k31+k41)/6)
        u[i+1] = u[i] + ((k12+2*k22+2*k32+k42)/6)
        t[i+1] = t[i] + h
        
    return y, u, t

# ---------------------------------------------------------------------------------------------------------------
# FITTING DIAGONALS
# ---------------------------------------------------------------------------------------------------------------


def kth_diag_indices(a, k):
    # gets indices from the kth diagonal of matrix a
    # k=0 being the true diagonal of the matrix a
    rows, cols = np.diag_indices_from(a)
    if k < 0:
        return rows[-k:], cols[:k]
    elif k > 0:
        return rows[:-k], cols[k:]
    else:
        return rows, cols

def ExpFit(x, m, a,t):
    # function to fit exponentially decaying data
    return m * np.exp(-t * x) + a

def fit_diagonal(current_g, dt, cutoff = 10**(-10)):
    # function to fit the diaonals to an exponential function 
    # current_g: 1D Array, containing the diagonal we want to fit
    # cutoff: cutoff for the minimum value to avoid numericall errors from small floats


    # limit our function up to the cutoff to avoid numerical errors from very small floats
    #itemindex = np.where(current_g <= cutoff)[0][0] # get first index below cutoff

    index_container = np.where(current_g <= cutoff)[0]
    if(len(index_container) ==0):
        itemindex = -1
    else:
        itemindex =  index_container[0]

    cutg = current_g[:itemindex]
    # "average" time for the diagonals
    tau = np.linspace(0,len(cutg)*dt,len(cutg))

    # first, use a simple polyfit to get a first guess on the parameters
    log_tofit = np.log(cutg) 
    a,b = np.polyfit(tau, log_tofit, deg=1) # a is the decay rate guess, b the "amplitude"
    p0 = (1.0, 1.0, abs(a)) # the abs is due to the change in definition

    # get the fit, cov is the covariance matrix
    params, cov = curve_fit(ExpFit, tau, cutg, p0)
    # index 0 is the amplitude , index 1 is the constant and index 2 is the decay rate
    gamma = params[2] # the decay rate
    g_fit = ExpFit(tau,params[0], params[1],gamma)

    perr = np.sqrt(np.diag(cov)) # second index is standard dev of the decay factor

    # get mean square error
    SE = np.abs(g_fit - cutg)**2
    MSE = np.mean(SE)
    # get relative MSE
    rel_mse = np.mean(np.divide(np.abs(g_fit - cutg)**2, cutg))

    return tau, params[0], perr[0], params[1], perr[1] ,gamma, perr[2], rel_mse, MSE

# ---------------------------------------------------------------------------------------------------------------
# PLOTTING 
# ---------------------------------------------------------------------------------------------------------------


def plot_contour(g_matrix, Tmin, Tmax,bwidth=7,bheight=5,title_size=23):
    
    # makes a contour plot
    
    valnorm = abs(g_matrix.real.min())
    fig, ax = plt.subplots(1,2,figsize=(bwidth*2.5,bwidth))

    im = ax[0].contourf(g_matrix.real, extent=[Tmin, Tmax,Tmin,Tmax],
                      cmap='seismic', vmin=-valnorm,vmax=valnorm)
    
    # plot guiding lines for the quadrants
    ax[0].scatter(0,0, c='grey',s=50)
    ax[0].axvline(0, c='grey',linewidth=2, linestyle='dashed')
    ax[0].axhline(0, c='grey',linewidth=2, linestyle='dashed')
    ax[0].set_title(r'$\rm{Re}[g(t_1,t_2)]$', fontsize=title_size-3)
    ax[0].set_ylabel(r'$t_2$', fontsize=26)
    ax[0].set_xlabel(r'$t_1$',fontsize=26)
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax, shrink=0.9)
    
    
    valnorm = abs(g_matrix.imag.min())
    im = ax[1].contourf(g_matrix.imag, extent=[Tmin, Tmax,Tmin,Tmax], 
                 cmap='seismic', vmin=-valnorm,vmax=valnorm)
    # plot guiding lines for the quadrants
    ax[1].scatter(0,0, c='grey',s=50)
    ax[1].axvline(0, c='grey',linewidth=2, linestyle='dashed')
    ax[1].axhline(0, c='grey',linewidth=2, linestyle='dashed')
    
    ax[1].set_title(r'$\rm{Im}[g(t_1,t_2)]$', fontsize=title_size-3)
    ax[1].set_ylabel(r'$t_2$', fontsize=26)
    ax[1].set_xlabel(r'$t_1$', fontsize=26)
    
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax, shrink=0.9)

    plt.tight_layout()
    

# ------------------------------------------------------------------------------------------------------------------------
# The Following code has been taken from Nick von Selzam's bachelor Thesis with permisson 
# ------------------------------------------------------------------------------------------------------------------------

def c_n(liste, a):
    # For liste an array containing the coefficients c_0, ..., c_n-1 for the power series expansion of g calculates the next coefficient c_n. Requires n >= 2.’’’
    n = len(liste)
    res = [4*liste[k]*liste[n-k]*k*(n-k) for k in range(1, n)]
    res = sum(np.array(res)) 
    res = res - a*liste[n-1] 
    res = res/(4*n*(2*n-1))
    
    return res

def build_coeffs(c_0, a, b):
    # For c_0 a guess for g(1/2) and given a, b returns the list of coefficients c_n up to n = N’’’
    
    c_1 = (1/4) * (a + b * np.exp(c_0)) 
    c_liste = [c_0, c_1]
    for n in range(2, N+1):
        c_liste = c_liste + [c_n(c_liste, a)] 
    return c_liste

def find_g(a, b, err_max = -15 ):
    # ’’’Given a, b searches for an approximation of c_0. Return 
    # a, b, err_N and the final list of coefficients up to c_N’’’
    I = [-10**3, 0] # Search interval for c_0. If g^(N)(0) calculated
    err_N = 1
    # use this two to track when the loop get stuck 
    err_prev = 0
    counter = 0
    # from c_0 = I[0] is not negative, I[0] has to be # chosen smaller
    while abs(err_N) > 10**(err_max):
        # Guess g(1/2) to be the middle of the interval I and
        # calculate the corresponding coefficients
        c_0 = (I[0]+I[1])/2
        c_liste = build_coeffs(c_0, a, b)
        # Calculate g^(N)(0)
        g0 = c_0
        for n in range(1, N+1):
            g0 = g0 + c_liste[n]/(4**n) 
            err_N = g0/abs(c_0)
        # Shrink the interval. g(0) < 0 means c_0 was too
        # negative and we use it as the new lower bound
        # g(0) > 0 means c_0 was to large and we use it as the
        # new upper bound
        if err_N < 0: 
            I[0] = c_0 
        else: 
            I[1] = c_0
            
        if err_N  == err_prev:
            counter = counter+1
            
        if counter>=30:
            print('bump')
            break
        err_prev = err_N
            
            
    return (a, b, c_liste, err_N)



