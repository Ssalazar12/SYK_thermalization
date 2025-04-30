
import kbanalysis as kb # custom module
import numpy as np
import math
import pandas as pd
from  scipy.optimize import curve_fit

# ------------------------------------------------------------------------------------------
# GLOBAL VARS AND CONSTANTS
# ------------------------------------------------------------------------------------------

# data_route = 'datasets/predictor_corrector/medium_T/'
# data_route = 'datasets/predictor_corrector/high_T/'
data_route = 'datasets/predictor_corrector/zero_T/'

# Low T
names_list = [
            "pred_sigma0.300000_Jtwo0.001000_J1.000000_dt0.050000_origin2000_mint-100.000000_maxt1000.000000.h5",
            "pred_sigma0.300000_Jtwo0.002000_J1.000000_dt0.050000_origin2000_mint-100.000000_maxt1000.000000.h5",
            "pred_sigma0.300000_Jtwo0.003000_J1.000000_dt0.050000_origin2000_mint-100.000000_maxt1000.000000.h5",
            "pred_sigma0.300000_Jtwo0.004000_J1.000000_dt0.050000_origin2000_mint-100.000000_maxt1000.000000.h5",
            "pred_sigma0.300000_Jtwo0.005000_J1.000000_dt0.050000_origin2000_mint-100.000000_maxt1000.000000.h5",
            "pred_sigma0.300000_Jtwo0.006000_J1.000000_dt0.050000_origin2000_mint-100.000000_maxt1000.000000.h5",
            "pred_sigma0.300000_Jtwo0.007000_J1.000000_dt0.050000_origin2000_mint-100.000000_maxt1000.000000.h5",
            "pred_sigma0.300000_Jtwo0.008000_J1.000000_dt0.050000_origin2000_mint-100.000000_maxt1000.000000.h5",
            "pred_sigma0.300000_Jtwo0.009000_J1.000000_dt0.050000_origin2000_mint-100.000000_maxt1000.000000.h5",
            "pred_sigma0.300000_Jtwo0.010000_J1.000000_dt0.050000_origin2000_mint-100.000000_maxt1000.000000.h5",
            "pred_sigma0.300000_Jtwo0.020000_J1.000000_dt0.050000_origin2000_mint-100.000000_maxt1000.000000.h5",
            "pred_sigma0.300000_Jtwo0.030000_J1.000000_dt0.050000_origin2000_mint-100.000000_maxt1000.000000.h5",
            "pred_sigma0.300000_Jtwo0.040000_J1.000000_dt0.050000_origin2000_mint-100.000000_maxt1000.000000.h5",
            "pred_sigma0.300000_Jtwo0.050000_J1.000000_dt0.050000_origin2000_mint-100.000000_maxt1000.000000.h5",
            ]

names_list = [
            "pred_sigma0.300000_Jtwo0.060000_J1.000000_dt0.050000_origin2000_mint-100.000000_maxt1000.000000.h5",
            "pred_sigma0.300000_Jtwo0.070000_J1.000000_dt0.050000_origin2000_mint-100.000000_maxt1000.000000.h5",
            "pred_sigma0.300000_Jtwo0.080000_J1.000000_dt0.050000_origin2000_mint-100.000000_maxt1000.000000.h5",
            "pred_sigma0.300000_Jtwo0.090000_J1.000000_dt0.050000_origin2000_mint-100.000000_maxt1000.000000.h5"
            ]

names_list = ["pred_sigma0.180000_Jtwo0.020000_J1.000000_dt0.060000_origin1500_mint-90.000000_maxt1470.000000.h5"]


# for the extrapolation error at medium T
"""names_list = [
            'rk_sigma0.740000_Jtwo0.300000_J1.000000_dt0.007000_origin13000_mint-91.000000_maxt49.000000.npy',
            'rk_sigma0.740000_Jtwo0.300000_J1.000000_dt0.010000_origin9000_mint-90.000000_maxt90.000000.npy',
            'rk_sigma0.740000_Jtwo0.300000_J1.000000_dt0.500000_origin200_mint-100.000000_maxt90.000000.npy',
            'rk_sigma0.740000_Jtwo0.300000_J1.000000_dt0.300000_origin300_mint-90.000000_maxt90.000000.npy',
            'rk_sigma0.740000_Jtwo0.300000_J1.000000_dt0.100000_origin900_mint-90.000000_maxt90.000000.npy',
            'rk_sigma0.740000_Jtwo0.300000_J1.000000_dt0.080000_origin1130_mint-90.400000_maxt90.000000.npy',
            'rk_sigma0.740000_Jtwo0.300000_J1.000000_dt0.050000_origin1800_mint-90.000000_maxt90.000000.npy',
            'rk_sigma0.740000_Jtwo0.300000_J1.000000_dt0.030000_origin3000_mint-90.000000_maxt90.000000.npy',
            'rk_sigma0.740000_Jtwo0.300000_J1.000000_dt0.020000_origin4500_mint-90.000000_maxt90.000000.npy'
            ]"""
            
# low T    
"""names_list = [
            "pred_sigma0.030000_Jtwo0.010000_J1.000000_dt0.050000_origin2000_mint-100.000000_maxt700.000000.npy",
            "pred_sigma0.030000_Jtwo0.020000_J1.000000_dt0.050000_origin2000_mint-100.000000_maxt700.000000.npy",
            "pred_sigma0.030000_Jtwo0.030000_J1.000000_dt0.050000_origin2000_mint-100.000000_maxt700.000000.npy",
            "pred_sigma0.030000_Jtwo0.040000_J1.000000_dt0.050000_origin2000_mint-100.000000_maxt700.000000.npy",
            "pred_sigma0.030000_Jtwo0.050000_J1.000000_dt0.050000_origin2000_mint-100.000000_maxt700.000000.npy",
            "pred_sigma0.030000_Jtwo0.060000_J1.000000_dt0.050000_origin2000_mint-100.000000_maxt700.000000.npy",
            "pred_sigma0.030000_Jtwo0.070000_J1.000000_dt0.050000_origin2000_mint-100.000000_maxt700.000000.npy"
             ]"""

"""names_list = [
            "pred_sigma0.030000_Jtwo0.090000_J1.000000_dt0.050000_origin2000_mint-100.000000_maxt700.000000.npy",
            "pred_sigma0.030000_Jtwo0.100000_J1.000000_dt0.050000_origin2000_mint-100.000000_maxt700.000000.npy",
            "pred_sigma0.030000_Jtwo0.200000_J1.000000_dt0.050000_origin2000_mint-100.000000_maxt400.000000.npy",
            "pred_sigma0.030000_Jtwo0.300000_J1.000000_dt0.050000_origin2000_mint-100.000000_maxt400.000000.npy",
            "pred_sigma0.030000_Jtwo0.500000_J1.000000_dt0.050000_origin2000_mint-100.000000_maxt400.000000.npy",
            "pred_sigma0.030000_Jtwo0.700000_J1.000000_dt0.050000_origin2000_mint-100.000000_maxt100.000000.npy",
            "pred_sigma0.030000_Jtwo1.000000_J1.000000_dt0.050000_origin2000_mint-100.000000_maxt100.000000.npy"
             ]"""

# hight T 
"""names_list = [
            "pred_sigma0.990000_Jtwo0.010000_J1.000000_dt0.050000_origin2000_mint-100.000000_maxt700.000000.npy",
            "pred_sigma0.990000_Jtwo0.070000_J1.000000_dt0.050000_origin2000_mint-100.000000_maxt700.000000.npy"
             ]"""


def twoExpFit(x, m, t):
    # function to fit exponentially decaying data
    return m * np.exp(-t * x) + kin_th

Nbeta  = 1000 # number of betas for the fit
diag_list = [400,425,450,475,500,525,550,575,600] # number of the diagonals to check thermalization
# diag_list = [10,20,30] for extrapolation error


data_dict = {'dt':[],'maxt':[],'sigma':[],'Jq':[],'J2':[],'beta0':[],'betaf':[], 'betaMSE':[] ,'E0':[], 'Ef':[], 'Neq_Kin': [], 
             'Neq_Int' : [], 'ReGamma':[],'ReErr':[], 'ImGamma':[], 'ImErr':[],'ReMSE':[],'ImMSE':[], 'Therm_MSE':[], 'mean_dE':[], 'max_dE': [],
                'thermalE':[], 'thermal_Kin':[], 'thermal_int':[], 'EGamma':[], 'Econst':[],'Eampli':[],'EGamm_err':[],'EMSE':[],
                'E2Gamma':[], 'E2ampli': [], 'E2MSE':[]}

# ---------------------------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------------------------

for str_name in names_list:
    print(' ')
    print('-------------------------------------')
    print('initializing for: ')
    print(str_name)
    gn , sig, J2, Jq, dt, origin_i, mint, maxt =  kb.initialize_data(data_route,str_name,use_h5=True)

    # create csv to append on the fly
    # data_str = 'data_analysis/data_sig{}_2p_imag_high.csv'.format(sig)
    # data_str = 'datasets/rk4/extrapolation_sig{:4f}_Jtwo{:.4f}_redu.csv'.format(sig, J2[-1])
    data_str = 'data_analysis/zero_T.csv'.format(sig)
    data_df = pd.DataFrame.from_dict(data_dict)
    data_df.to_csv(data_str)

    # calculating the energy --------------------------------------------------------------------------------------------------
    print("Calculating NEQ energy ...")
    Et1, Ekin, Eq = kb.calculate_energy(origin_i, gn, Jq, J2,dt,sig)
    pre_E = - Jq[0] * np.sqrt(1- (sig/Jq[0])**2) # prequench energy 
    # calculate the change in the energy
    DE = np.diff(Et1.real, n=1)
    DE_rel = np.abs(np.divide(DE, Et1[1:].real)) # percent change
    DE_mean = np.mean(DE_rel[-100:]) # mean percent change of the final data points
    DE_max = max(DE_rel[origin_i:]) # maximum percent change of the energy
    time_range = np.linspace(mint,maxt,len(gn))

    # Calculating final T -----------------------------------------------------------------------------------------------------
    # the estimated part of interest is the one after the quench so we average the last points
    estimated_E = np.mean(Et1[-500:-1])
    # since these are not constant we just take the last value
    Ekin_mean = Ekin[-1].real
    Eq_mean = Eq[-1].real
    rel_err = np.abs((pre_E.real-estimated_E.real)/pre_E.real)

    # find the initial temperature
    theta = np.arccos(sig/Jq[0])
    betaJ_0 = 2*theta*Jq[0]/sig

    # choosing the timeslice a the latest time
    g_slice = np.flip(gn[:,-1])

    # fit the final temperature imag version
    best_beta, best_g, thE_q ,MSE = kb.fit_temperature_imag(g_slice, maxt, Eq_mean ,dt,Jq, J2,Nbeta)
    thE_list = [estimated_E, estimated_E - thE_q ,thE_q]

    print('MSE:' ,MSE, "best beta: ", best_beta)

    # Fitting the diagonals to estimate the thermalization rate ---------------------------------------------------------------
    g_origin = gn[origin_i:,origin_i:]

    # save the rates from each diagonal
    im_list = []
    im_err_list = []
    re_list = []
    re_err_list = []
    reMSE_list = []
    imMSE_list = []

    print("Fitting the diagonals ...")

    for i in range(0,len(diag_list)):
        dindices = kb.kth_diag_indices(g_origin,diag_list[i])
        # we subtract the last value of the diagonal to make sure it is zero at infinity and get a nice fit plus a little
        gshift_RE = g_origin[dindices] - (g_origin[dindices][-1])
        gshift_RE = gshift_RE.real
        gshift_IM = np.abs(g_origin[dindices].imag)
        # this is the "average" time but since we are in the diagonal it has the same scale
        tau_g = np.linspace(0,len(gshift_RE)*dt,len(gshift_RE))
        # do an exponential fit for th real and imaginary parts
        # index 0 is the amplitude , index 1 is the constant and index 2 is the decay rate

        # FOR EXTRAPOLATION ERROR DATAFRAME DONT FIT DIAGONALS !
        try:
            tau, RA, RAerr, Rconst, Rcerr ,Rgamma, Rgammaerr, RrelMSE, Rmse =  kb.fit_diagonal(gshift_RE,dt ,cutoff = 10**(-10))
        except:
            print("curve fit of imaginary diagonal did not converge")
            tau = RA = RAerr = Rconst = Rcerr = Rgamma = Rgammaerr = RrelMSE = Rmse = 0

        try:
            _, IA, IAerr, Iconst, Icerr ,Igamma, Igammaerr, IrelMSE, Imse =  kb.fit_diagonal(gshift_IM,dt ,cutoff = 10**(-10))
        except:
            print("curve fit of real diagonal did not converge")
            IA = IAerr = Iconst = Icerr = Igamma = Igammaerr = IrelMSE = Imse = 0

        Rgfitted = kb.ExpFit(tau, RA,Rconst ,Rgamma)
        Igfitted = kb.ExpFit(tau, IA, Iconst,Igamma)

        re_list.append(Rgamma)
        re_err_list.append(Rgammaerr)
        im_list.append(Igamma)
        im_err_list.append(Igammaerr)
        reMSE_list.append(Rmse)
        imMSE_list.append(Imse)
        
    Rgamma = np.mean(re_list)
    Rgammaerr = np.mean(re_err_list)
    Igamma = np.mean(im_list)
    Igammaerr = np.mean(im_err_list)
    IMSE = np.mean(imMSE_list)
    RMSE = np.mean(reMSE_list)

    # FIT THERMALIZATION RATE FROM THE KINETIC ENERGY --------------------------------------------------------

    print("Fitting the Energy Components...")

    # FIRST THE 3 PARAMETER FIT
    # Fit the interaction term of the energy to get another estimate of the thermalization rate
    Ek_shift = Ekin[origin_i:].real - Et1[-1].real
    # define a minimum value where we cut-off our estimation for numerical reasons
    # cutoff = 10**(-10) # deprecated
    # itemindex = np.where(Eq_shift <= cutoff)[0][0] # get first index below cutoff
    # itemindex = -1
    cut_Ek = Ek_shift
    cut_t = np.linspace(0,len(cut_Ek)*dt,len(cut_Ek))
    # do a first guess with a simple polinomial fit
    log_tofit = np.log(cut_Ek) 
    a,b = np.polyfit(cut_t, log_tofit, deg=1) # a is the decay rate guess, b the "amplitude"
    p0 = (1.0,-1.0, abs(a))

    # try an exponential non-linear fit
    try:
        params, cov = curve_fit(kb.ExpFit, cut_t, Ekin[origin_i:].real , p0)
         # get the final non linear fit, cov is the covariance matrix
        perr = np.sqrt(np.diag(cov))
        gamma = params[2] # the decay rate
        ampli3p = params[0]
        const3p = params[1]
        err3p = perr[2]
    except:
        print(" 3 parameter nonlinear curve fit of energy did not converge")
        perr = 100
        gamma = a # the decay rate
        ampli3p = b
        const3p = Et1[-1].real
        err3p = 100

    # estimate the error
    E_fit = ampli3p*np.exp(-gamma*cut_t) + const3p
    Emse = ((E_fit - Ekin[origin_i:].real)**2).mean() # mse of the energy fit

    # NOW THE 2 PARAMETER NONLINEAR FIT
    kin_th = thE_list[1].real
    p0 = (1.0,abs(a))
    try:
        params, cov = curve_fit(twoExpFit, cut_t, Ekin[origin_i:].real , p0)
        perr = np.sqrt(np.diag(cov))
        gamma2p = params[1] 
        ampli2p = params[0]
        E_fit = ampli2p*np.exp(-gamma2p*cut_t) + kin_th
        Emse2p = ((E_fit - Ekin[origin_i:].real)**2).mean()

    except:
        print(" 2 parameter nonlinear curve fit of energy did not converge")
        perr = 100
        gamma2p = 0
        ampli2p = 0
        E_fit = 0
        Emse2p = 100

    # collet the relevant data in a dictionary
    data_dict['dt'].append(dt)
    data_dict['maxt'].append(maxt)
    data_dict['sigma'].append(sig)
    data_dict['Jq'].append(Jq[-1])
    data_dict['J2'].append(J2[-1])
    data_dict['beta0'].append(betaJ_0)
    data_dict['betaf'].append(best_beta)
    data_dict['E0'].append(pre_E.real)
    data_dict['Ef'].append(estimated_E.real)
    data_dict['ReGamma'].append(Rgamma)
    data_dict['ReErr'].append(Rgammaerr)
    data_dict['ImGamma'].append(Igamma)
    data_dict['ImErr'].append(Igammaerr)
    data_dict['ReMSE'].append(IMSE)
    data_dict['ImMSE'].append(RMSE) 
    data_dict['Therm_MSE'].append(MSE) 
    data_dict['mean_dE'].append(DE_mean) 
    data_dict['max_dE'].append(DE_max)
    data_dict['thermalE'].append(thE_list[0].real)
    data_dict['thermal_Kin'].append(thE_list[1].real) 
    data_dict['thermal_int'].append(thE_list[2].real)
    data_dict['Neq_Kin'].append(Ekin_mean)
    data_dict['Neq_Int'].append(Eq_mean)
    data_dict['EGamma'].append(gamma)
    data_dict['Econst'].append(const3p)
    data_dict['Eampli'].append(ampli3p)
    data_dict['EGamm_err'].append(err3p)
    data_dict['EMSE'].append(Emse)
    data_dict['E2Gamma'].append(gamma2p)
    data_dict['E2ampli'].append(ampli2p)
    data_dict['E2MSE'].append(Emse2p)
    data_dict['betaMSE'].append(MSE)
    
    # append data to existing csv 
    row_df = pd.DataFrame.from_dict(data_dict)

    with open(data_str , 'a') as f:
        row_df.to_csv(f, header=False)

    # put the output in a txt file for keeping track
    with open("analysis_output.txt", "a") as text_file:
        text_file.write("Finished")
        text_file.write("\n")
        text_file.write(str_name)
        text_file.write("\n")








	
