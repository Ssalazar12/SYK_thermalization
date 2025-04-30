# CREATES DATA THAT IS TIME DEPENDENT LIKE THE ENERGY AND GREEN'S FUNCTION SLICES IN WIGNER COORDINATES

import numpy as np
import h5py
import kbanalysis as kb

data_route = 'datasets/pulled_data/'

names_list = [
	"pred_sigma0.150000_Jtwo0.010000_J1.000000_dt0.050000_origin3000_mint-150.000000_maxt1700.000000.h5",
]


# list for the anti diagonals
anti_list = [25_000,24_000,23_000, 22_000,21_000,20_000,18_000,16_000,
	             14_000,12_000,10_000, 8_000,6000, 4000 ,2000, 0]

anti_list = [4000,2000,0] 


# ------------------------------
# FUNCTIONS 
# ------------------------------

def anti_diagonal(arr, anti_index):
    # main anti-diagonal is at anti_index=0
    # negative anti_index move the antidiagonal to the right (past the main anti)
    # positive anti_index move it to the left  (before the main anti)
    anti_arr = np.fliplr(arr)
    dindices = kb.kth_diag_indices(arr,anti_index)
    
    return anti_arr[dindices]

# ------------------------------
# Main 
# ------------------------------

for name_str in names_list:

	# Load data
	gn , sig, J2, Jq, dt, origin_i, mint, maxt =  kb.initialize_data(data_route,name_str,use_h5=True)

	file_name = "test_Tdat_sig{:.3f}_Jtwo{:.3f}_dt{:.3f}_jq{:.3f}_tmax{:.1f}.h5".format(sig, J2[-1], dt, Jq[-1], maxt)

	with h5py.File(file_name, "w") as f:
		print(file_name)
		theta = np.arccos(sig/Jq[0])
		beta0 = 2*theta*Jq[0]/sig
		Et1, Ekin, Eq = kb.calculate_energy(origin_i, gn, Jq, J2,dt,sig)
		time_range = np.linspace(mint,maxt,len(gn))

		# calculate the change in the energy
		DE = np.diff(Et1.real, n=1)
		DE_rel = np.abs(np.divide(DE, Et1[1:].real)) # percent change
		DE_mean = np.mean(DE_rel[-100:]) # mean percent change of the final data points
		DE_max = max(DE_rel[origin_i:]) # maximum percent change of the energy

		time_range = np.linspace(mint,maxt,len(gn))
		n_size = len(Et1)

		# create group for energy
		grp = f.create_group("Energy")
		dset = grp.create_dataset("interaction",(n_size,), dtype='f')
		dset[:]= Eq.real
		                 
		dset = grp.create_dataset("kinetic",(n_size,), dtype='f')
		dset[:]= Ekin.real

		dset = grp.create_dataset("timerange",(n_size,), dtype='f')
		dset[:]= time_range.real

		# create group for Green's funtion in Wigner coords
		grp = f.create_group("G_tau")

		# Getting the antidiagonals
		for i in range(0,len(anti_list)):
			anti = anti_list[i]
			g_tau = anti_diagonal(gn[origin_i+1:,origin_i+1:],anti_list[i])
			tau_value = maxt*0.5 - anti*dt*0.5 

			n_size = len(g_tau)
			dset = grp.create_dataset("G{:.0f}".format(tau_value), (n_size,), dtype='complex128')
			dset[:] = g_tau
			
		# create group for Derivative along the diaonals
		grp = f.create_group("Diagonal_deriv")	

		# Getting the derivative along the diagonals
		diag_list = np.linspace(400, maxt/(2*dt) - 300 , 20, dtype=int)
		# real
		resubgrp = grp.create_group("Re_Deriv")
		imsubgrp = grp.create_group("Im_Deriv")
		timesubgrp = grp.create_group("Times_Deriv")

		for d in diag_list:
			dindices = kb.kth_diag_indices(gn[origin_i:,origin_i:],d)
			g_diag = gn[dindices] 

			# calculate the derivate wrt to dT 
			dg_dtau = np.gradient(g_diag, dt,edge_order=2) # dy/dx 2nd order accurate
			t_ = np.linspace(0,len(dg_dtau)*dt,len(dg_dtau))
			# create the subgroups and datasets
			n_size = len(t_)
			
			dset = resubgrp.create_dataset("D{:.0f}".format(d),(n_size,), dtype='f')
			dset[:]= dg_dtau.real
			# imag
			dset = imsubgrp.create_dataset("D{:.0f}".format(d),(n_size,), dtype='f')
			dset[:]= dg_dtau.imag
			# times
			dset = timesubgrp.create_dataset("T{:.0f}".format(d),(n_size,), dtype='f')
			dset[:]= t_

	f.close()






