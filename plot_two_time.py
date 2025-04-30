import kbanalysis as kb

import numpy as np
import h5py
import re

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as ticker 

# define basic figure sizes
bwidth=7
bheight=5
base_font = 28

matplotlib.rcParams.update({'font.size': base_font, "font.family": "serif", 
                            "font.serif": "CMU Serif, Times New Roman",
                           'text.usetex' : True })


# data routes
raw_route = "datasets/mixed_quench/"
raw_name = "pred_sigma0.150000_Jtwo0.010000_J1.000000_dt0.050000_origin3000_mint-150.000000_maxt1700.000000.h5"

fig_route = 'figures/pub_figs/'


def component_contour(g_matrix, Tmin, Tmax, bwidth=7, axis_font=20):    
    # makes a contour plot
    valnorm = abs(g_matrix.min())
    fig, ax = plt.subplots(1,1,figsize=(bwidth,bwidth))

    im = ax.contourf(g_matrix.real, extent=[Tmin, Tmax,Tmin,Tmax],
                      cmap='seismic', vmin=-valnorm,vmax=valnorm)
    
    # plot guiding lines for the quadrants
    ax.scatter(0,0, c='grey',s=50)
    ax.axvline(0, c='grey',linewidth=2, linestyle='dashed')
    ax.axhline(0, c='grey',linewidth=2, linestyle='dashed')
    ax.set_ylabel(r'$t_2$', fontsize=axis_font)
    ax.set_xlabel(r'$t_1$',fontsize=axis_font)
    ax.tick_params(axis='both', which='major', labelsize=axis_font-6)
    ax.tick_params(axis='both', which='minor', labelsize=axis_font-8)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(im, cax, shrink=0.9)    
    cbar.ax.tick_params(labelsize=axis_font-8)
    cbar.formatter.set_powerlimits((0, 0))
    return fig, ax

# load data
gn , sig, J2, Jq, dt, origin_i, mint, maxt =  kb.initialize_data(raw_route,raw_name,use_h5=True)
J2_mag = J2[-1]

# we have to trim the matrix because the full one is too heavy to plot
# get every third element
g_trim = gn[::3,::3]

# # PLOT REAL
fig, ax = component_contour(g_trim.real, mint, maxt,bwidth=bwidth, 
                            axis_font=base_font+5);

# ax.annotate('A', xy=(25,25),fontsize=base_font+15,c='green')
# ax.annotate(r'$\mathcal{{J}}_2={:.2f}$'.format(J2_mag), xy=(200,960),fontsize=base_font+10,c='green')
ax.set_xticks([0, 700, 1400])
ax.set_yticks([0, 700, 1400])
ax.set_aspect('equal')
plt.tight_layout()

plt.savefig(fig_route+'Re_two_time_Jtwo{}_sig{}.pdf'.format(J2_mag,sig),bbox_inches='tight',dpi=200)

# PLOT IMAGINARY
fig, ax = component_contour(g_trim.imag, mint, maxt,bwidth=bwidth, 
                            axis_font=base_font+5);

ax.set_xticks([0, 700, 1400])
ax.set_yticks([0, 700, 1400])

ax.set_aspect('equal')
plt.tight_layout()

plt.savefig(fig_route+'Im_two_time_Jtwo{}_sig{}.pdf'.format(J2_mag,sig),bbox_inches='tight',dpi=200)

