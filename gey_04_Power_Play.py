#------------------------------------------------
# Some numerical investigations of power laws
#------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import os
import earthquake.quake as quake
import pickle

#---------------------------
#  Run parameters
#---------------------------

block = True
plot_cult = False


pkl = 'pkl/'
png = './png/'
if not os.path.isdir(png): os.mkdir(png)
if not os.path.isdir(pkl): os.mkdir(pkl)

run_mod = False
run_sim = True

a_list = [0.5, 0.7]
#---------------------------
#  (1) PLot power laws: see the power law line for a=0.5 & 1.0
#---------------------------

# if run_mod:

#     dr, r2 = 50.0, 4000.0
#     nr = int(r2/dr) 
#     r = np.linspace(dr, r2, nr)

#     lw = 3.
#     density = True
#     finite = False  # infinity when r=0
#     kpp = False
#     fig = quake.power_play(r, a_list, density=density, 
#                         finite=finite, lw=lw, kpp=kpp, verbose=1)
#     fig.savefig(png + 'Power_Laws.png')



#----------------------------
#  (2) Run simulation
#----------------------------


# Grid for por and perm
x1, x2, dx = 0., 900., 100. 
y1, y2, dy = x1, x2, dx       # PorePy domains are square
nx, ny = int((x2-x1)/dx) + 1, int((y2-y1)/dy) + 1




if run_sim:

    density = False
    finite = False   # write the part with finite condition

    # Porosity vs log-permeability trend: log_perm = alfa*por + beta
    # alfa, beta = 14.50208301607354, -1.966584795783571
    alfa, beta = 50, -1.966584795783571
 
    for jj, a in enumerate(a_list):
        print(jj,a)
        mu0_phi = 0.15   # porosity mean
        sig0_phi = 0.02 # porosity variance (diagonal)
        rho1 = 0.9 # Correlation with nearest neighbor:
        # as its infinity for r=0, so assign a random value to stablise       
        n = 1000

        x = np.linspace(x1, x2, nx)
        y = np.linspace(y1, y2, ny) 

        dd = quake.power_model(x, y, a, mu0_phi, sig0_phi, alfa, beta, n, 
                               rho1=rho1, dist='lognorm', verbose=1, kplot=True)

        # Save the figs to png
        a_str = str(int(100*a)).zfill(3)
        f_roots = ['corrrel', 'pdf', 'por', 'perm','per_g']
        for kk, fig in enumerate(dd['figs']):
            print(kk)
            fname = f'Power_Law_{f_roots[kk]}_{jj}_a{a_str}.png'
            fig.savefig(png + fname)
            
            
        # dump to pickle file
        a_str = str(int(100*a)).zfill(3)
        fname = f'Power_Law_Reservoir_Models_a{a_str}.pkl'
        with open(pkl + fname, 'wb') as fid:
            pickle.dump(dd, fid)            
            

plt.show(block=block)




