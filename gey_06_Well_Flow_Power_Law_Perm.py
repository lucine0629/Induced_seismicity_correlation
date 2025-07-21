#---------------------------------------------------------------------
#
#---------------------------------------------------------------------
""" Notes by KetilH:

Where to find stuff in the PorePy codes

permeability:
Defined in porepy.models.constitutive_laws.py
Replaced below in class ModifiedPerm

"""


#--------------------------------------------------------------------
# Single phase flow example. Based on sc_06_single_phase_flow.ipynb
#--------------------------------------------------------------------

from __future__ import annotations

import porepy as pp
import numpy as np
import pickle
from porepy.models.fluid_mass_balance import SinglePhaseFlow

import matplotlib.pyplot as plt

import earthquake.curvature as crv

#-------------------------------
#  Run parameters (square model)
#-------------------------------

block = False

png = 'porepy_png/'

ndim = 2        # Dimension (2D)
xscl = 1.0e3    # 1 if m, 1000 if km 

grid_type = 'cartesian'    # 'cartesian' or 'simplex'
cell_value = 'pressure' # Always

################################################
# # Permeability model: Hardcoded test model
# x_size = y_size = 2.0*xscl      # grid size
# dx_grid = dy_grid = 0.1*xscl   # grid spacing

# nx = ny = int(x_size/dx_grid)
# nnn = nx*ny
# perm_cartesian = 1.0*np.ones(nnn)

# i1, i2= nx//5, 4*nx//5
# for j in [nx//2-1, nx//2]:
#     perm_cartesian[j*nx+i1:j*nx+i2] = 5.
#################################################

#-----------------------------------------
#  Read the perm model from pickle file
#  Domain must be square
#-----------------------------------------

pkl = 'pkl/'

a = 0.7
a_str = 'a' + str(int(100*a)).zfill(3)
fname = f'Power_Law_Reservoir_Models_{a_str}.pkl'
with open(pkl + fname, 'rb') as fid:
    dd = pickle.load(fid)

nx = ny = dd['x'].shape[0]
nnn = nx*ny
dx_grid = dy_grid = np.diff(dd['x'])[0]
x_size = y_size = nx*dx_grid # Perm nodes are cell centered

#--------------------------------
#
#  Loop over all perm models
#
#--------------------------------

#kmod = 14 
# for kmod in [14]:

key_perm = 'perm_mods'
nmod = len(dd[key_perm])
kmod_list = [j for j in range(nmod)]
# kmod_list = [0]
# kmod_list = [0,2,4,7,8,14,18,19]
# kmod_list = [7,19]
# for kmod in kmod_list[::-1]:
for kmod in [1,2,3]:
    idd_mod = dd['ind_rand'][kmod]
    
    # PoerePy wants a 1d vetor of values
    # perm_cartesian = dd['perm_g_mods'][kmod].ravel()
    perm_cartesian = dd[key_perm][kmod].ravel()
    
    #------------------------------------
    #  PorePy setup
    #-----------------------------------
    
    # When grid_type is 'cartesian', this is just a simple copy
    perm_model = perm_cartesian
    
    # Setting pressur BC as a difference, i.e. zero on right side
    P_west, P_east = 0.5*xscl, 0       # Pressure (Diriclet) BC
    
    # Define fracture locations (must be parallell to x or y if grid_type='cartesian')
    frac_list = []
    # frac_list = [
    #     np.array([[200, 200], [200, 600]])
    #     ]
    
    # Units:
    x_unit = y_unit = 'm' # x unit; only 'm' works
    P_unit = 'Pa'
    
    #-------------------------------
    #  Modify the geometry
    #-------------------------------
    
    from porepy.applications.md_grids.domains import nd_cube_domain
    
    # Define a mixin class
    class ModifiedGeometry:
        def set_domain(self) -> None:
            """Defining a two-dimensional square domain with sidelength x_size [x_unit]"""
            size = self.solid.convert_units(x_size, x_unit)
            self._domain = nd_cube_domain(ndim, size)
    
        # Change this to 1 or 2 vertical fractures
        def set_fractures(self) -> None:
            # Setting fractures
            fractures = []
            for frac in frac_list:
                frac_points = self.solid.convert_units(frac, x_unit)
                fractures.append(pp.LineFracture(frac_points))
            
            self._fractures = fractures
    
    
    
        #     frac_1_points = self.solid.convert_units(
        #     np.array([[100, 200], [100, 600]]), "m"
        # )
        #     frac_1 = pp.LineFracture(frac_1_points)
        #     self._fractures = [frac_1]

    
    
        # Change this to use the Cartesian grid
        def grid_type(self) -> str:
            """Choosing the grid type for our domain.
    
            If we have a diagonal fracture we cannot use a cartesian grid.
            Cartesian grid is the default grid type
            """
            return self.params.get("grid_type", grid_type)
    
        def meshing_arguments(self) -> dict:
            """Meshing arguments for md-grid creation. Set the cell size"""
            # cell_size = self.solid.convert_units(0.25, "m")
            cell_size = self.solid.convert_units(dx_grid, x_unit)
            mesh_args: dict[str, float] = {"cell_size": cell_size}
            return mesh_args
    
    #----------------------------------------------
    # Add wells using the receipe from Eirik K.
    #----------------------------------------------

    class ModifiedSource:
        
        def fluid_source(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
                                                
            # Previosuly defined sources (if any)
            source = super().fluid_source(subdomains)

            # GEt the matrix subdmain (always 0)
            sd = subdomains[0]              

            # Make a producer/injector pair
            x_wells = [500, 750]
            y_wells = [200, 750]
            RATE = 1e5
            RATES = [RATE, -RATE] 
            n_well = len(x_wells)
            
            for jw in range(n_well):
                
                xw, yw = x_wells[jw], y_wells[jw]
                well_ind = sd.closest_cell(np.array([xw, yw, 0]).reshape((-1, 1)))
            
                inj_array = np.zeros(sd.num_cells)
                inj_array[well_ind] = RATES[jw]

                wrapped_array = pp.wrap_as_dense_ad_array(inj_array, name='cellwise_sources')

                # Accumulate source term
                source += wrapped_array
                
            return source 
    
    #-----------------------------------------------------
    #  Define a new class with the ModifiedGeometry mixin
    #-----------------------------------------------------
    
    class SinglePhaseFlowGeometry(
        ModifiedGeometry,
        SinglePhaseFlow):
        """Combining the modified geometry and the default model."""
        ...
    
    #-----------------------------------------------
    # Set boundary conditions
    #-----------------------------------------------
    
    from porepy.models.fluid_mass_balance import BoundaryConditionsSinglePhaseFlow
    
    class ModifiedBC(BoundaryConditionsSinglePhaseFlow):
        def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
            """Assign dirichlet to the west and east boundaries. The rest are Neumann by default."""
            bounds = self.domain_boundary_sides(sd)
            bc = pp.BoundaryCondition(sd, bounds.west + bounds.east, "dir")
            return bc
    
        def bc_values_pressure(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
            """Zero bc value on top and bottom, 5 on west side, 2 on east side."""
            bounds = self.domain_boundary_sides(boundary_grid)
            values = np.zeros(boundary_grid.num_cells)
            # See section on scaling for explanation of the conversion.
            values[bounds.west] = self.fluid.convert_units(P_west, "Pa")
            values[bounds.east] = self.fluid.convert_units(P_east, "Pa")
            return values
    
    #----------------------------------------------------------------
    #  Define a new class with the ModifiedGeometry and BC mixins
    #----------------------------------------------------------------
    
    class SinglePhaseFlowGeometryBC(
        ModifiedGeometry,
        ModifiedBC,
        SinglePhaseFlow):
        """Adding both geometry and modified boundary conditions to the default model."""
        ...
    
    #----------------------------------------------------------------
    #  Define a class for inhomogeneous permebility
    #  Based on hints from Eirik Keilegavlen at Univ. of Bergen
    #----------------------------------------------------------------
    
    class ModifiedPerm:
        """ Define a class to make 3D perm models. 
        
        Based on hints from Eirik Keilegavlen, UiB.
        
        The permability function defined below is called inside
        model.prepare_simulation(), which is the first function called  in
        porepy.models.run_models.run_time_dependent_model(model, params)
        """
        
        def permeability(self, subdomains: list[pp.Grid]) -> pp.SecondOrderTensor:
            """Function overriding default by mixin"""
            
            # Number of cells in the grids
            num_cells = [sd.num_cells for sd in subdomains]
            num_cells_total = sum(num_cells)
            
            perm = 1.0*np.ones(num_cells_total)
            if   num_cells_total == nnn:
                perm = perm_model
                
            else:
                pass
                
            print(f'permeability: num_cells = {num_cells}')
            
            # Pack the perm values
            perm_ad = pp.wrap_as_dense_ad_array(perm, name='permeability')
            
            return self.isotropic_second_order_tensor(subdomains, perm_ad)
    
    #-----------------------------------------------------------------------------
    #  Define a new class with the ModifiedGeometry ModifiedPerm and BC mixins
    #-----------------------------------------------------------------------------
    
    class SinglePhaseFlowGeometryPermBC(
        ModifiedGeometry,
        ModifiedSource,
        ModifiedPerm,
        ModifiedBC,
        SinglePhaseFlow):
        """Adding both modified geometry, perm and BC to the default model."""
        ...
    
    #----------------------------------------------------------------
    #  Modify solid and fluid medium parameterss
    #----------------------------------------------------------------
    
    fluid_constants = pp.FluidConstants({"viscosity": 0.1, "density": 0.2})
    
    # solid_constants = pp.SolidConstants({"permeability": 0.5, "porosity": 0.25})
    solid_constants = pp.SolidConstants({"porosity": 0.25})
    
    material_constants = {"fluid": fluid_constants, "solid": solid_constants}
    params = {"material_constants": material_constants}
    
    #-----------------------------------------------------------------------------
    # Create model with modified parameters and run simulation.
    # there are 2 mixins (ModifiedGeometry,ModifiedBC) modifying SinglePhaseFlow
    #-----------------------------------------------------------------------------
    
    model = SinglePhaseFlowGeometryPermBC(params)
    pp.run_time_dependent_model(model, params)
    
    # pp.plot_grid(model.mdg, cell_value, figsize=(10, 8), linewidth=0.25, 
    #              title="Pressure", plot_2d=True)
    
    #---------------------------------
    #  PLot results
    #---------------------------------
    
    ### Added by KetilH
    
    print(f'num_subdomains = {model.mdg.num_subdomains()}')
    print(f'dim_min = {model.mdg.dim_min()}')
    print(f'dim_max = {model.mdg.dim_max()}')
    
    ### Print some shit
    for dim in [1,10,100]:
        print(f'dim = {dim}, cell_value={cell_value}:')
        sd_list = model.mdg.subdomains(return_data=True, dim=dim)
        for sd, sd_data in sd_list:
            print(sd_data.get(pp.TIME_STEP_SOLUTIONS, {}).get(cell_value, {}).get(0,None))
    
    ### Print more shit
    for index, (sd, sd_data) in enumerate(model.mdg.subdomains(return_data=True)):
        print(f'index = {index}, cell_value={cell_value}:')
        print(sd_data.get(pp.TIME_STEP_SOLUTIONS, {}).get(cell_value, {}).get(0,None))
    
    
    # PLot the location of cell centers
    # subs_list = model.mdg.subdomains(return_data=True)
    # nsub = len(subs_list)
    # lab_list = ['main_grid'] + [f'fracture_{jj}' for jj in range(nsub-1)]
    # fig, ax = plt.subplots(1)
    
    # for jj, (sd, sd_data) in enumerate(subs_list):
    #     xi, yi, zi = sd.cell_centers
    #     ax.scatter(xi, yi, marker='o', label=lab_list[jj])
        
    # ax.legend()
    # ax.set_title('Subdomains')
    # fig.savefig('Subdomains.png')    
    
    # Get the 2D output pressure data
    sd, sd_data = model.mdg.subdomains(return_data=True)[0]
    wrk = sd_data[pp.TIME_STEP_SOLUTIONS]
    pres = sd_data[pp.TIME_STEP_SOLUTIONS][cell_value][0]
    xc, yc, zc = sd.cell_centers
    
    # Make some 2D arrays for plotting:
    xarr = np.linspace(0,x_size, nx)
    nn = int(np.sqrt(nnn))
    qx = xc.reshape(nn,nn)
    qy = yc.reshape(nn,nn)
    pres_2d = pres.reshape(nn,nn)
    perm_2d = perm_model.reshape(nn,nn)
    
    # # PLot pressure curves
    # fig, ax = plt.subplots(1)
    # for jy in range(ny//2):
    #     ax.plot(xarr, pres_2d[jy,:], '-', label=f'y={jy*dy_grid:.2f} {y_unit}')
    
    # ax.legend()
    # ax.set_xlabel(f'x [{x_unit}]')
    # ax.set_ylabel(f'P [{P_unit}]')
    # ax.set_title('Pressure at y=const')
    # fig.savefig('Pressure_Curves.png')
    
    # PLot perm model, pressure and flow arrows
    cmap = 'viridis'
    figa, axs = plt.subplots(1,2, figsize=(15,6))
    
    ax = axs.ravel()[0]
    im = ax.pcolormesh(qx, qy, perm_2d, cmap=cmap)
    cb = ax.figure.colorbar(im, ax=ax)
    ax.set_title('log10 Permeability')
    
    ax = axs.ravel()[1]
    im = ax.pcolormesh(qx, qy, pres_2d, cmap=cmap)
    cb = ax.figure.colorbar(im, ax=ax)
    ax.set_title('Pressure: variable perm')
    
    # Compute pressure gradient (works only on catesian grid)
    kh = crv.grad_and_hess(pres_2d, qx, qy, ret_xy=True)
    flow_x = -perm_2d*kh['gx']
    flow_y = -perm_2d*kh['gy']
    
    ax = axs.ravel()[1]
    ax.quiver(qx, qy, flow_x, flow_y, color='r') 
    #          angles='xy', scale_units='xy', scale=10)
    
    # Add some axis text
    for kx in axs.ravel():
        kx.axis('scaled')
        kx.set_xlabel(f'x [{x_unit}]')
        kx.set_ylabel(f'y [{y_unit}]')
    
    figa.suptitle(f'({kmod}) Model {idd_mod}')
    
    figa.tight_layout(pad=2.)
    figa.savefig(png + f'{a_str}_{key_perm}_Perm_and_Flow_well_{idd_mod}.png')

plt.show(block=block)

# PLot Darcy flux for the 2d grid: This doesn't look right
# sd0 = model.mdg.subdomains()[0] # Make it a list
# darcy_flux = model.darcy_flux([sd0]).value(model.equation_system)
# unit_normal = sd0.face_normals/sd0.face_areas  
# flux_vector = darcy_flux*unit_normal
# pp.plot_grid(sd0, vector_value=flux_vector)

# Plot the flow vectors


###################################################