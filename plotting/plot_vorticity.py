debug = False 
if not debug:
    import matplotlib
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
import xarray as xr 
import metpy.calc as mpcalc
from metpy.units import units
from glob import glob
import numpy as np
import pandas as pd
from multiprocessing import Pool
from functools import partial
import os 
from utils import *
import sys

# The one employed for the figure name when exported 
variable_name = 'vort_850'

print_message('Starting script to plot '+variable_name)

# Get the projection as system argument from the call so that we can 
# span multiple instances of this script outside
if not sys.argv[1:]:
    print_message('Projection not defined, falling back to default (euratl, it, de)')
    projections = ['euratl','it','de']
else:    
    projections=sys.argv[1:]

def main():
    """In the main function we basically read the files and prepare the variables to be plotted.
    This is not included in utils.py as it can change from case to case."""
    dset, time, cum_hour = read_dataset(variables=['U','V'])

    u = dset['u'].metpy.sel(vertical=850 * units.hPa).load()
    v = dset['v'].metpy.sel(vertical=850 * units.hPa).load()

    dx, dy = mpcalc.lat_lon_grid_deltas(dset['lon'], dset['lat'])
    vort = mpcalc.vorticity(u, v, dx[None, :, :], dy[None, :, :])
    vort = xr.DataArray(vort, coords=u.coords,
                        attrs={'standard_name': 'vorticity',
                               'units': vort.units})

    levels_vort = np.linspace(-0.0005, 0.0005, 51)

    cmap = plt.get_cmap('BrBG')
    
    for projection in projections:# This works regardless if projections is either single value or array
        fig = plt.figure(figsize=(figsize_x, figsize_y))
        
        ax  = plt.gca()

        u, v, vort = subset_arrays([u, v, vort], projection)

        lon, lat = get_coordinates(u)
        lon2d, lat2d = np.meshgrid(lon, lat)
        
        m, x, y = get_projection(lon2d, lat2d, projection, labels=True)

        # All the arguments that need to be passed to the plotting function
        args=dict(m=m, x=x, y=y, ax=ax, cmap=cmap,
                 vort=vort, u=u, v=v, levels_vort=levels_vort,
                 time=time, projection=projection, cum_hour=cum_hour)
        
        print_message('Pre-processing finished, launching plotting scripts')
        if debug:
            plot_files(time[1:2], **args)
        else:
            # Parallelize the plotting by dividing into chunks and processes 
            dates = chunks(time, chunks_size)
            plot_files_param=partial(plot_files, **args)
            p = Pool(processes)
            p.map(plot_files_param, dates)

def plot_files(dates, **args):
    # Using args we don't have to change the prototype function if we want to add other parameters!
    first = True
    for date in dates:
        # Find index in the original array to subset when plotting
        i = np.argmin(np.abs(date - args['time'])) 
        # Build the name of the output image
        filename = subfolder_images[args['projection']]+'/'+variable_name+'_%s.png' % args['cum_hour'][i]#date.strftime('%Y%m%d%H')#

        cs = args['ax'].contourf(args['x'], args['y'], args['vort'][i], extend='both', cmap=args['cmap'],
                                    levels=args['levels_vort'])

        # We need to reduce the number of points before plotting the vectors,
        # these values work pretty well
        if args['projection'] == 'euratl':
            density=25
            scale = 5e2
        else:
            density = 6
            scale = 3e2

        cv = args['ax'].quiver(args['x'][::density,::density],
                               args['y'][::density,::density],
                               args['u'][i,::density,::density],
                               args['v'][i,::density,::density],
                               scale=scale,
                               alpha=0.6, color='gray')

        an_fc = annotation_forecast(args['ax'],args['time'][i])
        an_var = annotation(args['ax'], 'Relative vorticity '+str(args['vort'].units) ,loc='lower left', fontsize=6)
        an_run =annotation_run(args['ax'], args['time'])

        if first:
            plt.colorbar(cs, orientation='horizontal', label='Vorticity', pad=0.035, fraction=0.035, format='%.0e')
        
        if debug:
            plt.show(block=True)
        else:
            plt.savefig(filename, **options_savefig)        
        
        remove_collections([cs, an_fc, an_var, an_run, cv])

        first = False 


if __name__ == "__main__":
    import time
    start_time=time.time()
    main()
    elapsed_time=time.time()-start_time
    print_message("script took " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
