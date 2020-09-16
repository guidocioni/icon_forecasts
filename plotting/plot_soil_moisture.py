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
variable_name = 'soil_moisture'

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
    file = glob(input_file)
    print_message('Using file '+file[0])
    dset = xr.open_dataset(file[0])
    dset = dset.metpy.parse_cf()

    saturation = xr.open_dataset(soil_saturation_file)['soil_saturation']

    # Convert to normal soil moisture units
    depths = dset['W_SO'].depth.values
    depths[0]=depths[0]*2
    rho_w=1000. 

    w_so = dset['W_SO'].copy() # Otherwise it overwrites it

    for i, depth in enumerate(depths):
        w_so[:,i,:,:] = dset['W_SO'][:,i,:,:]/(depth*rho_w)

    # Expand the saturation matrix to have 
    saturation = np.repeat(saturation.values[None,:,:], w_so.depth.shape[0], axis=0)
    saturation = np.repeat(saturation[None,:,:,:], w_so.time.shape[0], axis=0)
    w_so_sat = (w_so / saturation)*100.

    # Fix weird points with ice/rock
    w_so_sat = w_so_sat.where(w_so!=0,0.)[:,0,:,:]

    lon, lat = get_coordinates(dset)
    lon2d, lat2d = np.meshgrid(lon, lat)

    time = pd.to_datetime(dset.time.values)
    cum_hour=np.array((time-time[0]) / pd.Timedelta('1 hour')).astype("int")

    levels_sm = np.arange(0, 100, 10.)

    cmap = plt.get_cmap('terrain_r')
    
    for projection in projections:# This works regardless if projections is either single value or array
        fig = plt.figure(figsize=(figsize_x, figsize_y))
        ax  = plt.gca()        
        m, x, y =get_projection(lon2d, lat2d, projection, labels=True)

        # All the arguments that need to be passed to the plotting function
        args=dict(x=x, y=y, ax=ax, cmap=cmap,
                 w_so_sat=w_so_sat, levels_sm=levels_sm,
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

        cs = args['ax'].contourf(args['x'], args['y'], args['w_so_sat'][i], extend='both', cmap=args['cmap'],
                                    levels=args['levels_sm'])

        an_fc = annotation_forecast(args['ax'],args['time'][i])
        an_var = annotation(args['ax'], 'Soil Moisture saturation' ,loc='lower left', fontsize=6)
        an_run = annotation_run(args['ax'], args['time'])

        if first:
            plt.colorbar(cs, orientation='horizontal', label='Saturation [%]', pad=0.03, fraction=0.04)
        
        if debug:
            plt.show(block=True)
        else:
            plt.savefig(filename, **options_savefig)        
        
        remove_collections([cs, an_fc, an_var, an_run])

        first = False 


if __name__ == "__main__":
    import time
    start_time=time.time()
    main()
    elapsed_time=time.time()-start_time
    print_message("script took " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

