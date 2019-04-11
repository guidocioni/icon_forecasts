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
variable_name = 'precip_acc'

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

    # Select 850 hPa level using metpy
    precip_acc = dset['tp']
    mslp = dset['prmsl'].metpy.unit_array.to('hPa')

    lon, lat = get_coordinates(dset)
    lon2d, lat2d = np.meshgrid(lon, lat)

    time = pd.to_datetime(dset.time.values)
    cum_hour=np.array((time-time[0]) / pd.Timedelta('1 hour')).astype("int")

    levels_precip = (5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 150, 200, 250)
    levels_mslp = np.arange(mslp.min().astype("int"), mslp.max().astype("int"), 4.)

    cmap = truncate_colormap(plt.get_cmap('gist_stern_r'), 0., 0.9)
    cmap, norm = get_colormap_norm("rain_acc", levels_precip)

    for projection in projections:# This works regardless if projections is either single value or array
        fig = plt.figure(figsize=(figsize_x, figsize_y))
        ax  = plt.gca()
        m, x, y =get_projection(lon2d, lat2d, projection)
        m.shadedrelief(scale=0.7, alpha=0.8)

        # All the arguments that need to be passed to the plotting function
        args=dict(m=m, x=x, y=y, ax=ax,
                 precip_acc=precip_acc, mslp=mslp, levels_precip=levels_precip,
                 levels_mslp=levels_mslp, time=time, projection=projection, cum_hour=cum_hour,
                 cmap=cmap, norm=norm)
        
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

        cs = args['ax'].contourf(args['x'], args['y'], args['precip_acc'][i],
                         extend='max', cmap=args['cmap'], norm=args['norm'],
                         levels=args['levels_precip'])

        # Unfortunately m.contour with tri = True doesn't work because of a bug 
        c = args['ax'].contour(args['x'], args['y'], args['mslp'][i],
                             levels=args['levels_mslp'], colors='black', linewidths=1.)

        labels = args['ax'].clabel(c, c.levels, inline=True, fmt='%4.0f' , fontsize=6)

        maxlabels = plot_maxmin_points(args['ax'], args['x'], args['y'], args['mslp'][i],
                                        'max', 100, symbol='H', color='royalblue', random=True)
        minlabels = plot_maxmin_points(args['ax'], args['x'], args['y'], args['mslp'][i],
                                        'min', 100, symbol='L', color='coral', random=True)

        an_fc = annotation_forecast(args['ax'],args['time'][i])
        an_var = annotation(args['ax'], 'Accumulated precipitation and MSLP [hPa]' ,loc='lower left', fontsize=6)
        an_run = annotation_run(args['ax'], args['time'])

        if first:
            plt.colorbar(cs, orientation='horizontal', label='Accumulated precipitation [mm]', pad=0.035, fraction=0.035)
        
        if debug:
            plt.show(block=True)
        else:
            plt.savefig(filename, **options_savefig)        
        
        remove_collections([c, cs, labels, an_fc, an_var, an_run, maxlabels, minlabels])

        first = False 


if __name__ == "__main__":
    import time
    start_time=time.time()
    main()
    elapsed_time=time.time()-start_time
    print_message("script took " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

