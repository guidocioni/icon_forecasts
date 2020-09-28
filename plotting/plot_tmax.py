debug = False 
if not debug:
    import matplotlib
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool
from functools import partial
import os 
from utils import *
import sys

# The one employed for the figure name when exported 
variable_name = 'tmax'

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
    dset, time, cum_hour = read_dataset(variables=['TMAX_2M'])

    dset['TMAX_2M'].metpy.convert_units('degC')
    tmax2m = dset['TMAX_2M']

    levels_t2m = np.arange(-25, 40, 1)

    cmap = get_colormap("temp")
    
    for projection in projections:# This works regardless if projections is either single value or array
        print_message('Projection = %s' % projection)
        fig = plt.figure(figsize=(figsize_x, figsize_y))
        
        ax  = plt.gca()        

        tmax2m = subset_arrays([tmax2m], projection)[0]

        lon, lat = get_coordinates(tmax2m)
        lon2d, lat2d = np.meshgrid(lon, lat)

        m, x, y = get_projection(lon2d, lat2d, projection, labels=True)

        # All the arguments that need to be passed to the plotting function
        args=dict(m=m, x=x, y=y, ax=ax, cmap=cmap,
                 tmax2m=tmax2m, levels_t2m=levels_t2m,
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
        filename = subfolder_images[args['projection']]+'/'+variable_name+'_%s.png' % args['cum_hour'][i]

        cs = args['ax'].contourf(args['x'], args['y'], args['tmax2m'][i], extend='both', cmap=args['cmap'],
                                    levels=args['levels_t2m'])
        
        # plot every -th element
        if args['projection']=="euratl":
            density = 29
        elif args['projection']=="it":
            density = 6
        elif args['projection']=="de":
            density = 5
        
        vals = add_vals_on_map(args['ax'], args['m'], args['tmax2m'][i], args['levels_t2m'], cmap=args['cmap'],
                                density=density)

        an_fc = annotation_forecast(args['ax'],args['time'][i])
        an_var = annotation(args['ax'], 'Maximum 2m Temperature in last 6 hours' ,loc='lower left', fontsize=6)
        an_run = annotation_run(args['ax'], args['time'])

        if first:
            plt.colorbar(cs, orientation='horizontal', label='Temperature [C]', pad=0.03, fraction=0.04)
        
        if debug:
            plt.show(block=True)
        else:
            plt.savefig(filename, **options_savefig)        
        
        remove_collections([cs, an_fc, an_var, an_run, vals])

        first = False 

if __name__ == "__main__":
    import time
    start_time=time.time()
    main()
    elapsed_time=time.time()-start_time
    print_message("script took " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
