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
from matplotlib.colors import from_levels_and_colors
import seaborn as sns

# The one employed for the figure name when exported 
variable_name = 'hsnow'

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

    hsnow_acc = dset['sd']
    hsnow = hsnow_acc*0.
    for i, _ in enumerate(hsnow_acc[1:]):
        hsnow[i] = (hsnow_acc[i] - hsnow_acc[0])*100.
    hsnow = hsnow.where((hsnow>0.5) | (hsnow<-0.5))

    snowlmt = dset['SNOWLMT'].metpy.unit_array.to('m')

    lon, lat = get_coordinates(dset)
    lon2d, lat2d = np.meshgrid(lon, lat)

    time = pd.to_datetime(dset.time.values)
    cum_hour=np.array((time-time[0]) / pd.Timedelta('1 hour')).astype("int")

    levels_hsnow = (-50, -40, -30, -20, -10, -5, -2.5, -2, -1, -0.5, 0, 0.5, 1, 2, 2.5, 5, 10, 20, 30, 40, 50)
    levels_snowlmt = np.arange(0., 3000., 500.)

    cmap, norm = from_levels_and_colors(levels_hsnow, sns.color_palette("PuOr", n_colors=len(levels_hsnow)+1),
                                                    extend='both')
    
    for projection in projections:# This works regardless if projections is either single value or array
        fig = plt.figure(figsize=(figsize_x, figsize_y))
        ax  = plt.gca()        
        m, x, y =get_projection(lon2d, lat2d, projection, labels=True)
        img=m.arcgisimage(service='World_Shaded_Relief', xpixels = 1000, verbose=False)
        img.set_alpha(0.8)

        # All the arguments that need to be passed to the plotting function
        args=dict(m=m, x=x, y=y, ax=ax, cmap=cmap, norm=norm,
                 hsnow=hsnow, snowlmt=snowlmt, levels_hsnow=levels_hsnow,
                 levels_snowlmt=levels_snowlmt, time=time, projection=projection, cum_hour=cum_hour)
        
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

        cs = args['ax'].contourf(args['x'], args['y'], args['hsnow'][i], extend='both', cmap=args['cmap'],
                                    norm=args['norm'], levels=args['levels_hsnow'])

        # Unfortunately m.contour with tri = True doesn't work because of a bug 
        c = args['ax'].contour(args['x'], args['y'], args['snowlmt'][i], levels=args['levels_snowlmt'],
                             colors='red', linewidths=0.5)

        labels = args['ax'].clabel(c, c.levels, inline=True, fmt='%4.0f' , fontsize=5)
        
        an_fc = annotation_forecast(args['ax'],args['time'][i])
        an_var = annotation(args['ax'], 'Snow depth change since initialization time' ,loc='lower left', fontsize=6)
        an_run = annotation_run(args['ax'], args['time'])

        if first:
            plt.colorbar(cs, orientation='horizontal', label='Snow depth change [m]', pad=0.035, fraction=0.03)
        
        if debug:
            plt.show(block=True)
        else:
            plt.savefig(filename, **options_savefig)        
        
        remove_collections([c, cs, labels, an_fc, an_var, an_run])

        first = False 

if __name__ == "__main__":
    main()
