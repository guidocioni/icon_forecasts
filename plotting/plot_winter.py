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
variable_name = 'winter'

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
    dset, time, cum_hour  = read_dataset()

    # Compute rain and snow 
    # Note that we have to load since they are Dask arrays
    rain_acc = dset['RAIN_GSP'] + dset['RAIN_CON']
    snow_acc = dset['SNOW_GSP'] + dset['SNOW_CON']
    rain = rain_acc - rain_acc[0, :, :]
    snow = snow_acc - snow_acc[0, :, :]

    dset['SNOWLMT'].metpy.convert_units('m')
    snowlmt = dset['SNOWLMT']

    levels_snow = (1, 5, 10, 15, 20, 30, 40, 50, 70, 90, 120)
    levels_rain = (10, 15, 25, 35, 50, 75, 100, 125, 150)
    levels_snowlmt = np.arange(0., 3000., 500.)

    cmap_snow, norm_snow = get_colormap_norm("snow_discrete", levels_snow)
    cmap_rain, norm_rain = get_colormap_norm("rain", levels_rain)

    for projection in projections:# This works regardless if projections is either single value or array
        fig = plt.figure(figsize=(figsize_x, figsize_y))
        
        ax  = plt.gca()

        rain, snow, snowlmt = subset_arrays([rain, snow, snowlmt], projection)

        lon, lat = get_coordinates(rain)
        lon2d, lat2d = np.meshgrid(lon, lat)
        
        m, x, y = get_projection(lon2d, lat2d, projection)

        m.fillcontinents(color='lightgray',lake_color='whitesmoke', zorder=0)

        # All the arguments that need to be passed to the plotting function
        args=dict(m=m, x=x, y=y, ax=ax, rain=rain, snow=snow, snowlmt=snowlmt,
                 levels_snowlmt=levels_snowlmt, levels_rain=levels_rain, levels_snow=levels_snow,
                 time=time, projection=projection, cum_hour=cum_hour, norm_snow=norm_snow,
                 cmap_rain=cmap_rain, cmap_snow=cmap_snow, norm_rain=norm_rain)
        
        print_message('Pre-processing finished, launching plotting scripts')
        if debug:
            plot_files(time[-2:-1], **args)
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

        cs_rain = args['ax'].contourf(args['x'], args['y'], args['rain'][i],
                         extend='max', cmap=args['cmap_rain'], norm=args['norm_rain'],
                         levels=args['levels_rain'], alpha=0.8)
        cs_snow = args['ax'].contourf(args['x'], args['y'], args['snow'][i],
                         extend='max', cmap=args['cmap_snow'], norm=args['norm_snow'],
                         levels=args['levels_snow'], alpha=0.8)

        c = args['ax'].contour(args['x'], args['y'], args['snowlmt'][i], levels=args['levels_snowlmt'],
                             colors='red', linewidths=0.5)

        labels = args['ax'].clabel(c, c.levels, inline=True, fmt='%4.0f' , fontsize=5)  

        an_fc = annotation_forecast(args['ax'],args['time'][i])
        an_var = annotation(args['ax'], 'Snow and rain accumulated' ,loc='lower left', fontsize=6)
        an_run = annotation_run(args['ax'], args['time'])

        if first:
            if args['projection'] == "euratl": 
                x_cbar_0, y_cbar_0, x_cbar_size, y_cbar_size     = 0.15, 0.15, 0.35, 0.02
                x_cbar2_0, y_cbar2_0, x_cbar2_size, y_cbar2_size = 0.55, 0.15, 0.35, 0.02  
            elif args['projection'] == "de":
                x_cbar_0, y_cbar_0, x_cbar_size, y_cbar_size     = 0.17, 0.05, 0.32, 0.02
                x_cbar2_0, y_cbar2_0, x_cbar2_size, y_cbar2_size = 0.55, 0.05, 0.32, 0.02 
            elif args['projection'] == "it":
                x_cbar_0, y_cbar_0, x_cbar_size, y_cbar_size     = 0.18, 0.05, 0.3, 0.02
                x_cbar2_0, y_cbar2_0, x_cbar2_size, y_cbar2_size = 0.55, 0.05, 0.3, 0.02 
            
            ax_cbar = plt.gcf().add_axes([x_cbar_0, y_cbar_0, x_cbar_size, y_cbar_size])
            ax_cbar_2 = plt.gcf().add_axes([x_cbar2_0, y_cbar2_0, x_cbar2_size, y_cbar2_size])
            cbar_snow = plt.gcf().colorbar(cs_snow, cax=ax_cbar, orientation='horizontal',
             label='Snow')
            cbar_rain = plt.gcf().colorbar(cs_rain, cax=ax_cbar_2, orientation='horizontal',
             label='Rain')
            cbar_snow.ax.tick_params(labelsize=8) 
            cbar_rain.ax.tick_params(labelsize=8)
        
        if debug:
            plt.show(block=True)
        else:
            plt.savefig(filename, **options_savefig)        
        
        remove_collections([cs_rain, cs_snow, c, labels, an_fc, an_var, an_run])

        first = False 


if __name__ == "__main__":
    import time
    start_time=time.time()
    main()
    elapsed_time=time.time()-start_time
    print_message("script took " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

