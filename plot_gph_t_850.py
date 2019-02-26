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
variable_name = 'gph_t_850'

print('Starting script to plot '+variable_name)

# Get the projection as system argument from the call so that we can 
# span multiple instances of this script outside
if not sys.argv[1:]:
    print('Projection not defined, falling back to default (euratl, it, de)')
    projections = ['euratl','it','de']
else:    
    projections=sys.argv[1:]

def main():
    """In the main function we basically read the files and prepare the variables to be plotted.
    This is not included in utils.py as it can change from case to case."""
    file = glob(input_file)
    print('Using file '+file[0])
    dset = xr.open_dataset(file[0])
    dset = dset.metpy.parse_cf()

    # Select 850 hPa level using metpy
    temp_850 = dset['t'].metpy.sel(vertical=850 * units.hPa).metpy.unit_array.to('degC')
    gph_500 = mpcalc.geopotential_to_height(dset['z'].metpy.sel(vertical=500 * units.hPa))

    lon, lat = get_coordinates(dset)
    lon2d, lat2d = np.meshgrid(lon, lat)

    time = pd.to_datetime(dset.time.values)
    cum_hour=np.array((time-time[0]) / pd.Timedelta('1 hour')).astype("int")

    levels_temp = np.arange(-30., 30., 1.)
    levels_gph = np.arange(4800., 5800., 70.)

    cmap = get_colormap('temp')
    
    for projection in projections:# This works regardless if projections is either single value or array
        fig = plt.figure(figsize=(figsize_x, figsize_y))
        ax  = plt.gca()        
        m, x, y =get_projection(lon2d, lat2d, projection, labels=True)

        # All the arguments that need to be passed to the plotting function
        args=dict(m=m, x=x, y=y, ax=ax, cmap=cmap,
                 temp_850=temp_850, gph_500=gph_500, levels_temp=levels_temp,
                 levels_gph=levels_gph, time=time, projection=projection, cum_hour=cum_hour)
        
        print('Pre-processing finished, launching plotting scripts')
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
        # Test if the image already exists, although this behaviour should be removed in the future
        # since we always want to overwrite old files.
        # if os.path.isfile(filename):
        #     print('Skipping '+str(filename))
        #     continue 

        cs = args['ax'].contourf(args['x'], args['y'], args['temp_850'][i], extend='both', cmap=args['cmap'],
                                    levels=args['levels_temp'])

        # Unfortunately m.contour with tri = True doesn't work because of a bug 
        c = args['ax'].contour(args['x'], args['y'], args['gph_500'][i], levels=args['levels_gph'],
                             colors='white', linewidths=0.5)

        labels = args['ax'].clabel(c, c.levels, inline=True, fmt='%4.0f' , fontsize=5)
        annotation(args['ax'],'Forecast for %s' % date.strftime('%d %b %Y at %H UTC') ,loc='upper left')
        annotation(args['ax'], 'Geopotential height @500hPa [m] and temperature @850hPa [C]' ,loc='lower left', fontsize=6)
        annotation_run(args['ax'], args['time'])

        if first:
            plt.colorbar(cs, orientation='horizontal', label='Temperature', pad=0.03, fraction=0.04)
        
        if debug:
            plt.show(block=True)
        else:
            plt.savefig(filename, **options_savefig)        
        
        remove_collections([c, cs, labels])

        first = False 

if __name__ == "__main__":
    main()