debug = False
if not debug:
    import matplotlib
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
from metpy.units import units
import numpy as np
from multiprocessing import Pool
from functools import partial
import os
from utils import *
import sys

# The one employed for the figure name when exported
variable_name = 'cape'

print_message('Starting script to plot ' + variable_name)

# Get the projection as system argument from the call so that we can
# span multiple instances of this script outside
if not sys.argv[1:]:
    print_message(
        'Projection not defined, falling back to default (euratl, it, de)')
    projections = ['euratl', 'it', 'de']
else:
    projections = sys.argv[1:]


def main():
    """In the main function we basically read the files and prepare the variables to be plotted.
    This is not included in utils.py as it can change from case to case."""
    dset, time, cum_hour  = read_dataset(variables=['CAPE_ML', 'U', 'V'])

    ######## Variable read  #################################
    cape = dset['CAPE_ML'].squeeze()
    # Select 850 hPa level using metpy
    uwind_850 = dset['u'].metpy.sel(vertical=850 * units.hPa)
    vwind_850 = dset['v'].metpy.sel(vertical=850 * units.hPa)
    #########################################################

    ######## Levels definition ###############################
    levels_cape = np.arange(250., 5000., 50.)
    #########################################################

    ######## Colormaps definition ############################
    cmap = get_colormap("winds")
    #########################################################

    for projection in projections:  # This works regardless if projections is either single value or array
        fig = plt.figure(figsize=(figsize_x, figsize_y))
        
        ax = plt.gca()

        # Subset array to avoid memory/CPU extra usage when plotting
        cape, uwind_850, vwind_850 = subset_arrays([cape, uwind_850, vwind_850],
                                                     projection)

        # Get coordinates from one of the variables
        lon, lat = get_coordinates(cape)
        lon2d, lat2d = np.meshgrid(lon, lat)
    
        m, x, y = get_projection(lon2d, lat2d, projection, labels=True)
        # additional maps adjustment for this map
        m.fillcontinents(color='lightgray', lake_color='whitesmoke', zorder=0)

        # All the arguments that need to be passed to the plotting function
        # we pass only arrays to avoid the pickle problem when unpacking in multiprocessing
        args = dict(x=x, y=y, ax=ax, cmap=cmap,
                    cape=cape.values, uwind_850=uwind_850.values, vwind_850=vwind_850.values,
                    levels_cape=levels_cape,
                    time=time, projection=projection, cum_hour=cum_hour)

        print_message('Pre-processing finished, launching plotting scripts')
        if debug:
            plot_files(time[1:2], **args)
        else:
            # Parallelize the plotting by dividing into chunks and processes
            dates = chunks(time, chunks_size)
            plot_files_param = partial(plot_files, **args)
            p = Pool(processes)
            p.map(plot_files_param, dates)


def plot_files(dates, **args):
    # Using args we don't have to change the prototype function if we want to add other parameters!
    first = True
    for date in dates:
        # Find index in the original array to subset when plotting
        i = np.argmin(np.abs(date - args['time']))
        # Build the name of the output image
        filename = subfolder_images[args['projection']] + '/' + variable_name + \
            '_%s.png' % args['cum_hour'][i]  # date.strftime('%Y%m%d%H')#

        cs = args['ax'].contourf(args['x'], args['y'], args['cape'][i], extend='max', cmap=args['cmap'],
                                    levels=args['levels_cape'])

        # We need to reduce the number of points before plotting the vectors,
        # these values work pretty well
        if args['projection'] == 'euratl':
            density = 25
            scale = None
        else:
            density = 5
            scale = 2.5e2
        cv = args['ax'].quiver(args['x'][::density, ::density],
                               args['y'][::density, ::density],
                               args['uwind_850'][i, ::density, ::density],
                               args['vwind_850'][i,::density, ::density],
                               scale=scale,
                               alpha=0.8, color='gray')

        an_fc = annotation_forecast(args['ax'], args['time'][i])
        an_var = annotation(
            args['ax'], 'Convective Available Potential Energy and Winds @ 850 hPa', loc='lower left', fontsize=6)
        an_run = annotation_run(args['ax'], args['time'])

        if first:
            plt.colorbar(cs, orientation='horizontal',
                         label='CAPE [J/kg]', pad=0.03, fraction=0.04, extend="max")

        if debug:
            plt.show(block=True)
        else:
            plt.savefig(filename, **options_savefig)

        remove_collections([cs, an_fc, an_var, an_run, cv])

        first = False


if __name__ == "__main__":
    import time
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print_message("script took " + time.strftime("%H:%M:%S",
                                                 time.gmtime(elapsed_time)))
