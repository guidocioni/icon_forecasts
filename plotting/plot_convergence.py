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
variable_name = 'conv'

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
    dset, time, cum_hour = read_dataset(variables=['V_10M', 'U_10M'])

    ######## Variable read  #################################
    u = dset['10u'].load()
    v = dset['10v'].load()
    # We could move this computation in the projection part to make
    # it fast, but this would mean that we have to do it many times...
    dx, dy = mpcalc.lat_lon_grid_deltas(dset['lon'], dset['lat'])
    conv = - mpcalc.divergence(u, v, dx[None, :, :], dy[None, :, :])
    # Just add the attributes back to make a DataArray
    conv = xr.DataArray(conv, coords=u.coords,
                        attrs={'standard_name': 'convergence',
                               'units': conv.units})

    ######## Levels definition ###############################
    levels_conv = np.linspace(-0.0005, 0.0005, 21)
    ##########################################################

    ######## Colormaps definition ############################
    cmap = plt.get_cmap('BrBG')
    #########################################################

    for projection in projections:  # This works regardless if projections is either single value or array
        fig = plt.figure(figsize=(figsize_x, figsize_y))

        ax = plt.gca()

        u, v, conv = subset_arrays([u, v, conv], projection)

        lon, lat = get_coordinates(u)
        lon2d, lat2d = np.meshgrid(lon, lat)

        m, x, y = get_projection(lon2d, lat2d, projection, labels=True)

        # All the arguments that need to be passed to the plotting function
        args = dict(x=x, y=y, ax=ax, cmap=cmap,
                    conv=conv, u=u, v=v, levels_conv=levels_conv,
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

        cs = args['ax'].contourf(args['x'], args['y'], args['conv'][i], extend='both', cmap=args['cmap'],
                                 levels=args['levels_conv'])

        # We need to reduce the number of points before plotting the vectors,
        # these values work pretty well
        if args['projection'] == 'euratl':
            density = 25
            scale = None
        else:
            density = 5
            scale = 2e2
        cv = args['ax'].quiver(args['x'][::density, ::density],
                               args['y'][::density, ::density],
                               args['u'][i, ::density, ::density],
                               args['v'][i, ::density, ::density],
                               scale=scale,
                               alpha=0.7, color='gray')

        an_fc = annotation_forecast(args['ax'], args['time'][i])
        an_var = annotation(args['ax'], 'Convergence [' + str(args['conv'].units) + ']',
                            loc='lower left', fontsize=6)
        an_run = annotation_run(args['ax'], args['time'])
        logo = add_logo_on_map(ax=args['ax'],
                                zoom=0.1, pos=(0.95, 0.08))

        if first:
            plt.colorbar(cs, orientation='horizontal', label='Convergence [' + str(
                args['conv'].units) + ']', pad=0.035, fraction=0.035, format='%.0e')

        if debug:
            plt.show(block=True)
        else:
            plt.savefig(filename, **options_savefig)

        remove_collections([cs, an_fc, an_var, an_run, cv, logo])

        first = False


if __name__ == "__main__":
    import time
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print_message("script took " + time.strftime("%H:%M:%S",
                                                 time.gmtime(elapsed_time)))
