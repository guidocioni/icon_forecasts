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
    dset, time, cum_hour = read_dataset()

    saturation = xr.open_dataset(soil_saturation_file)['soil_saturation']

    # Convert to normal soil moisture units
    w_so = dset['W_SO'].squeeze()

    rho_w = 1000.
    w_so = w_so / (w_so.depth * 2 * rho_w)  

    w_so_sat = (w_so.values[:, :, :] / saturation.values[None, :, :]) * 100.

    w_so_sat = xr.DataArray(w_so_sat, coords=w_so.coords,
                           attrs={'standard_name': 'Soil moisture saturation',
                                  'units': '%'})

    # Fix weird points with ice/rock
    w_so_sat = w_so_sat.where(w_so != 0, 0.)

    levels_sm = np.arange(0, 100, 10.)

    cmap = plt.get_cmap('terrain_r')

    for projection in projections:  # This works regardless if projections is either single value or array
        fig = plt.figure(figsize=(figsize_x, figsize_y))

        ax = plt.gca()
        w_so_sat = subset_arrays([w_so_sat], projection)[0]

        lon, lat = get_coordinates(w_so_sat)
        lon2d, lat2d = np.meshgrid(lon, lat)

        m, x, y = get_projection(lon2d, lat2d, projection, labels=True)

        # All the arguments that need to be passed to the plotting function
        args = dict(x=x, y=y, ax=ax, cmap=cmap,
                    w_so_sat=w_so_sat, levels_sm=levels_sm,
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

        cs = args['ax'].contourf(args['x'], args['y'], args['w_so_sat'][i], extend='both', cmap=args['cmap'],
                                 levels=args['levels_sm'])

        an_fc = annotation_forecast(args['ax'], args['time'][i])
        an_var = annotation(
            args['ax'], 'Soil Moisture saturation', loc='lower left', fontsize=6)
        an_run = annotation_run(args['ax'], args['time'])

        if first:
            plt.colorbar(cs, orientation='horizontal',
                         label='Saturation [%]', pad=0.03, fraction=0.04)

        if debug:
            plt.show(block=True)
        else:
            plt.savefig(filename, **options_savefig)

        remove_collections([cs, an_fc, an_var, an_run])

        first = False


if __name__ == "__main__":
    import time
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print_message("script took " + time.strftime("%H:%M:%S",
                                                 time.gmtime(elapsed_time)))
