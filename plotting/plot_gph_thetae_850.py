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
variable_name = 'gph_thetae_850'

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
    dset, time, cum_hour  = read_dataset(variables=['T', 'RELHUM', 'PMSL'])

    t_850 = dset['t'].metpy.sel(vertical=850 * units.hPa).load()
    rh_850 = dset['r'].metpy.sel(vertical=850 * units.hPa).load()
    theta_e = mpcalc.equivalent_potential_temperature(850 * units.hPa, t_850,
                                                      mpcalc.dewpoint_rh(t_850, rh_850 / 100.)).to('degC')

    theta_e = xr.DataArray(theta_e, coords= t_850.coords,
                           attrs={'standard_name': 'Equivalent potential temperature',
                                  'units': theta_e.units})
    del t_850
    del rh_850
    mslp = dset['prmsl'].load()
    mslp.metpy.convert_units('hPa')

    levels_temp = np.arange(-10, 80, .5)
    levels_mslp = np.arange(mslp.min().astype("int"), mslp.max().astype("int"), 2.5)

    cmap = plt.get_cmap('nipy_spectral')

    for projection in projections:  # This works regardless if projections is either single value or array
        fig = plt.figure(figsize=(figsize_x, figsize_y))

        ax = plt.gca()

        theta_e, mslp = subset_arrays([theta_e, mslp], projection)
        
        lon, lat = get_coordinates(mslp)
        lon2d, lat2d = np.meshgrid(lon, lat)
        
        m, x, y = get_projection(lon2d, lat2d, projection, labels=True)

        # All the arguments that need to be passed to the plotting function
        args = dict(x=x, y=y, ax=ax, cmap=cmap,
                    theta_e=theta_e, mslp=mslp, levels_temp=levels_temp,
                    levels_mslp=levels_mslp, time=time, projection=projection, cum_hour=cum_hour)

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

        cs = args['ax'].contourf(args['x'], args['y'], args['theta_e'][i], extend='both', cmap=args['cmap'],
                                 levels=args['levels_temp'])

        # Unfortunately m.contour with tri = True doesn't work because of a bug
        c = args['ax'].contour(args['x'], args['y'], args['mslp'][i], levels=args['levels_mslp'],
                               colors='white', linewidths=1.)

        labels = args['ax'].clabel(
            c, c.levels, inline=True, fmt='%4.0f', fontsize=6)

        maxlabels = plot_maxmin_points(args['ax'], args['x'], args['y'], args['mslp'][i],
                                       'max', 150, symbol='H', color='royalblue', random=True)
        minlabels = plot_maxmin_points(args['ax'], args['x'], args['y'], args['mslp'][i],
                                       'min', 150, symbol='L', color='coral', random=True)

        an_fc = annotation_forecast(args['ax'], args['time'][i])
        an_var = annotation(
            args['ax'], 'MSLP [hPa] and $\theta_e$ @850hPa [C]', loc='lower left', fontsize=6)
        an_run = annotation_run(args['ax'], args['time'])

        if first:
            plt.colorbar(cs, orientation='horizontal',
                         label='Temperature', pad=0.035, fraction=0.04)

        if debug:
            plt.show(block=True)
        else:
            plt.savefig(filename, **options_savefig)

        remove_collections([c, cs, labels, an_fc, an_var,
                            an_run, maxlabels, minlabels])

        first = False


if __name__ == "__main__":
    import time
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print_message("script took " + time.strftime("%H:%M:%S",
                                                 time.gmtime(elapsed_time)))
