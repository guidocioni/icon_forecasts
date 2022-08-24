import numpy as np
from multiprocessing import Pool
from functools import partial
from utils import *
import sys
from computations import compute_geopot_height
import metpy.calc as mpcalc

debug = False
if not debug:
    import matplotlib
    matplotlib.use('Agg')

import matplotlib.pyplot as plt

# The one employed for the figure name when exported 
variable_name = 'gph_500_mslp'

print_message('Starting script to plot '+variable_name)

# Get the projection as system argument from the call so that we can 
# span multiple instances of this script outside
if not sys.argv[1:]:
    print_message(
        'Projection not defined, falling back to default (euratl)')
    projection = 'euratl'
else:
    projection = sys.argv[1]


def main():
    """In the main function we basically read the files and prepare the variables to be plotted.
    This is not included in utils.py as it can change from case to case."""
    dset = read_dataset(variables=['FI', 'PMSL'], level=[50000],
                        projection=projection)

    dset = compute_geopot_height(dset, zvar='z', level=50000)
    dset['prmsl'] = dset['prmsl'].metpy.convert_units('hPa').metpy.dequantify()

    levels_gph = np.arange(5000., 6000., 40.)

    cmap = get_colormap('gph')
    #cmap = truncate_colormap(cmap, 0.05, 0.9)

    _ = plt.figure(figsize=(figsize_x, figsize_y))

    ax  = plt.gca()
    # Get coordinates from dataset
    m, x, y = get_projection(dset, projection, labels=True)

    dset = dset.drop(['lon', 'lat', 'z']).load()

    levels_mslp = np.arange(dset['prmsl'].min().astype("int"),
                            dset['prmsl'].max().astype("int"), 4.)

    # All the arguments that need to be passed to the plotting function
    args = dict(x=x, y=y, ax=ax, cmap=cmap,
                levels_gph=levels_gph,
                levels_mslp=levels_mslp)

    print_message('Pre-processing finished, launching plotting scripts')
    if debug:
        plot_files(dset.isel(time=slice(0, 2)), **args)
    else:
        # Parallelize the plotting by dividing into chunks and processes 
        dss = chunks_dataset(dset, chunks_size)
        plot_files_param = partial(plot_files, **args)
        p = Pool(processes)
        p.map(plot_files_param, dss)


def plot_files(dss, **args):
    # Using args we don't have to change the prototype function if we want to add other parameters!
    first = True
    for time_sel in dss.time:
        data = dss.sel(time=time_sel)
        data['prmsl'].values = mpcalc.smooth_n_point(data['prmsl'].values, n=9, passes=10)
        time, run, cum_hour = get_time_run_cum(data)
        # Build the name of the output image
        filename = subfolder_images[projection] + '/' + variable_name + '_%s.png' % cum_hour

        cs = args['ax'].contourf(args['x'], args['y'],
                                 data['geop'],
                                 extend='both', 
                                 cmap=args['cmap'],
                                 levels=args['levels_gph'])

        c = args['ax'].contour(args['x'], args['y'],
                               data['prmsl'], 
                               levels=args['levels_mslp'],
                               colors='white', 
                               linewidths=1.5)

        labels = args['ax'].clabel(c, c.levels, inline=True, fmt='%4.0f', 
                                   fontsize=6)

        maxlabels = plot_maxmin_points(args['ax'], args['x'], args['y'], data['prmsl'],
                                        'max', 100, symbol='H', color='royalblue', random=True)
        minlabels = plot_maxmin_points(args['ax'], args['x'], args['y'], data['prmsl'],
                                        'min', 100, symbol='L', color='coral', random=True)

        an_fc = annotation_forecast(args['ax'], time)
        an_var = annotation(args['ax'], 
            'Geopotential height @500hPa [m] and MSLP (hPa)',
             loc='lower left', fontsize=6)
        an_run = annotation_run(args['ax'], run)
        logo = add_logo_on_map(ax=args['ax'],
                                zoom=0.1, pos=(0.95, 0.08))

        if first:
            plt.colorbar(cs, orientation='horizontal', label='Geopotential height [m]', pad=0.03, fraction=0.04)

        if debug:
            plt.show(block=True)
        else:
            plt.savefig(filename, **options_savefig)        

        remove_collections([c, cs, labels, an_fc, an_var, an_run, maxlabels, minlabels, logo])

        first = False 


if __name__ == "__main__":
    import time
    start_time=time.time()
    main()
    elapsed_time=time.time()-start_time
    print_message("script took " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
