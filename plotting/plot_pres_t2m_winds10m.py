import numpy as np
from multiprocessing import Pool
from functools import partial
from utils import *
import sys
import metpy.calc as mpcalc

debug = False
if not debug:
    import matplotlib
    matplotlib.use('Agg')

import matplotlib.pyplot as plt

# The one employed for the figure name when exported 
variable_name = 't_v_pres'

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
    dset  = read_dataset(variables=['U_10M', 'V_10M', 'T_2M', 'PMSL'],
                         projection=projection)

    dset['2t'] = dset['2t'].metpy.convert_units('degC').metpy.dequantify()
    dset['prmsl'] = dset['prmsl'].metpy.convert_units('hPa').metpy.dequantify()

    levels_t2m = np.arange(-25, 50, 1)

    cmap = get_colormap("temp")
    _ = plt.figure(figsize=(figsize_x, figsize_y))

    ax = plt.gca()
    # Get coordinates from dataset
    m, x, y = get_projection(dset, projection, labels=True)

    dset = dset.drop(['lon', 'lat']).load()

    levels_mslp = np.arange(dset['prmsl'].min().astype("int"),
                            dset['prmsl'].max().astype("int"), 4.)

    # All the arguments that need to be passed to the plotting function
    args = dict(x=x, y=y, ax=ax, cmap=cmap,
             levels_t2m=levels_t2m, levels_mslp=levels_mslp,
             time=dset.time)

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
    first = True
    for time_sel in dss.time:
        data = dss.sel(time=time_sel)
        data['prmsl'].values = mpcalc.smooth_n_point(data['prmsl'].values, n=9, passes=10)
        time, run, cum_hour = get_time_run_cum(data)
        # Build the name of the output image
        filename = subfolder_images[projection] + '/' + variable_name + '_%s.png' % cum_hour

        cs = args['ax'].contourf(args['x'], args['y'],
                                 data['2t'],
                                 extend='both',
                                 cmap=args['cmap'],
                                 levels=args['levels_t2m'])

        cs2 = args['ax'].contour(args['x'], args['y'],
                                 data['2t'],
                                 extend='both',
                                 levels=args['levels_t2m'][::5],
                                 linewidths=0.3,
                                 colors='gray', alpha=0.7)

        c = args['ax'].contour(args['x'], args['y'],
                               data['prmsl'],
                               levels=args['levels_mslp'],
                               colors='white', linewidths=1.)

        labels = args['ax'].clabel(c, c.levels, inline=True, fmt='%4.0f', fontsize=6)
        labels2 = args['ax'].clabel(cs2, cs2.levels, inline=True, fmt='%2.0f', fontsize=7)

        maxlabels = plot_maxmin_points(args['ax'], args['x'], args['y'], data['prmsl'],
                                       'max', 150, symbol='H', color='royalblue', random=True)
        minlabels = plot_maxmin_points(args['ax'], args['x'], args['y'], data['prmsl'], 
                                       'min', 150, symbol='L', color='coral', random=True)

        # We need to reduce the number of points before plotting the vectors,
        # these values work pretty well
        if projection == 'euratl':
            density = 25
            scale = 5e2
        else:
            density = 5
            scale = 3e2
        cv = args['ax'].quiver(args['x'][::density, ::density],
                               args['y'][::density, ::density],
                               data['10u'][::density, ::density],
                               data['10v'][::density, ::density],
                               scale=scale,
                               alpha=0.8, color='gray')

        an_fc = annotation_forecast(args['ax'], time)
        an_var = annotation(args['ax'],
            'MSLP [hPa], Winds@10m and Temperature@2m' ,loc='lower left', fontsize=6)
        an_run = annotation_run(args['ax'], run)
        logo = add_logo_on_map(ax=args['ax'],
                                zoom=0.1, pos=(0.95, 0.08))

        if first:
            plt.colorbar(cs, orientation='horizontal', label='Temperature [C]', pad=0.03, fraction=0.04)
        
        if debug:
            plt.show(block=True)
        else:
            plt.savefig(filename, **options_savefig)        
        
        remove_collections([cs, cs2, c, labels, labels2, an_fc, an_var, an_run, cv, maxlabels, minlabels, logo])

        first = False 


if __name__ == "__main__":
    import time
    start_time=time.time()
    main()
    elapsed_time=time.time()-start_time
    print_message("script took " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

