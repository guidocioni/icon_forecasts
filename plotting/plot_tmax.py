import numpy as np
from multiprocessing import Pool
from functools import partial
from utils import *
import sys

debug = False
if not debug:
    import matplotlib
    matplotlib.use('Agg')

import matplotlib.pyplot as plt

# The one employed for the figure name when exported 
variable_name = 'tmax'

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
    dset = read_dataset(variables=['TMAX_2M'], projection=projection)
    dset['TMAX_2M'] = dset['TMAX_2M'].metpy.convert_units('degC').metpy.dequantify()

    levels_t2m = np.arange(-25, 50, 1)

    cmap = get_colormap("temp")

    _ = plt.figure(figsize=(figsize_x, figsize_y))

    ax  = plt.gca()
    m, x, y = get_projection(dset, projection, labels=True)

    # All the arguments that need to be passed to the plotting function
    args=dict(x=x, y=y, ax=ax, cmap=cmap,
             levels_t2m=levels_t2m,
             time=dset.time)

    print_message('Pre-processing finished, launching plotting scripts')
    if debug:
        plot_files(dset.isel(time=slice(0, 2)), **args)
    else:
        # Parallelize the plotting by dividing into chunks and processes
        dss = chunks_dataset(dset, chunks_size)
        plot_files_param = partial(plot_files, **args)
        p = Pool(10)
        p.map(plot_files_param, dss)


def plot_files(dss, **args):
    first = True
    for time_sel in dss.time:
        data = dss.sel(time=time_sel)
        time, run, cum_hour = get_time_run_cum(data)
        # Build the name of the output image
        filename = subfolder_images[projection] + '/' + variable_name + '_%s.png' % cum_hour

        cs = args['ax'].contourf(args['x'], args['y'],
                                 data['TMAX_2M'],
                                 extend='both',
                                 cmap=args['cmap'],
                                 levels=args['levels_t2m'])

        # plot every -th element
        if projection == "euratl":
            density = 23
        elif projection == "it":
            density = 5
        elif projection == "de":
            density = 4
        elif projection == 'south_it':
            density = 3
        elif projection == 'spain_france':
            density = 8
        else:
            density = 10

        vals = add_vals_on_map(args['ax'], projection,
                               data['TMAX_2M'], args['levels_t2m'],
                               cmap=args['cmap'],
                               density=density)

        an_fc = annotation_forecast(args['ax'], time)
        an_var = annotation(args['ax'], 'Maximum 2m Temperature in last 6 hours',
                            loc='lower left', fontsize=6)
        an_run = annotation_run(args['ax'], run)
        logo = add_logo_on_map(ax=args['ax'],
                                zoom=0.1, pos=(0.95, 0.08))

        if first:
            plt.colorbar(cs, orientation='horizontal', label='Temperature [C]',
                         pad=0.03, fraction=0.04)

        if debug:
            plt.show(block=True)
        else:
            plt.savefig(filename, **options_savefig)        

        remove_collections([cs, an_fc, an_var, an_run, vals, logo])

        first = False


if __name__ == "__main__":
    import time
    start_time=time.time()
    main()
    elapsed_time=time.time()-start_time
    print_message("script took " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
