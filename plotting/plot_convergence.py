import numpy as np
from multiprocessing import Pool
from functools import partial
from utils import *
import sys
from computations import compute_convergence

debug = False
if not debug:
    import matplotlib
    matplotlib.use('Agg')

import matplotlib.pyplot as plt

# The one employed for the figure name when exported
variable_name = 'conv'

print_message('Starting script to plot ' + variable_name)

# Get the projection as system argument from the call so that we can
# span multiple instances of this script outside
if not sys.argv[1:]:
    print_message(
        'Projection not defined, falling back to default (euratl)')
    projection = 'euratl'
else:
    projection = sys.argv[1]


def main():
    dset = read_dataset(variables=['V_10M', 'U_10M'],
                        projection=projection)

    # We could move this computation in the projection part to make
    # it fast, but this would mean that we have to do it many times...
    dset = compute_convergence(dset, uvar='U_10M', vvar='V_10M')

    levels_conv = np.linspace(-0.0005, 0.0005, 21)
    cmap = plt.get_cmap('BrBG')

    _ = plt.figure(figsize=(figsize_x, figsize_y))
    ax = plt.gca()
    # Get coordinates from dataset
    m, x, y = get_projection(dset, projection, labels=True)

    dset = dset.drop(['lon', 'lat']).load()

    # All the arguments that need to be passed to the plotting function
    args = dict(x=x, y=y, ax=ax, cmap=cmap,
                levels_conv=levels_conv,
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
        time, run, cum_hour = get_time_run_cum(data)
        # Build the name of the output image
        filename = subfolder_images[projection] + '/' + variable_name + '_%s.png' % cum_hour

        cs = args['ax'].contourf(args['x'], args['y'],
                                 data['conv'],
                                 extend='both',
                                 cmap=args['cmap'],
                                 levels=args['levels_conv'])

        # We need to reduce the number of points before plotting the vectors,
        # these values work pretty well
        if projection == 'euratl':
            density = 25
            scale = None
        else:
            density = 5
            scale = 2e2
        cv = args['ax'].quiver(args['x'][::density, ::density],
                               args['y'][::density, ::density],
                               data['10u'][::density, ::density],
                               data['10v'][::density, ::density],
                               scale=scale,
                               alpha=0.7, color='gray')

        an_fc = annotation_forecast(args['ax'], time)
        an_var = annotation(args['ax'], 'Convergence [' + str(data['conv'].units) + ']',
                            loc='lower left', fontsize=6)
        an_run = annotation_run(args['ax'], run)
        logo = add_logo_on_map(ax=args['ax'],
                                zoom=0.1, pos=(0.95, 0.08))

        if first:
            plt.colorbar(cs, orientation='horizontal', label='Convergence [' + str(
                data['conv'].units) + ']', pad=0.035, fraction=0.035, format='%.0e')

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
